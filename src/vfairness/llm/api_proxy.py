"""
LLM API abstraction layer for vfairness.

Provides a unified interface for communicating with OpenAI-compatible,
Anthropic, and custom LLM API endpoints. Handles authentication,
retries, and response normalization.

Supported API Formats:
    - openai: OpenAI Chat Completions API (also works with vLLM, Ollama)
    - anthropic: Anthropic Messages API
    - custom: Raw POST with prompt in request body
"""

import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import requests
from urllib3.util.retry import Retry

from vfairness.net.egress import GuardedSession, SSRFError, validate_endpoint

logger = logging.getLogger(__name__)


_DEFAULT_TIMEOUT = 120  # seconds
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BACKOFF_FACTOR = 1.0


@dataclass
class _ApiConfig:
    """Internal configuration for API connections."""

    endpoint_url: str
    api_format: str
    auth_token: Optional[str]
    model_name: Optional[str]
    timeout: int
    max_retries: int


class LLMApiProxy:
    """
    Unified proxy for LLM API endpoints.

    Wraps OpenAI-compatible, Anthropic, and custom APIs behind a single
    interface for prompt submission and batch execution.

    Args:
        endpoint_url: Base URL of the LLM API endpoint.
        api_format: API wire format. One of 'openai', 'anthropic', 'custom'.
        auth_token: Bearer token or API key for authentication.
        model_name: Model identifier to include in requests.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retry attempts on transient failures.

    Example:
        >>> proxy = LLMApiProxy(
        ...     endpoint_url="https://api.openai.com/v1/chat/completions",
        ...     api_format="openai",
        ...     auth_token="sk-...",
        ...     model_name="gpt-4",
        ... )
        >>> result = proxy.send_prompt("What is fairness in AI?")
        >>> print(result["text"])
    """

    _SUPPORTED_FORMATS = ("openai", "anthropic", "custom")

    def __init__(
        self,
        endpoint_url: str,
        api_format: str = "openai",
        auth_token: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: int = _DEFAULT_TIMEOUT,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        allow_loopback: bool = False,
    ) -> None:
        if not endpoint_url or not isinstance(endpoint_url, str):
            raise ValueError("endpoint_url must be a non-empty string")
        if not endpoint_url.startswith(("http://", "https://")):
            raise ValueError(
                f"endpoint_url must start with http:// or https://, got '{endpoint_url}'"
            )
        # SSRF guard: refuse a non-public endpoint (loopback / private / link-local /
        # cloud-metadata) and capture the vetted IP so the session can pin to it against
        # DNS rebinding. allow_http mirrors the check just above; the IP rule is the guard.
        # allow_loopback is an explicit caller opt-in for a model server on this machine
        # (Ollama, vLLM, a test stub). It unlocks loopback ONLY: RFC1918, link-local and
        # the cloud-metadata addresses stay refused. Never set it from an untrusted URL.
        try:
            self._vetted_host, _vetted_port, self._vetted_ips = validate_endpoint(
                endpoint_url, allow_http=True, allow_loopback=allow_loopback
            )
        except SSRFError as exc:
            raise ValueError(f"endpoint_url refused by the egress guard: {exc}")
        # Kept for the session below: the guard must re-run on redirect hops with the
        # SAME loopback posture as construction, so a 3xx cannot reach a host the
        # __init__ check would have refused.
        self._allow_loopback = allow_loopback
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")
        if api_format not in self._SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported api_format '{api_format}'. Must be one of {self._SUPPORTED_FORMATS}"
            )

        self._config = _ApiConfig(
            endpoint_url=endpoint_url.rstrip("/"),
            api_format=api_format,
            auth_token=auth_token,
            model_name=model_name,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._session = self._build_session()
        logger.info(
            "LLMApiProxy initialized: endpoint=%s, format=%s, model=%s",
            self._config.endpoint_url,
            api_format,
            model_name,
        )

    def _build_session(self) -> requests.Session:
        """Build a guarded requests session with retry logic.

        A GuardedSession re-runs the SSRF guard on EVERY hop, including redirect
        targets: it pins each request to the vetted IP (SNI/cert stay on the hostname,
        defeating same-host DNS rebinding) AND re-runs validate_endpoint() on any 3xx
        Location before following it. The plain requests default (follow redirects,
        validate only the first URL) let a single 302 to a private / loopback /
        link-local / cloud-metadata host defeat the guard entirely; see
        vfairness.net.egress.GuardedSession.
        """
        retry_strategy = Retry(
            total=self._config.max_retries,
            backoff_factor=_DEFAULT_BACKOFF_FACTOR,
            status_forcelist=[408, 429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"],
        )
        # allow_http mirrors the __init__ validation (on-prem/self-hosted cleartext
        # endpoints); allow_loopback carries the same opt-in so a redirect cannot reach
        # a host the construction-time check would have refused.
        return GuardedSession(
            allow_http=True,
            allow_loopback=self._allow_loopback,
            max_retries=retry_strategy,
        )

    def _build_headers(self) -> Dict[str, str]:
        """Build request headers based on API format."""
        headers = {"Content-Type": "application/json"}
        token = self._config.auth_token

        if self._config.api_format == "anthropic":
            if token:
                headers["x-api-key"] = token
            headers["anthropic-version"] = "2023-06-01"
        elif token:
            headers["Authorization"] = f"Bearer {token}"

        return headers

    def _build_payload(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Build the request payload for the configured API format.

        LF-04 (2026-09-09): ``top_p`` and ``seed`` travel with the request when
        given. A fairness run is only comparable to the deployed system when
        it samples the way the deployed system samples; the Navigator used to
        show temperature and top-p controls that reached no request at all.
        ``seed`` is sent on the OpenAI-compatible and custom formats only; the
        Anthropic messages API has no seed parameter and it is omitted there
        rather than sent to be rejected.
        """
        fmt = self._config.api_format

        if fmt == "openai":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            payload: Dict[str, Any] = {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if top_p is not None:
                payload["top_p"] = top_p
            if seed is not None:
                payload["seed"] = seed
            if self._config.model_name:
                payload["model"] = self._config.model_name
            return payload

        if fmt == "anthropic":
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if top_p is not None:
                payload["top_p"] = top_p
            if system_prompt:
                payload["system"] = system_prompt
            if self._config.model_name:
                payload["model"] = self._config.model_name
            return payload

        # custom format: flat body
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if top_p is not None:
            payload["top_p"] = top_p
        if seed is not None:
            payload["seed"] = seed
        if system_prompt:
            payload["system_prompt"] = system_prompt
        if self._config.model_name:
            payload["model"] = self._config.model_name
        return payload

    def _extract_text(self, response_json: Any) -> tuple[str, Optional[str]]:
        """The generated text, or the named reason no text could be read.

        Returns ``(text, unavailable_reason)``. ``unavailable_reason`` is
        ``None`` only when a real string was read out of the field the
        configured format puts the generation in, and that INCLUDES the empty
        string a model genuinely returns when it generates nothing. Anything
        else is "this body carried no generation", which is named rather than
        turned into text.

        Three states, never two (BGL3, 2026-09-27). Both halves of the old
        version collapsed "the endpoint sent no generation" into "the
        generation was this":

        * custom format, body ``{"error": "quota exceeded", "code": 429}``:
          the final ``str(response_json)`` made the ERROR BODY the answer.
          Measured on the real method, ``send_prompt`` returned
          ``text="{'error': 'quota exceeded', 'code': 429}"``, 40 characters,
          and logged nothing, because the text was not empty. Every scorer
          downstream then read that repr as a model answer: five words of
          response length, a measured sentiment, a measured framing.
        * openai format, body ``{}`` or ``{"error": {...}}``: ``.get("text",
          "")`` returned ``""`` and the returned dict carried no marker at
          all, so an unparseable 200 was indistinguishable from a model that
          answered nothing. ``""`` scores a clean 0.0 on every presence
          scorer (toxicity, stereotype, representation, refusal) and reads as
          perfect stability across runs in ``nondeterminism``.
        * openai format, body ``{"choices": [{"message": {"content": null}}]}``
          (what a content filter returns): ``content`` came back ``None``,
          the ``except (KeyError, IndexError)`` did not catch it, and
          ``len(text)`` in a debug log line raised ``TypeError: object of
          type 'NoneType' has no len()`` from inside ``send_prompt``. A
          non-string is now "no text" like any other.

        The caller sees the reason on the returned dict as
        ``text_unavailable_reason``; ``send_batch`` carries it too.
        """
        fmt = self._config.api_format

        if not isinstance(response_json, dict):
            return "", "response_body_not_an_object"

        if fmt == "openai":
            try:
                content = response_json["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                content = None
            if isinstance(content, str):
                return content, None
            fallback = response_json.get("text")
            if isinstance(fallback, str):
                return fallback, None
            return "", "no_text_in_response"

        if fmt == "anthropic":
            try:
                content = response_json["content"][0]["text"]
            except (KeyError, IndexError, TypeError):
                content = None
            if isinstance(content, str):
                return content, None
            fallback = response_json.get("text")
            if isinstance(fallback, str):
                return fallback, None
            return "", "no_text_in_response"

        # custom: no wire contract, so the common spellings are tried. A bare
        # number under one of them is odd but unambiguous and is coerced; a
        # dict or a list is a STRUCTURE, and its repr is not a generation.
        for key in ("text", "output", "response", "generated_text"):
            if key not in response_json:
                continue
            value = response_json[key]
            if isinstance(value, str):
                return value, None
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                # BGL5 A-llm-3 (2026-09-27). A NON-FINITE NUMBER IS THE ABSENCE
                # OF A NUMBER, NOT A GENERATION. The guard tested isinstance
                # only, and json.loads accepts the bare literals: measured,
                # json.loads('{"text": NaN}') -> {'text': nan}, and
                # send_prompt on that body returned
                #   {'text': 'nan', 'latency_ms': 0.0, 'token_count': -1,
                #    'reported_model': None, 'text_unavailable_reason': None}
                # so three characters every presence scorer reads as a model
                # answer, with the reason field saying the text was genuine.
                # Same for Infinity ('inf'). After: both fall through to
                # text='' with text_unavailable_reason 'no_text_in_response',
                # and a finite number under the key is still coerced
                # ({'text': 42} -> '42', reason None), which is the
                # over-correction control.
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                return str(value), None
        return "", "no_text_in_response"

    def _extract_reported_model(self, response_json: Dict[str, Any]) -> Optional[str]:
        """
        The model name the ENDPOINT reported on its own response, or None.

        LF-14. This is not the model that was requested. A provider can serve a
        different build under a stable name, a wrapper can route to a model
        nobody named, and a behavioural finding is a statement about the model
        it was measured on. So the endpoint's own echo is recorded as its own
        fact, beside the configured name, and the two are never merged.

        NEVER FALLS BACK TO THE REQUESTED MODEL. Returning the request's own
        `model` parameter here would manufacture agreement out of nothing: the
        caller would see a reported name identical to the configured one and
        read it as the endpoint confirming what answered. None means the
        endpoint said nothing, and that is a limit on the evidence, not a match.

        Both the OpenAI and the Anthropic response shapes carry a top-level
        `model`. The custom format has no contract, so a few common spellings
        are tried and anything else is None rather than a guess.
        """
        if not isinstance(response_json, dict):
            return None

        def _clean(value: Any) -> Optional[str]:
            if not isinstance(value, str):
                return None
            trimmed = value.strip()
            return trimmed or None

        reported = _clean(response_json.get("model"))
        if reported is not None:
            return reported

        # Some gateways nest it. Only names, never ids: an id identifies the
        # request, not the model that served it.
        for key in ("model_name", "model_id", "served_model_name"):
            reported = _clean(response_json.get(key))
            if reported is not None:
                return reported

        nested = response_json.get("metadata")
        if isinstance(nested, dict):
            return _clean(nested.get("model"))
        return None

    def _extract_token_count(self, response_json: Dict[str, Any]) -> int:
        """Extract token count from API response, or return -1 if unavailable.

        ``-1`` is the documented "the endpoint did not say" sentinel, and it has
        to survive a body that says it badly. ``usage: null`` is a real shape
        (several gateways emit it on a streamed or filtered completion) and
        ``"completion_tokens" in None`` raised ``TypeError: argument of type
        'NoneType' is not iterable`` out of ``send_prompt``, killing the whole
        ``send_batch`` run because ``TypeError`` is not one of the exceptions it
        retries. Measured 2026-09-27 on body
        ``{"choices": [{"message": {"content": "hello"}}], "usage": None}``.
        """
        if not isinstance(response_json, dict):
            return -1
        usage = response_json.get("usage")
        if not isinstance(usage, dict):
            return -1
        for key in ("completion_tokens", "output_tokens"):
            if key in usage:
                try:
                    return int(usage[key])
                except (TypeError, ValueError):
                    # A count that cannot be read is unknown, not zero.
                    return -1
        return -1

    def send_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Send a single prompt to the LLM and return the response.

        Args:
            prompt: The user prompt text.
            system_prompt: Optional system-level instruction.
            temperature: Sampling temperature (0 = deterministic).
            max_tokens: Maximum tokens to generate.
            top_p: Nucleus-sampling probability mass in (0, 1]; omitted from
                the request when ``None``.
            seed: Sampling seed for providers that honour one; omitted when
                ``None`` and never sent on the Anthropic format.

        Returns:
            Dict with keys:
                - text (str): Generated response text.
                - latency_ms (float): Round-trip latency in milliseconds.
                - token_count (int): Number of output tokens (-1 if unknown).
                - reported_model (str | None): The model name the ENDPOINT put on
                  its response. None when it reported none. This is never
                  filled in from the requested model name: a matching name the
                  caller supplied itself would be evidence of nothing (LF-14).
                - text_unavailable_reason (str | None): None when ``text`` is
                  what the endpoint generated, INCLUDING a genuinely empty
                  generation. A string when NO text could be read out of the
                  body ('no_text_in_response', 'response_body_not_an_object'),
                  in which case ``text`` is ``""`` and means "nothing was
                  read", not "the model said nothing" (BGL3). Never read
                  ``text`` alone to decide a model returned nothing.

        Raises:
            requests.HTTPError: On non-retryable HTTP errors.
            requests.ConnectionError: If the endpoint is unreachable.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")
        if top_p is not None and not (0.0 < float(top_p) <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError(f"seed must be an int, got {type(seed).__name__}")

        logger.debug(
            "send_prompt: prompt_len=%d, temperature=%.2f, top_p=%s, seed=%s",
            len(prompt),
            temperature,
            top_p,
            seed,
        )
        payload = self._build_payload(
            prompt, system_prompt, temperature, max_tokens, top_p=top_p, seed=seed
        )
        headers = self._build_headers()

        t0 = time.perf_counter()
        resp = self._session.post(
            self._config.endpoint_url,
            json=payload,
            headers=headers,
            timeout=self._config.timeout,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        resp.raise_for_status()
        body = resp.json()

        text, text_unavailable_reason = self._extract_text(body)
        body_keys = list(body.keys())[:5] if isinstance(body, dict) else type(body).__name__
        if text_unavailable_reason is not None:
            # NOT the same warning as an empty generation, and deliberately
            # louder: nothing was read here, so the "" below is the absence of a
            # measurement rather than one.
            logger.warning(
                "No text could be read from the API response (format=%s, reason=%s). "
                "status=%d, body_keys=%s, endpoint=%s. Returning text='' with "
                "text_unavailable_reason set: this is NOT an empty generation.",
                self._config.api_format,
                text_unavailable_reason,
                resp.status_code,
                body_keys,
                self._config.endpoint_url[:60],
            )
        elif not text.strip():
            logger.warning(
                "Empty text extracted from API response. status=%d, body_keys=%s, endpoint=%s",
                resp.status_code,
                body_keys,
                self._config.endpoint_url[:60],
            )

        logger.debug(
            "send_prompt: latency=%.1fms, status=%d, text_len=%d",
            latency_ms,
            resp.status_code,
            len(text),
        )
        return {
            "text": text,
            "latency_ms": round(latency_ms, 2),
            "token_count": self._extract_token_count(body),
            "reported_model": self._extract_reported_model(body),
            "text_unavailable_reason": text_unavailable_reason,
        }

    def send_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        n_runs: int = 25,
        temperature: float = 0.0,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Send multiple prompts, each repeated n_runs times for non-determinism.

        For each prompt in *prompts*, issues *n_runs* identical requests and
        collects all responses. This is essential for distinguishing systematic
        bias from stochastic variance.

        Args:
            prompts: List of prompt strings to evaluate.
            system_prompt: Optional system-level instruction applied to all.
            n_runs: Number of repeated runs per prompt (default 25).
            temperature: Sampling temperature.

        Returns:
            List of dicts, one per prompt, each containing:
                - prompt (str): The original prompt.
                - responses (list[dict]): List of n_runs response dicts (each
                  with text, latency_ms, token_count, reported_model and
                  text_unavailable_reason; plus ``error`` on a run that never
                  got a response at all).

            A response whose ``text`` is ``""`` is only a model that generated
            nothing when BOTH ``text_unavailable_reason`` and ``error`` are
            absent. Otherwise no measurement was taken on that run, and
            treating the blank as an answer makes a dead endpoint look like a
            perfectly stable, perfectly non-toxic one (BGL3).
        """
        if n_runs < 1:
            raise ValueError("n_runs must be >= 1")
        if not prompts:
            # Empty list in, empty list out, said out loud. "No findings" and
            # "nothing was examined" are the same [] to a caller that only
            # counts rows, and every metric computed over zero prompts is a
            # metric over nothing.
            logger.warning(
                "send_batch: no prompts supplied. Nothing was sent and no responses are "
                "being returned; this is not a run in which the model produced nothing."
            )
            return []

        logger.info(
            "send_batch: %d prompts x %d runs, temperature=%.2f",
            len(prompts),
            n_runs,
            temperature,
        )
        total = len(prompts) * n_runs
        completed = 0
        results: List[Dict[str, Any]] = []
        for prompt in prompts:
            responses = []
            for run_idx in range(n_runs):
                import time as _time

                last_err: Optional[Exception] = None
                for attempt in range(3):  # Up to 3 attempts per call
                    try:
                        resp = self.send_prompt(
                            prompt,
                            system_prompt=system_prompt,
                            temperature=temperature,
                            top_p=top_p,
                            seed=seed,
                        )
                        responses.append(resp)
                        last_err = None
                        break
                    except requests.exceptions.HTTPError as exc:
                        last_err = exc
                        if (
                            hasattr(exc, "response")
                            and exc.response is not None
                            and exc.response.status_code == 429
                        ):
                            wait = min(5 * (attempt + 1), 15)  # 5s, 10s, 15s
                            logger.warning(
                                "Rate limited (429), waiting %ds before retry %d/3",
                                wait,
                                attempt + 1,
                            )
                            _time.sleep(wait)
                            continue
                        break  # Non-429 HTTP error, don't retry
                    except (requests.RequestException, ValueError) as exc:
                        last_err = exc
                        err_str = str(exc)
                        if ("429" in err_str or "rate" in err_str.lower()) and attempt < 2:
                            wait = min(5 * (attempt + 1), 15)
                            logger.warning(
                                "Rate limited (429 in exception), waiting %ds before retry %d/3",
                                wait,
                                attempt + 1,
                            )
                            _time.sleep(wait)
                            continue
                        if "Connection" in err_str and attempt < 2:
                            logger.warning(
                                "Connection error, waiting 3s before retry %d/3: %s",
                                attempt + 1,
                                exc,
                            )
                            _time.sleep(3)
                            continue
                        break
                if last_err is not None:
                    # Log the prompt length, not its content: prompts can contain PII.
                    logger.warning(
                        "Request failed after retries (prompt %d chars): %s", len(prompt), last_err
                    )
                    responses.append(
                        {
                            "text": "",
                            "latency_ms": -1.0,
                            "token_count": -1,
                            # No response arrived, so nothing was reported. This
                            # is the same None as "the endpoint reported no
                            # model", and the `error` key beside it is what
                            # tells the two apart.
                            "reported_model": None,
                            # Same reason field the successful path carries, so
                            # one check over a batch finds every run that
                            # measured nothing, whatever the cause.
                            "text_unavailable_reason": "request_failed",
                            "error": str(last_err),
                        }
                    )
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, total)
            # Disclosure at the level a caller reads: when NO run for a prompt
            # produced readable text, the per-response reasons are easy to miss
            # and the n_runs blanks that remain are byte-identical, which
            # nondeterminism reads as perfect consistency and every presence
            # scorer reads as 0.0. Prompt LENGTH only, never its content: a
            # prompt can carry PII.
            unreadable = [
                r for r in responses if r.get("text_unavailable_reason") or r.get("error")
            ]
            if responses and len(unreadable) == len(responses):
                logger.warning(
                    "No readable text from any of the %d run(s) for a %d-char prompt "
                    "(reasons: %s). Every response carries text='' with a reason: that is "
                    "an absent measurement, not a model that answered nothing.",
                    len(responses),
                    len(prompt),
                    sorted({str(r.get("text_unavailable_reason")) for r in unreadable}),
                )
            results.append({"prompt": prompt, "responses": responses})
        return results

    def test_connection(self) -> Dict[str, Any]:
        """
        Perform a minimal health check against the configured endpoint.

        Sends a trivial prompt and verifies a valid response is returned.

        Returns:
            Dict with keys:
                - ok (bool): Whether the connection succeeded AND text came
                  back that could be read.
                - latency_ms (float): Round-trip latency. Negative means no
                  round trip was measured at all.
                - error (str | None): Why this is not an ``ok`` connection.
                  None ONLY when ``ok`` is True: a False with no reason is not
                  a finding, it is a dropped one.
                - reported_model (str | None): The model name the endpoint put
                  on its response, or None when it reported none (LF-14).
                - text_unavailable_reason (str | None): The reason
                  ``send_prompt`` gave for reading no text out of the body
                  ('no_text_in_response', 'response_body_not_an_object'), or
                  'request_failed' when nothing arrived. None means text WAS
                  read, including a genuinely empty generation. This is the
                  field that separates "the endpoint answered and nothing
                  could be read" from "the model generated nothing".
        """
        logger.info("test_connection: endpoint=%s", self._config.endpoint_url)
        try:
            result = self.send_prompt("Say 'ok'.", max_tokens=8)
            # BGL5 A-llm-3 (2026-09-27). THREE STATES, AND THE LOG MAY NOT
            # ANSWER BEFORE THE RECORD DOES. Measured on five 200 responses
            # that carry no readable generation (custom {'error': 'quota
            # exceeded', 'code': 429}; openai {}; openai {'choices':
            # [{'message': {'content': None}}]}, which is what a content filter
            # returns; a body that is not an object; and a genuinely empty
            # generation), every one of them returned the byte-identical record
            #
            #   {'ok': False, 'latency_ms': 0.0, 'error': None,
            #    'reported_model': None}
            #
            # while the INFO line said "test_connection: ok=True" for all five,
            # because it was formatted before ok was computed. So a body nothing
            # could be read from was indistinguishable from a model that
            # generated nothing, both were reported as a failure with no reason
            # at all, and the reason existed one layer down on the SAME call:
            # send_prompt had text_unavailable_reason='no_text_in_response' in
            # hand and this boundary dropped it.
            #
            # After: the four unreadable bodies return ok False with
            # text_unavailable_reason 'no_text_in_response' /
            # 'response_body_not_an_object' and an error sentence naming it; the
            # genuinely empty generation returns ok False, reason None and its
            # own distinct sentence; and the log reports the ok that is
            # returned. The measuring control is unchanged: a body carrying
            # 'ok' still gives ok True with error None and reason None.
            reason = result.get("text_unavailable_reason")
            text = result.get("text") or ""
            ok = reason is None and bool(text)
            if reason is not None:
                error = (
                    f"the endpoint answered but no text could be read out of its body "
                    f"({reason}). COULD NOT CHECK: this is neither a model that "
                    f"generated nothing nor a transport failure."
                )
            elif not ok:
                error = (
                    "the endpoint answered and the body parsed, but the generation was "
                    "empty, so there was nothing to check."
                )
            else:
                error = None
            logger.info(
                "test_connection: ok=%s, latency=%.1fms, text_unavailable_reason=%s",
                ok,
                result["latency_ms"],
                reason,
            )
            return {
                "ok": ok,
                "latency_ms": result["latency_ms"],
                "error": error,
                "reported_model": result.get("reported_model"),
                "text_unavailable_reason": reason,
            }
        except Exception as exc:
            logger.error("test_connection failed: %s", exc)
            return {
                "ok": False,
                "latency_ms": -1.0,
                "error": str(exc),
                "reported_model": None,
                # Same spelling send_batch uses for a request that never
                # landed, so one check over either surface finds every record
                # that measured nothing, whatever the cause.
                "text_unavailable_reason": "request_failed",
            }
