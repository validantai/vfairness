# Security Policy

vfairness produces evidence used in fairness audits, so we take security seriously.

## Reporting a vulnerability

Please report suspected vulnerabilities privately, not in a public issue.

- Use GitHub's private vulnerability reporting on this repository ("Report a vulnerability" under the Security tab), or
- email security@validant.ai.

Include a description, affected version, and, if possible, a minimal reproduction. We aim to acknowledge a report within 5 working days and to agree a disclosure timeline with you.

## Supported versions

vfairness is pre-1.0. Security fixes land on the latest release line. Pin a version in your requirements and upgrade promptly when a fix is published.

## Scope and safe handling

- The library never makes network calls on the metric-computation path and ships no telemetry by default.
- Loading untrusted model files can be dangerous. Do not load pickled models (`pickle`, `joblib`, unrestricted `torch.load`) from untrusted sources; prefer a non-executable serialization format.
- Outbound model-endpoint calls made through `LLMApiProxy` go through the `vfairness.net` egress guard. It resolves the target, refuses non-public destinations (loopback, RFC1918, link-local, cloud metadata), and pins the vetted IP for the session so DNS rebinding cannot redirect it. A model server on the local machine is reachable only via an explicit `allow_loopback=True` opt-in, which unlocks loopback and nothing else. The Pulse LLM probe builds an `LLMApiProxy` and is therefore covered.
- **Every call site that takes a caller-supplied URL is now guarded.** The three this section used to name as unguarded were wired up on 2026-08-27 and this text was not updated: the LLM judge scorer (`vfairness.llm.scorers`) routes through `guarded_post`, the groundedness judge (`vfairness.validity.judge`) through a `GuardedSession` so redirects are re-vetted hop by hop, and the XAI sidecar client (`vfairness.xai.sidecar_cli`) calls `validate_endpoint` before any credential is attached and then posts through `guarded_post`. Do not read the old advice to pre-validate those URLs yourself as a sign they are unprotected; it is simply no longer necessary.
- **Two outbound calls do not use the guard, and neither takes a URL from you.** `operations.pulse.task_handlers` fetches a first-party signed artifact URL with its own defences (https, a certifi CA bundle, a size cap), and `preprocessing.bias_detection.geographic_data` fetches a fixed public API whose URL is built from a module constant plus a whitelisted city id. They are listed here because a security note that says "universal" without qualification would be the same kind of overstatement this bullet just corrected.
- Every push and pull request touching the library runs Bandit (`-ll`, medium and high severity) over `src` and a `pip-audit --strict` dependency audit; the audit also runs weekly on a schedule.

## Maintainer

Glinz & Company GmbH (validant.ai).
