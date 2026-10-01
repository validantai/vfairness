"""
pgmq consumer loop.

This is a deliberately minimal runner. The platform-side spec (internal,
not published with this library) leaves pgmq provisioning + the
``xai_jobs`` queue creation to whoever operates the worker host. This
module assumes the queue already exists; it polls,
dispatches and updates ``xai_jobs.status`` so the frontend Realtime
subscription sees live progress.

Run with::

    python -m vfairness.xai.worker

or instantiate :class:`WorkerLoop` and call ``run()`` from a host
program.
"""

from __future__ import annotations

import logging
import os
import signal
import time
from dataclasses import dataclass, field
from typing import Any

from ..explainers import (
    get_explainer,
    route_explainer,
)
from ..storage import SupabaseWriter

log = logging.getLogger("vfairness.xai.worker")

#: Process exit status for a run that finished without ever learning whether
#: there was work: every poll failed, or no poll happened at all.
#:
#: BGL5 A-xai-1, 2026-09-27. ``WorkerLoop.run`` grew a three-state summary and
#: ``main`` dropped it: measured with a stubbed client whose every pgmq read
#: raised RuntimeError('relation "pgmq.q_xai_jobs" does not exist'), the log said
#: "NOT ONE answered read ... this is NOT a clean stop" and the PROCESS EXIT
#: STATUS was 0. A supervisor reads the status, not the log, and
#: ``python -m vfairness.xai.worker`` is a documented invocation with its own
#: __main__.py written so deployment scripts can use it, so a could-not-check was
#: being handed over as a clean run.
#:
#: 3 rather than 1, so it cannot be confused with an unhandled exception (1) or
#: with a signal-terminated process (128+n), which is the whole point of
#: distinguishing "nothing was measured" from "something failed".
EXIT_NOTHING_MEASURED = 3

#: Process exit status for a run that DID learn there was work and then failed on
#: all of it: every loop tick that reached a dispatch raised.
#:
#: BGL6 F12, 2026-09-29. ``run`` counted POLLS only, so a run in which every
#: dispatched job raised had three ANSWERED polls and zero failed ones, and fell
#: straight into the else: "XAI worker stopped cleanly", status 0. A zero from a
#: crashed tool reads as clean. The loop's own ``except Exception`` logged each
#: crash and then the summary contradicted all three of them.
#:
#: Distinct from :data:`EXIT_NOTHING_MEASURED` because the two are different
#: states and a supervisor should be able to tell them apart: 3 means the queue
#: was never read, 4 means it was read, work came back, and none of it ran.
EXIT_WORK_ALL_FAILED = 4


@dataclass
class WorkerConfig:
    queue_name: str = "xai_jobs"
    poll_interval_s: float = 5.0
    concurrency: int = 1
    artifact_bucket: str = "xai-audit-artifacts"
    supabase_url: str | None = None
    supabase_service_role_key: str | None = None

    @classmethod
    def from_env(cls) -> "WorkerConfig":
        return cls(
            queue_name=os.environ.get("PGMQ_QUEUE", "xai_jobs"),
            poll_interval_s=float(os.environ.get("WORKER_POLL_INTERVAL_S", 5.0)),
            concurrency=int(os.environ.get("WORKER_CONCURRENCY", 1)),
            artifact_bucket=os.environ.get("WORKER_ARTIFACT_BUCKET", "xai-audit-artifacts"),
            supabase_url=os.environ.get("SUPABASE_URL"),
            supabase_service_role_key=os.environ.get("SUPABASE_SERVICE_ROLE_KEY"),
        )


@dataclass
class WorkerLoop:
    """Polling loop. Stops on SIGINT / SIGTERM."""

    config: WorkerConfig = field(default_factory=WorkerConfig.from_env)
    _stopped: bool = field(default=False, init=False)
    _writer: SupabaseWriter | None = field(default=None, init=False)
    #: Polls that got an ANSWER from pgmq (a message, or a queue that is provably
    #: empty) and polls whose answer could not be obtained. Two counters rather
    #: than one, because "no work" and "could not tell whether there is work" are
    #: different states and ``_pop_one`` has to return None for both.
    _polls_answered: int = field(default=0, init=False)
    _polls_failed: int = field(default=0, init=False)
    #: Loop ticks that reached a DISPATCH, and how they ended. A poll is not a
    #: tick: an answered poll of an empty queue dispatches nothing, so it counts in
    #: neither. Separate from the poll counters because "the queue answered" says
    #: nothing about whether the work it handed over ran. BGL6 F12, 2026-09-29: the
    #: summary read the poll counters alone, so a run whose every dispatch raised
    #: reported "stopped cleanly" with status 0.
    #: G12, 2026-09-30. THE F12 GUARD COULD NOT FIRE THROUGH THE REAL CODE PATH.
    #: The counters above were maintained by the LOOP, around its own
    #: ``except Exception``, and ``_dispatch`` catches every exception a job
    #: raises (it has to: it writes ``status="failed"`` on the job row). So the
    #: only way to reach ``_ticks_failed`` was a ``_dispatch`` that RAISES, which
    #: is how F12's own test drives it and is not something the shipped
    #: ``_dispatch`` ever does. Measured with the real ``_dispatch`` over three
    #: valid messages, and ``_execute`` raising the NotImplementedError it raises
    #: on EVERY job today:
    #:     3x 'ERROR job N failed', 3x update_job_progress(status='failed'),
    #:     then 'INFO XAI worker stopped cleanly', ticks (3 completed, 0 failed),
    #:     status 0
    #: Every job in the queue was recorded FAILED in the database and the
    #: supervisor was handed a clean run. The invalid-message branch of
    #: ``_dispatch`` did the same from the other side: it logs and RETURNS, so a
    #: run of nothing but unroutable messages also counted three completed ticks.
    #: The outcome of a job is only known inside ``_dispatch``, so that is where
    #: it is now recorded; see :meth:`_record_tick`.
    #: ``_ticks_invalid`` is its own counter because a message carrying no
    #: runnable job is a different state from a job that ran and raised: the first
    #: is a producer or queue-shape problem, the second is the explainer.
    _ticks_completed: int = field(default=0, init=False)
    _ticks_failed: int = field(default=0, init=False)
    _ticks_invalid: int = field(default=0, init=False)

    def _record_tick(self, outcome: str) -> None:
        """Record how one dispatched tick ended: completed / failed / invalid."""
        if outcome == "completed":
            self._ticks_completed += 1
        elif outcome == "failed":
            self._ticks_failed += 1
        else:
            self._ticks_invalid += 1

    @property
    def _ticks_recorded(self) -> int:
        return self._ticks_completed + self._ticks_failed + self._ticks_invalid

    def stop(self, *_: Any) -> None:
        log.info("worker stop requested")
        self._stopped = True

    def run(self) -> int:
        """Poll until stopped, and RETURN the exit status the run earned.

        ``0`` only when this run learned whether there was work: every poll
        answered, or some answered and some did not (degraded, and warned about,
        but measured). :data:`EXIT_NOTHING_MEASURED` when it learned nothing,
        which covers both the queue that never answered and the worker that never
        asked. BGL5 A-xai-1, 2026-09-27: this returned None on all outcomes, so
        the three states existed only in the log and ``main`` handed the
        supervisor an exit status of 0 for a worker whose every read had failed.
        """
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        self._writer = SupabaseWriter(
            url=self.config.supabase_url,
            service_role_key=self.config.supabase_service_role_key,
        )
        log.info(
            "XAI worker started; queue=%s, poll=%.1fs",
            self.config.queue_name,
            self.config.poll_interval_s,
        )
        while not self._stopped:
            try:
                msg = self._pop_one()
                if msg is None:
                    time.sleep(self.config.poll_interval_s)
                    continue
                recorded_before = self._ticks_recorded
                self._dispatch(msg)
                if self._ticks_recorded == recorded_before:
                    # ``_dispatch`` recorded no outcome for this message. That is
                    # a host program's own override, or a test double; keep the
                    # pre-G12 behaviour and count the tick as completed, so a
                    # replacement dispatcher is not accused of failing.
                    self._ticks_completed += 1
            except Exception:
                self._ticks_failed += 1
                log.exception("worker loop tick failed; sleeping before retry")
                time.sleep(self.config.poll_interval_s)
        # Three states at the only surface a supervisor reads, never two.
        # Measured 2026-09-27 with a stubbed client, three ticks each: a queue
        # that answered "empty" and a queue whose read raised
        # RuntimeError('relation "pgmq.q_xai_jobs" does not exist') on EVERY tick
        # produced the same final line, "XAI worker stopped cleanly". The second
        # worker never learned whether there was work; the per-tick warning said
        # so and the summary contradicted it.
        #
        # THE FOURTH STATE, BGL5 A-xai-1, 2026-09-27. The branch below tested
        # _polls_failed alone, so a run with BOTH counters at zero fell through to
        # the else. Measured with a stubbed client, stop() called before run(),
        # which is SIGTERM arriving during startup:
        #     before -> 'INFO XAI worker stopped cleanly', polls_answered 0,
        #               polls_failed 0, run() returned None, process exit status 0
        #     after  -> 'ERROR XAI worker stopped after 0 poll(s) of queue
        #               xai_jobs: it never asked whether there was work ...',
        #               run() returns 3, process exit status 3
        # A worker that never asked is not a worker that found nothing, and
        # "stopped cleanly" is the exact sentence this block exists to prevent.
        if not self._polls_failed and not self._polls_answered:
            log.error(
                "XAI worker stopped after 0 poll(s) of queue %s: it never asked whether "
                "there was work, so this is NOT a clean stop. It was stopped before its "
                "first poll (SIGTERM during startup, or stop() before run()).",
                self.config.queue_name,
            )
            return EXIT_NOTHING_MEASURED
        if self._polls_failed and not self._polls_answered:
            log.error(
                "XAI worker stopped after %d failed poll(s) and NOT ONE answered read of "
                "queue %s: it never learned whether there was work, so this is NOT a "
                "clean stop. Verify the pgmq schema bridge.",
                self._polls_failed,
                self.config.queue_name,
            )
            return EXIT_NOTHING_MEASURED
        if self._polls_failed:
            log.warning(
                "XAI worker stopped; %d of %d poll(s) of queue %s could not be answered, "
                "so any job queued during those polls may never have been seen.",
                self._polls_failed,
                self._polls_failed + self._polls_answered,
                self.config.queue_name,
            )
            # No return here: a partly answered poll history is a degraded MEASUREMENT
            # of the queue, and the tick verdict below is a separate question that
            # must still be reached. Before BGL6 F12 this returned 0 immediately.
        # THE WORK ITSELF, BGL6 F12, 2026-09-29. Every branch above reads the POLL
        # counters, which say only whether the queue answered. A run whose every
        # dispatch raised has answered polls and no failed ones, so it reached the
        # else and logged "XAI worker stopped cleanly" with status 0 after three
        # ERROR records reading "worker loop tick failed". Measured with a stubbed
        # client and a _dispatch that raises RuntimeError on every message:
        #     before -> 3 ERROR ticks, 'INFO XAI worker stopped cleanly', status 0
        #     after  -> 3 ERROR ticks, 'ERROR XAI worker stopped after 3 tick(s) of
        #               work and NOT ONE of them completed', status 4
        # An empty result from a crashed tool is the loudest outcome, not the
        # mildest, and "stopped cleanly" is the exact sentence this block exists to
        # prevent.
        #
        # G12, 2026-09-30: both branches below read ``_ticks_failed`` ALONE, and
        # the shipped ``_dispatch`` never let an exception reach the loop, so
        # neither could fire on a real run. They now read every tick that did not
        # complete, which is the question being asked; see the ``_ticks_invalid``
        # comment on the field for the two measured runs.
        unrun = self._ticks_failed + self._ticks_invalid
        if unrun and not self._ticks_completed:
            log.error(
                "XAI worker stopped after %d tick(s) of work on queue %s and NOT ONE of "
                "them completed: %d dispatched job(s) raised and %d message(s) carried no "
                "runnable job, so this is NOT a clean stop. The jobs were seen and none of "
                "them ran.",
                unrun,
                self.config.queue_name,
                self._ticks_failed,
                self._ticks_invalid,
            )
            return EXIT_WORK_ALL_FAILED
        if unrun:
            log.warning(
                "XAI worker stopped; %d of %d dispatched tick(s) on queue %s did not run to "
                "completion (%d raised, %d carried no runnable job).",
                unrun,
                unrun + self._ticks_completed,
                self.config.queue_name,
                self._ticks_failed,
                self._ticks_invalid,
            )
            # Still 0: some job DID complete, so the worker is functioning and a
            # supervisor restarting on it would be restarting on a per-job failure.
            return 0
        if self._polls_failed:
            # Warned above. Still 0: some poll DID answer, so this run measured the
            # queue and a restart would be a restart on a transient.
            return 0
        log.info("XAI worker stopped cleanly")
        return 0

    # internals
    def _pop_one(self) -> dict[str, Any] | None:
        """Pop one ready message from pgmq.

        Uses ``pgmq.read(queue, vt, qty)`` via the Supabase RPC bridge.
        Returns the message payload or ``None`` if the queue is empty.

        ``None`` cannot separate "the queue is empty" from "the read failed", so
        the two are counted instead: ``_polls_answered`` vs ``_polls_failed``.
        ``run`` reports on them when it stops, and a host program can read them.
        """
        assert self._writer is not None
        try:
            resp = self._writer._client.rpc(
                "pgmq_read",
                {"queue_name": self.config.queue_name, "vt": 60, "qty": 1},
            ).execute()
            rows = resp.data or []
            self._polls_answered += 1
            if not rows:
                return None
            return rows[0]
        except Exception:
            self._polls_failed += 1
            log.warning(
                "pgmq_read RPC failed (%d failed poll(s) so far, %d answered); verify the "
                "pgmq schema bridge exists. This is NOT an empty queue.",
                self._polls_failed,
                self._polls_answered,
                exc_info=True,
            )
            return None

    def _dispatch(self, msg: dict[str, Any]) -> None:
        """Route a single job message to the right explainer.

        Records the tick's outcome through :meth:`_record_tick` on every path,
        because this is the only place that knows it: the ``except`` below is
        where a failing job STOPS, so nothing above can see that it failed.
        """
        assert self._writer is not None
        payload = msg.get("message", {}) or {}
        job_id = msg.get("msg_id") or payload.get("job_id")
        owner = payload.get("owner")
        if not owner or not job_id:
            # Log only identifiers/shape, never the raw message: it carries the
            # owner (auth0_sub) and job payload fields (potential PII).
            log.error(
                "invalid pgmq message (missing owner/job_id): msg_id=%r, message_keys=%s",
                msg.get("msg_id"),
                sorted((payload or {}).keys()),
            )
            self._record_tick("invalid")
            return

        try:
            self._writer.update_job_progress(job_id=str(job_id), status="running", progress=5)
            self._execute(payload, owner=owner, job_id=str(job_id))
            self._writer.update_job_progress(job_id=str(job_id), status="succeeded", progress=100)
            self._record_tick("completed")
        except Exception as exc:
            log.exception("job %s failed", job_id)
            # Recorded BEFORE the progress write, so a writer that also fails
            # cannot turn a failed job into an unrecorded one.
            self._record_tick("failed")
            self._writer.update_job_progress(
                job_id=str(job_id),
                status="failed",
                progress=0,
                message=str(exc),
            )

    def _execute(self, payload: dict[str, Any], *, owner: str, job_id: str) -> None:
        """Run the explainer that the router picks for the payload.

        Stub: this is where the host program plugs in its model loader.
        In production we look up the model + dataset through the
        platform's existing pulse-artifact subsystem and pass them in.
        """
        decision = route_explainer(
            model_type=payload.get("model_type", "blackbox"),
            goal=payload.get("goal", "global_understanding"),
            counterfactual_needed=bool(payload.get("counterfactual_needed", False)),
            background_available=bool(payload.get("background_available", False)),
            ood_risk=payload.get("ood_risk", "low"),
        )
        log.info(
            "job %s routed -> %s (component=%s)", job_id, decision.primary, decision.component_id
        )
        # Resolve the routed method to a concrete adapter. This fails
        # loudly here (NotImplementedError naming the backlog item) if the
        # router ever picks a method whose adapter has not landed: better
        # than a silent miss after the model load.
        explainer = get_explainer(decision.primary)
        log.info("job %s adapter=%s", job_id, type(explainer).__name__)
        # The actual model + data load happens here. Left as a stub for
        # operations to wire against the existing pulse-artifact loader;
        # once loaded, call explainer.explain_local / explain_global.
        raise NotImplementedError(
            "WorkerLoop._execute: host program must load the model + data "
            "via the existing pulse-artifact subsystem and call the routed "
            f"explainer ({type(explainer).__name__}). See vfairness.xai docs."
        )


def main() -> None:
    """Process entry point. Raises SystemExit when the run measured nothing.

    The exit status is the ONLY channel a supervisor or a deployment script
    reads, so run()'s could-not-check has to arrive there. Measured 2026-09-27
    before this, with a stubbed client whose every pgmq read raised
    RuntimeError('relation "pgmq.q_xai_jobs" does not exist'): the log carried
    'NOT ONE answered read of queue xai_jobs ... this is NOT a clean stop' and
    `echo $?` printed 0. After: the same run prints the same line and exits 3.

    A status of 0 is returned by RETURNING, never by raising SystemExit(0): the
    loop is replaced by a recorder in the suite, and a recorder's run() returns
    None, which is falsy and must stay a clean exit. That is also why this branches
    on the value rather than passing it to SystemExit unconditionally.

    Carries :data:`EXIT_WORK_ALL_FAILED` through the same branch, for the run that
    read the queue, got work back and crashed on all of it (BGL6 F12, 2026-09-29).
    """
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    status = WorkerLoop().run()
    if status:
        raise SystemExit(status)


if __name__ == "__main__":  # pragma: no cover
    main()
