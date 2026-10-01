"""
vfairness.xai.worker
====================

pgmq consumer loop that picks XAI compute jobs off the queue, dispatches
to the right explainer, runs the Lundberg decomposition (when
requested), writes results back through ``SupabaseWriter``, and updates
``xai_jobs.status`` for the Realtime subscription on the frontend.

Designed to run as a long-lived worker process alongside the vfairness
producer. Configuration via environment:

* ``SUPABASE_URL``                  -- Supabase project URL
* ``SUPABASE_SERVICE_ROLE_KEY``     -- service-role JWT (bypasses RLS)
* ``PGMQ_QUEUE``                    -- queue name (default: ``xai_jobs``)
* ``WORKER_CONCURRENCY``            -- parallel jobs (default: 1)
* ``WORKER_POLL_INTERVAL_S``        -- poll cadence (default: 5)
* ``WORKER_ARTIFACT_BUCKET``        -- Storage bucket (default: ``xai-audit-artifacts``)
"""

from .runner import WorkerConfig, WorkerLoop

__all__ = ["WorkerLoop", "WorkerConfig"]
