"""Poll a backend job while showing progress."""

from __future__ import annotations

import time

import streamlit as st

from frontend.api_client import ApiClient, ApiError


def wait_for_job(client: ApiClient, job_id: str, label: str, poll_seconds: float = 0.8, timeout: float = 3600) -> dict:
    """Blocks (inside an st.status box) until the job finishes. Returns the final job record."""
    with st.status(label, expanded=True) as status:
        bar = st.progress(0.0, text="Starting")
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                job = client.get_job(job_id)
            except ApiError as exc:
                status.update(label=f"{label}: lost contact with the API", state="error")
                return {"status": "failed", "error": str(exc)}
            bar.progress(min(max(float(job.get("progress", 0.0)), 0.0), 1.0), text=str(job.get("stage", "")))
            if job["status"] == "succeeded":
                status.update(label=f"{label}: done", state="complete", expanded=False)
                return job
            if job["status"] == "failed":
                status.update(label=f"{label}: failed", state="error", expanded=True)
                return job
            time.sleep(poll_seconds)
        status.update(label=f"{label}: timed out", state="error")
        return {"status": "failed", "error": "Timed out waiting for the job to finish."}
