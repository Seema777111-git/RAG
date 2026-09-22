"""Thin HTTP client for the FastAPI backend. Returns plain dicts; raises `ApiError` with a readable message."""

from __future__ import annotations

from typing import Any

import httpx

API_PREFIX = "/api/v1"


class ApiError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class ApiClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------ plumbing
    def _request(self, method: str, path: str, *, timeout: float = 30.0, **kwargs: Any) -> Any:
        url = f"{self.base_url}{API_PREFIX}{path}"
        try:
            response = httpx.request(method, url, timeout=timeout, **kwargs)
        except httpx.ConnectError as exc:
            raise ApiError(f"Cannot reach the API at {self.base_url}. Is the backend running?") from exc
        except httpx.TimeoutException as exc:
            raise ApiError("The API took too long to respond.") from exc
        if response.is_error:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            if isinstance(detail, list):  # FastAPI validation errors
                detail = "; ".join(str(d.get("msg", d)) for d in detail)
            raise ApiError(str(detail), response.status_code)
        return response.json()

    # ------------------------------------------------------------------ system
    def stats(self) -> dict:
        return self._request("GET", "/stats", timeout=10)

    def check_llm(self) -> dict:
        return self._request("POST", "/llm/check", timeout=60)

    # ------------------------------------------------------------------ documents
    def list_documents(self) -> list[dict]:
        return self._request("GET", "/documents")

    def upload_document(self, filename: str, data: bytes) -> dict:
        return self._request("POST", "/documents", files={"file": (filename, data, "application/pdf")}, timeout=120)

    def delete_document(self, doc_id: str) -> dict:
        return self._request("DELETE", f"/documents/{doc_id}")

    def get_job(self, job_id: str) -> dict:
        return self._request("GET", f"/jobs/{job_id}")

    # ------------------------------------------------------------------ query
    def query(self, payload: dict) -> dict:
        return self._request("POST", "/query", json=payload, timeout=180)

    # ------------------------------------------------------------------ evaluation
    def evaluation_metrics(self) -> list[str]:
        return self._request("GET", "/evaluate/metrics")["metrics"]

    def generate_cases(self, num_questions: int) -> list[dict]:
        return self._request("POST", "/evaluate/generate", json={"num_questions": num_questions}, timeout=180)

    def start_evaluation(self, payload: dict) -> dict:
        return self._request("POST", "/evaluate", json=payload)

    def list_reports(self) -> list[dict]:
        return self._request("GET", "/evaluate/reports")

    def get_report(self, name: str) -> dict:
        return self._request("GET", f"/evaluate/reports/{name}")
