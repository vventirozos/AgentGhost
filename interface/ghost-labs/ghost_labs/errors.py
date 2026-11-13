"""Error shape (contract v1 §9): {"error","message","hint",...extra}."""
from __future__ import annotations

from typing import Any, Dict

STATUS = {
    "unauthorized": 401, "bad_request": 400, "unknown_template": 404, "unknown_lab": 404,
    "unknown_member": 404, "unknown_backup": 404, "budget_exceeded": 409, "lab_limit": 409,
    "not_ready": 409, "busy": 409, "owner_active": 409, "data_exists": 409, "not_kept": 409,
    "no_rollback": 409, "ports_exhausted": 409, "already_kept": 409, "owner_only": 403,
    "not_supported": 501, "internal": 500, "tor_unavailable": 503, "firewall_missing": 503,
    "disk_low": 507, "unknown_job": 404, "build_failed": 422,
}


class LabError(Exception):
    def __init__(self, code: str, message: str, hint: str = "", **extra: Any):
        super().__init__(message)
        if code not in STATUS:
            raise ValueError(f"unknown error code {code}")
        self.code, self.message, self.hint, self.extra = code, message, hint, extra

    @property
    def status(self) -> int:
        return STATUS[self.code]

    def body(self) -> Dict[str, Any]:
        return {"error": self.code, "message": self.message, "hint": self.hint, **self.extra}
