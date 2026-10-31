"""API-key authentication with two roles, keys from the environment (.env):

  API_KEYS_ADMIN   comma-separated keys with the admin role (everything, incl. /metrics and /admin/*)
  API_KEYS_VIEWER  comma-separated keys with the viewer role (POST /ask and read-only GET endpoints)

The key travels in the X-API-Key header. Keys are compared in constant time and never logged.
"""

import hmac
import os

from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader

HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
ROLES = ("viewer", "admin")


def keys(role: str) -> list[str]:
    var = {"admin": "API_KEYS_ADMIN", "viewer": "API_KEYS_VIEWER"}[role]
    return [k.strip() for k in os.environ.get(var, "").split(",") if len(k.strip()) >= 16]


def role_of(key: str | None) -> str | None:
    if not key:
        return None
    for role in ("admin", "viewer"):
        if any(hmac.compare_digest(key, k) for k in keys(role)):
            return role
    return None


def require(minimum: str):
    """Dependency: 401 without a valid key, 403 when the key's role is below `minimum`."""
    def dep(request: Request, key: str | None = Depends(HEADER)) -> str:
        role = role_of(key)
        if role is None:
            raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")
        if ROLES.index(role) < ROLES.index(minimum):
            raise HTTPException(status_code=403, detail=f"the {role} role cannot access this endpoint")
        request.state.role = role
        return role
    return dep
