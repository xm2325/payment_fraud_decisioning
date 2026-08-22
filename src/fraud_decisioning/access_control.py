from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


class Role(str, Enum):
    RESEARCHER = "researcher"
    REVIEWER = "reviewer"
    ADMIN = "admin"


@dataclass(frozen=True)
class Principal:
    subject: str
    roles: frozenset[Role]


_bearer = HTTPBearer(auto_error=False)


# This local validator is intentionally simple: it provides a deterministic
# authentication boundary for tests and local demos. Production deployment
# should replace it with OIDC/JWKS validation against the configured IdP.
def authenticate(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> Principal:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials.strip()
    # Local token contract: subject|role[,role...]
    try:
        subject, role_text = token.split("|", 1)
        roles = frozenset(Role(item.strip()) for item in role_text.split(",") if item.strip())
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None

    if not subject or not roles:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return Principal(subject=subject, roles=roles)


def require_roles(*allowed: Role):
    allowed_set = frozenset(allowed)

    def dependency(principal: Principal = Depends(authenticate)) -> Principal:
        if principal.roles.isdisjoint(allowed_set):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient role",
            )
        return principal

    return dependency


def has_any_role(principal: Principal, roles: Iterable[Role]) -> bool:
    return not principal.roles.isdisjoint(frozenset(roles))
