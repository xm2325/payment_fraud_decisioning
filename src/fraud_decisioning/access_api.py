from __future__ import annotations

import os
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from fraud_decisioning.access_control import Principal, Role, require_roles
from fraud_decisioning.access_store import AccessStore

router = APIRouter(prefix="/research-access", tags=["research-access"])

_store = AccessStore(os.getenv("FRAUD_ACCESS_DB", ":memory:"))


class AccessRequestCreate(BaseModel):
    dataset: str = Field(min_length=1, max_length=200)
    purpose: str = Field(min_length=3, max_length=1000)


class AccessReview(BaseModel):
    decision: str


def get_store() -> AccessStore:
    return _store


@router.post("/requests", status_code=status.HTTP_201_CREATED)
def create_access_request(
    body: AccessRequestCreate,
    principal: Principal = Depends(require_roles(Role.RESEARCHER, Role.ADMIN)),
    store: AccessStore = Depends(get_store),
):
    try:
        record = store.create_request(principal.subject, body.dataset, body.purpose)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return asdict(record)


@router.get("/requests/{request_id}")
def get_access_request(
    request_id: int,
    principal: Principal = Depends(require_roles(Role.RESEARCHER, Role.REVIEWER, Role.ADMIN)),
    store: AccessStore = Depends(get_store),
):
    record = store.get_request(request_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Access request not found")
    if Role.ADMIN not in principal.roles and Role.REVIEWER not in principal.roles and record.requester != principal.subject:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot view another researcher's request")
    return asdict(record)


@router.post("/requests/{request_id}/review")
def review_access_request(
    request_id: int,
    body: AccessReview,
    principal: Principal = Depends(require_roles(Role.REVIEWER, Role.ADMIN)),
    store: AccessStore = Depends(get_store),
):
    if body.decision not in {"approved", "rejected"}:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="decision must be approved or rejected")
    try:
        record = store.review_request(request_id, principal.subject, body.decision)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Access request not found")
    return asdict(record)


@router.get("/audit-events")
def list_audit_events(
    _principal: Principal = Depends(require_roles(Role.ADMIN)),
    store: AccessStore = Depends(get_store),
):
    return {"events": store.list_audit()}
