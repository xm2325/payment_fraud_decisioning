from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from fraud_decisioning.access_api import get_store, router
from fraud_decisioning.access_store import AccessStore


def _client() -> tuple[TestClient, AccessStore]:
    app = FastAPI()
    app.include_router(router)
    store = AccessStore(":memory:")
    app.dependency_overrides[get_store] = lambda: store
    return TestClient(app), store


def _auth(subject: str, roles: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {subject}|{roles}"}


def test_missing_token_is_401():
    client, _ = _client()
    response = client.post("/research-access/requests", json={"dataset": "NGRL-demo", "purpose": "approved research"})
    assert response.status_code == 401


def test_wrong_role_is_403():
    client, _ = _client()
    response = client.post(
        "/research-access/requests",
        headers=_auth("rev-1", "reviewer"),
        json={"dataset": "NGRL-demo", "purpose": "approved research"},
    )
    assert response.status_code == 403


def test_create_duplicate_review_and_audit_flow():
    client, store = _client()
    payload = {"dataset": "NGRL-demo", "purpose": "rare disease method development"}

    created = client.post("/research-access/requests", headers=_auth("researcher-1", "researcher"), json=payload)
    assert created.status_code == 201
    request_id = created.json()["request_id"]
    assert created.json()["status"] == "pending"

    duplicate = client.post("/research-access/requests", headers=_auth("researcher-1", "researcher"), json=payload)
    assert duplicate.status_code == 409

    other_researcher = client.get(f"/research-access/requests/{request_id}", headers=_auth("researcher-2", "researcher"))
    assert other_researcher.status_code == 403

    reviewed = client.post(
        f"/research-access/requests/{request_id}/review",
        headers=_auth("reviewer-1", "reviewer"),
        json={"decision": "approved"},
    )
    assert reviewed.status_code == 200
    assert reviewed.json()["status"] == "approved"
    assert reviewed.json()["reviewed_by"] == "reviewer-1"

    repeated_review = client.post(
        f"/research-access/requests/{request_id}/review",
        headers=_auth("reviewer-1", "reviewer"),
        json={"decision": "rejected"},
    )
    assert repeated_review.status_code == 409

    audit_denied = client.get("/research-access/audit-events", headers=_auth("reviewer-1", "reviewer"))
    assert audit_denied.status_code == 403

    audit = client.get("/research-access/audit-events", headers=_auth("admin-1", "admin"))
    assert audit.status_code == 200
    actions = [event["action"] for event in audit.json()["events"]]
    assert actions == ["access_request.created", "access_request.approved"]
    assert len(store.list_audit()) == 2


def test_not_found_and_invalid_decision():
    client, _ = _client()
    missing = client.get("/research-access/requests/999", headers=_auth("reviewer-1", "reviewer"))
    assert missing.status_code == 404

    created = client.post(
        "/research-access/requests",
        headers=_auth("researcher-1", "researcher"),
        json={"dataset": "NGRL-demo", "purpose": "cancer research"},
    )
    request_id = created.json()["request_id"]

    invalid = client.post(
        f"/research-access/requests/{request_id}/review",
        headers=_auth("reviewer-1", "reviewer"),
        json={"decision": "maybe"},
    )
    assert invalid.status_code == 422
