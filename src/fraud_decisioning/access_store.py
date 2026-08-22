from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class AccessRequestRecord:
    request_id: int
    requester: str
    dataset: str
    purpose: str
    status: str
    reviewed_by: str | None


class AccessStore:
    """Small transactional store used by the API demo and tests.

    SQLite is the local reference implementation. SQL is deliberately kept
    portable enough for the same repository interface to be backed by
    PostgreSQL/RDS in deployment.
    """

    def __init__(self, database_path: str | Path = ":memory:") -> None:
        self.database_path = str(database_path)
        self._keeper: sqlite3.Connection | None = None
        if self.database_path == ":memory:":
            self._keeper = sqlite3.connect(":memory:", check_same_thread=False)
            self._keeper.row_factory = sqlite3.Row
            self._initialise(self._keeper)
        else:
            with self.connection() as conn:
                self._initialise(conn)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        if self._keeper is not None:
            conn = self._keeper
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            return

        conn = sqlite3.connect(self.database_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _initialise(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS access_requests (
                request_id INTEGER PRIMARY KEY AUTOINCREMENT,
                requester TEXT NOT NULL,
                dataset TEXT NOT NULL,
                purpose TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('pending','approved','rejected')),
                reviewed_by TEXT,
                UNIQUE(requester, dataset, purpose)
            );
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                actor TEXT NOT NULL,
                action TEXT NOT NULL,
                resource_type TEXT NOT NULL,
                resource_id TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            """
        )
        conn.commit()

    def create_request(self, requester: str, dataset: str, purpose: str) -> AccessRequestRecord:
        with self.connection() as conn:
            try:
                cursor = conn.execute(
                    "INSERT INTO access_requests(requester, dataset, purpose, status) VALUES (?, ?, ?, 'pending')",
                    (requester, dataset, purpose),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("duplicate access request") from exc
            request_id = int(cursor.lastrowid)
            self._append_audit(conn, requester, "access_request.created", "access_request", str(request_id), {"dataset": dataset})
            row = conn.execute("SELECT * FROM access_requests WHERE request_id = ?", (request_id,)).fetchone()
            assert row is not None
            return self._to_record(row)

    def get_request(self, request_id: int) -> AccessRequestRecord | None:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM access_requests WHERE request_id = ?", (request_id,)).fetchone()
            return self._to_record(row) if row is not None else None

    def review_request(self, request_id: int, reviewer: str, decision: str) -> AccessRequestRecord | None:
        if decision not in {"approved", "rejected"}:
            raise ValueError("invalid decision")
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM access_requests WHERE request_id = ?", (request_id,)).fetchone()
            if row is None:
                return None
            if row["status"] != "pending":
                raise ValueError("request already reviewed")
            conn.execute(
                "UPDATE access_requests SET status = ?, reviewed_by = ? WHERE request_id = ?",
                (decision, reviewer, request_id),
            )
            self._append_audit(conn, reviewer, f"access_request.{decision}", "access_request", str(request_id), {})
            updated = conn.execute("SELECT * FROM access_requests WHERE request_id = ?", (request_id,)).fetchone()
            assert updated is not None
            return self._to_record(updated)

    def list_audit(self) -> list[dict[str, object]]:
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM audit_events ORDER BY event_id").fetchall()
            return [
                {
                    "event_id": int(row["event_id"]),
                    "actor": row["actor"],
                    "action": row["action"],
                    "resource_type": row["resource_type"],
                    "resource_id": row["resource_id"],
                    "payload": json.loads(row["payload_json"]),
                }
                for row in rows
            ]

    @staticmethod
    def _append_audit(
        conn: sqlite3.Connection,
        actor: str,
        action: str,
        resource_type: str,
        resource_id: str,
        payload: dict[str, object],
    ) -> None:
        conn.execute(
            "INSERT INTO audit_events(actor, action, resource_type, resource_id, payload_json) VALUES (?, ?, ?, ?, ?)",
            (actor, action, resource_type, resource_id, json.dumps(payload, sort_keys=True)),
        )

    @staticmethod
    def _to_record(row: sqlite3.Row) -> AccessRequestRecord:
        return AccessRequestRecord(
            request_id=int(row["request_id"]),
            requester=str(row["requester"]),
            dataset=str(row["dataset"]),
            purpose=str(row["purpose"]),
            status=str(row["status"]),
            reviewed_by=str(row["reviewed_by"]) if row["reviewed_by"] is not None else None,
        )
