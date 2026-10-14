"""SQLite cache, budget reservation and usage ledger."""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator


class BudgetExceeded(RuntimeError):
    def __init__(self, daily_limit: float, already_committed: float, requested_estimate: float):
        self.daily_limit = daily_limit
        self.already_committed = already_committed
        self.requested_estimate = requested_estimate
        super().__init__(
            f"Daily LLM budget exceeded: limit=${daily_limit:.6f}, "
            f"committed=${already_committed:.6f}, request_estimate=${requested_estimate:.6f}."
        )


class SQLiteStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        if db_path != ":memory:":
            Path(db_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=15, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 15000")
        return connection

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS response_cache (
                    cache_key TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    answer TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS usage_ledger (
                    request_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    day_utc TEXT NOT NULL,
                    model TEXT NOT NULL,
                    route_reason TEXT NOT NULL DEFAULT '',
                    estimated_cost_usd REAL NOT NULL,
                    actual_cost_usd REAL,
                    status TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    cached_input_tokens INTEGER NOT NULL DEFAULT 0,
                    cache_write_input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    latency_ms REAL NOT NULL DEFAULT 0,
                    cache_hit INTEGER NOT NULL DEFAULT 0,
                    usage_estimated INTEGER NOT NULL DEFAULT 0,
                    error_type TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_usage_day_status ON usage_ledger(day_utc, status);
                """
            )
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(usage_ledger)")}
            if "retry_count" not in columns:
                conn.execute("ALTER TABLE usage_ledger ADD COLUMN retry_count INTEGER NOT NULL DEFAULT 0")
            if "usage_estimated" not in columns:
                conn.execute("ALTER TABLE usage_ledger ADD COLUMN usage_estimated INTEGER NOT NULL DEFAULT 0")

    def get_cached(self, cache_key: str, now: float | None = None) -> tuple[str, dict[str, Any]] | None:
        moment = now if now is not None else time.time()
        with self._connection() as conn:
            row = conn.execute(
                "SELECT answer, metadata_json, expires_at FROM response_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if row is None:
                return None
            if row["expires_at"] <= moment:
                conn.execute("DELETE FROM response_cache WHERE cache_key = ?", (cache_key,))
                return None
            return row["answer"], json.loads(row["metadata_json"])

    def put_cached(
        self,
        cache_key: str,
        answer: str,
        metadata: dict[str, Any],
        ttl_seconds: int,
        now: float | None = None,
    ) -> None:
        moment = now if now is not None else time.time()
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO response_cache(cache_key, created_at, expires_at, answer, metadata_json)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(cache_key) DO UPDATE SET created_at=excluded.created_at,
                     expires_at=excluded.expires_at, answer=excluded.answer, metadata_json=excluded.metadata_json""",
                (cache_key, moment, moment + ttl_seconds, answer, json.dumps(metadata, sort_keys=True)),
            )

    @staticmethod
    def _utc_day(now: float | None = None) -> str:
        moment = datetime.fromtimestamp(now if now is not None else time.time(), timezone.utc)
        return moment.date().isoformat()

    def reserve_budget(
        self,
        *,
        request_id: str,
        model: str,
        estimated_cost_usd: float,
        daily_limit_usd: float | None,
        route_reason: str = '',
        now: float | None = None,
    ) -> None:
        if estimated_cost_usd < 0:
            raise ValueError("Estimated cost cannot be negative")
        moment = now if now is not None else time.time()
        day_utc = self._utc_day(moment)
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """SELECT COALESCE(SUM(CASE
                       WHEN status = 'RESERVED' THEN estimated_cost_usd
                       WHEN status = 'SETTLED' THEN COALESCE(actual_cost_usd, 0)
                       WHEN status = 'FAILED' THEN estimated_cost_usd
                       ELSE 0 END), 0) AS committed
                   FROM usage_ledger WHERE day_utc = ?""",
                (day_utc,),
            ).fetchone()
            committed = float(row["committed"] or 0.0)
            if daily_limit_usd is not None and committed + estimated_cost_usd > daily_limit_usd:
                conn.rollback()
                raise BudgetExceeded(daily_limit_usd, committed, estimated_cost_usd)
            conn.execute(
                """INSERT INTO usage_ledger(request_id, created_at, day_utc, model, route_reason,
                   estimated_cost_usd, actual_cost_usd, status)
                   VALUES (?, ?, ?, ?, ?, ?, NULL, 'RESERVED')""",
                (request_id, moment, day_utc, model, route_reason[:240], estimated_cost_usd),
            )
            conn.commit()

    def settle(
        self,
        *,
        request_id: str,
        actual_cost_usd: float,
        input_tokens: int,
        cached_input_tokens: int,
        cache_write_input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        retry_count: int = 0,
        usage_estimated: bool = False,
    ) -> None:
        with self._connection() as conn:
            cursor = conn.execute(
                """UPDATE usage_ledger SET actual_cost_usd=?, status='SETTLED', input_tokens=?,
                   cached_input_tokens=?, cache_write_input_tokens=?, output_tokens=?, latency_ms=?,
                   retry_count=?, usage_estimated=?
                   WHERE request_id=? AND status='RESERVED'""",
                (actual_cost_usd, input_tokens, cached_input_tokens, cache_write_input_tokens,
                 output_tokens, latency_ms, retry_count, int(usage_estimated), request_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"Could not settle ledger reservation {request_id}")

    def mark_failed(self, request_id: str, error_type: str, *, retry_count: int = 0) -> None:
        # Keep estimated_cost_usd as a conservative budget reservation. A timeout
        # can happen after the provider has already processed and billed the request.
        # The daily guard counts FAILED reservations; usage_summary counts only settled cost.
        with self._connection() as conn:
            conn.execute(
                "UPDATE usage_ledger SET status='FAILED', error_type=?, retry_count=? "
                "WHERE request_id=? AND status='RESERVED'",
                (error_type[:120], retry_count, request_id),
            )

    def record_cache_hit(
        self,
        *,
        request_id: str,
        model: str,
        route_reason: str = "Exact response cache hit",
        now: float | None = None,
    ) -> None:
        moment = now if now is not None else time.time()
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO usage_ledger(request_id, created_at, day_utc, model, route_reason,
                   estimated_cost_usd, actual_cost_usd, status, cache_hit)
                   VALUES (?, ?, ?, ?, ?, 0, 0, 'CACHED', 1)""",
                (request_id, moment, self._utc_day(moment), model, route_reason[:240]),
            )

    def usage_summary(self, now: float | None = None) -> dict[str, float | int]:
        day = self._utc_day(now)
        with self._connection() as conn:
            row = conn.execute(
                """SELECT COUNT(*) AS request_count,
                   SUM(CASE WHEN status='SETTLED' THEN 1 ELSE 0 END) AS api_calls,
                   SUM(CASE WHEN cache_hit=1 THEN 1 ELSE 0 END) AS cache_hits,
                   SUM(CASE WHEN status='SETTLED' AND usage_estimated=1 THEN 1 ELSE 0 END) AS estimated_usage_calls,
                   SUM(retry_count) AS retry_count,
                   SUM(CASE WHEN status='SETTLED' THEN COALESCE(actual_cost_usd, 0) ELSE 0 END) AS spend_usd,
                   AVG(CASE WHEN status='SETTLED' THEN latency_ms END) AS avg_latency_ms
                   FROM usage_ledger WHERE day_utc=?""",
                (day,),
            ).fetchone()
        return {
            "utc_day": day,
            "request_count": int(row["request_count"] or 0),
            "api_calls": int(row["api_calls"] or 0),
            "cache_hits": int(row["cache_hits"] or 0),
            "estimated_usage_calls": int(row["estimated_usage_calls"] or 0),
            "retry_count": int(row["retry_count"] or 0),
            "spend_usd": float(row["spend_usd"] or 0.0),
            "avg_latency_ms": float(row["avg_latency_ms"] or 0.0),
        }
