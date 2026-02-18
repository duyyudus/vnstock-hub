from __future__ import annotations

import json
import sqlite3
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Any

from tools.cafef_scraper.types import CoverageStatus, DocStatus, DocType, SourceType


class Catalog:
    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        self._ensure_schema()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _ensure_schema(self) -> None:
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS crawl_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    command TEXT NOT NULL,
                    params_json TEXT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    status TEXT NOT NULL,
                    stats_json TEXT,
                    error TEXT
                );

                CREATE TABLE IF NOT EXISTS crawl_state (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    article_id TEXT,
                    url TEXT NOT NULL UNIQUE,
                    published_at TEXT,
                    title TEXT,
                    normalized_title TEXT,
                    source_type TEXT NOT NULL,
                    discovered_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_articles_article_id ON articles(article_id);
                CREATE INDEX IF NOT EXISTS ix_articles_published_at ON articles(published_at);

                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    article_id TEXT NOT NULL,
                    article_url TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    doc_type TEXT NOT NULL,
                    event_date TEXT NOT NULL,
                    pdf_url TEXT NOT NULL,
                    local_path TEXT,
                    sha256 TEXT,
                    size_bytes INTEGER,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    derived_from_published INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(doc_type, event_date, pdf_url)
                );
                CREATE INDEX IF NOT EXISTS ix_documents_event_date_doc_type_status
                    ON documents(event_date, doc_type, status);
                CREATE INDEX IF NOT EXISTS ix_documents_sha256 ON documents(sha256);

                CREATE TABLE IF NOT EXISTS daily_coverage (
                    date TEXT PRIMARY KEY,
                    basket_status TEXT NOT NULL DEFAULT 'MISSING',
                    swap_end_status TEXT NOT NULL DEFAULT 'MISSING',
                    note TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_daily_coverage_date ON daily_coverage(date);

                CREATE TABLE IF NOT EXISTS http_failures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    url TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    status_code INTEGER,
                    error TEXT,
                    attempt_no INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_http_failures_run_id ON http_failures(run_id);

                CREATE TABLE IF NOT EXISTS article_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope_key TEXT NOT NULL,
                    url TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'DISCOVERED',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    discovered_at TEXT NOT NULL,
                    fetched_at TEXT,
                    UNIQUE(scope_key, url)
                );
                CREATE INDEX IF NOT EXISTS ix_article_queue_scope_status
                    ON article_queue(scope_key, status);
                """
            )

    def start_run(self, command: str, params: dict[str, Any]) -> int:
        started_at = _utc_now_iso()
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO crawl_runs(command, params_json, started_at, status)
                VALUES(?, ?, ?, ?)
                """,
                (command, json.dumps(params, ensure_ascii=False), started_at, "RUNNING"),
            )
            return int(cursor.lastrowid)

    def finish_run(
        self,
        run_id: int,
        status: str,
        stats: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        finished_at = _utc_now_iso()
        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE crawl_runs
                SET finished_at = ?, status = ?, stats_json = ?, error = ?
                WHERE id = ?
                """,
                (
                    finished_at,
                    status,
                    json.dumps(stats or {}, ensure_ascii=False),
                    error,
                    run_id,
                ),
            )

    def set_state(self, key: str, value: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO crawl_state(key, value, updated_at)
                VALUES(?, ?, ?)
                ON CONFLICT(key) DO UPDATE
                    SET value = excluded.value,
                        updated_at = excluded.updated_at
                """,
                (key, value, _utc_now_iso()),
            )

    def get_state(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM crawl_state WHERE key = ?",
                (key,),
            ).fetchone()
            return str(row["value"]) if row else None

    def delete_state_prefix(self, prefix: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM crawl_state WHERE key LIKE ?",
                (f"{prefix}%",),
            )

    def enqueue_article_urls(
        self,
        scope_key: str,
        source_type: SourceType,
        urls: list[str],
    ) -> int:
        now = _utc_now_iso()
        inserted = 0
        with self._lock, self._conn:
            for url in urls:
                cursor = self._conn.execute(
                    """
                    INSERT INTO article_queue(scope_key, url, source_type, status, discovered_at)
                    VALUES (?, ?, ?, 'DISCOVERED', ?)
                    ON CONFLICT(scope_key, url) DO NOTHING
                    """,
                    (scope_key, url, source_type.value, now),
                )
                if cursor.rowcount == 1:
                    inserted += 1
        return inserted

    def list_pending_article_queue(self, scope_key: str, include_filtered: bool = False) -> list[sqlite3.Row]:
        statuses = ("DISCOVERED", "FAILED", "FILTERED") if include_filtered else ("DISCOVERED", "FAILED")
        placeholders = ",".join("?" for _ in statuses)
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT *
                FROM article_queue
                WHERE scope_key = ? AND status IN ({placeholders})
                ORDER BY id ASC
                """,
                (scope_key, *statuses),
            ).fetchall()
            return list(rows)

    def mark_article_queue_fetched(self, queue_id: int) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE article_queue
                SET status = 'FETCHED',
                    attempts = attempts + 1,
                    last_error = NULL,
                    fetched_at = ?
                WHERE id = ?
                """,
                (_utc_now_iso(), queue_id),
            )

    def mark_article_queue_failed(self, queue_id: int, error: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE article_queue
                SET status = 'FAILED',
                    attempts = attempts + 1,
                    last_error = ?,
                    fetched_at = NULL
                WHERE id = ?
                """,
                (error[:500], queue_id),
            )

    def mark_article_queue_filtered(self, queue_id: int, reason: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE article_queue
                SET status = 'FILTERED',
                    last_error = ?,
                    fetched_at = NULL
                WHERE id = ?
                """,
                (reason[:500], queue_id),
            )

    def reset_discovery_scope(self, scope_key: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM article_queue WHERE scope_key = ?",
                (scope_key,),
            )
        self.delete_state_prefix(f"discovery:{scope_key}:")

    def record_http_failure(
        self,
        run_id: int | None,
        url: str,
        stage: str,
        status_code: int | None,
        error: str | None,
        attempt_no: int,
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO http_failures(run_id, url, stage, status_code, error, attempt_no, created_at)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, url, stage, status_code, error, attempt_no, _utc_now_iso()),
            )

    def upsert_article(
        self,
        article_id: str | None,
        url: str,
        published_at: datetime | None,
        title: str | None,
        normalized_title: str | None,
        source_type: SourceType,
    ) -> int:
        now = _utc_now_iso()
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO articles(article_id, url, published_at, title, normalized_title, source_type, discovered_at)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE
                    SET article_id = COALESCE(excluded.article_id, articles.article_id),
                        published_at = COALESCE(excluded.published_at, articles.published_at),
                        title = COALESCE(excluded.title, articles.title),
                        normalized_title = COALESCE(excluded.normalized_title, articles.normalized_title),
                        source_type = excluded.source_type
                """,
                (
                    article_id,
                    url,
                    published_at.isoformat() if published_at else None,
                    title,
                    normalized_title,
                    source_type.value,
                    now,
                ),
            )
            row = self._conn.execute("SELECT id FROM articles WHERE url = ?", (url,)).fetchone()
            return int(row["id"])

    def upsert_document(
        self,
        article_id: str,
        article_url: str,
        source_type: SourceType,
        doc_type: DocType,
        event_date: date,
        pdf_url: str,
        derived_from_published: bool,
        status: DocStatus = DocStatus.MISSING,
    ) -> int:
        now = _utc_now_iso()
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO documents(
                    article_id, article_url, source_type, doc_type, event_date, pdf_url,
                    status, derived_from_published, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_type, event_date, pdf_url) DO UPDATE
                    SET article_id = excluded.article_id,
                        article_url = excluded.article_url,
                        source_type = excluded.source_type,
                        derived_from_published = excluded.derived_from_published,
                        updated_at = excluded.updated_at
                """,
                (
                    article_id,
                    article_url,
                    source_type.value,
                    doc_type.value,
                    event_date.isoformat(),
                    pdf_url,
                    status.value,
                    1 if derived_from_published else 0,
                    now,
                    now,
                ),
            )
            row = self._conn.execute(
                """
                SELECT id FROM documents
                WHERE doc_type = ? AND event_date = ? AND pdf_url = ?
                """,
                (doc_type.value, event_date.isoformat(), pdf_url),
            ).fetchone()
            return int(row["id"])

    def get_document_by_id(self, doc_id: int) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()

    def get_document_by_sha256(self, sha256: str) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM documents WHERE sha256 = ? LIMIT 1",
                (sha256,),
            ).fetchone()

    def mark_document_found(self, doc_id: int, local_path: str, sha256: str, size_bytes: int) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE documents
                SET local_path = ?, sha256 = ?, size_bytes = ?, status = ?, last_error = NULL,
                    attempts = attempts + 1, updated_at = ?
                WHERE id = ?
                """,
                (
                    local_path,
                    sha256,
                    size_bytes,
                    DocStatus.FOUND.value,
                    _utc_now_iso(),
                    doc_id,
                ),
            )

    def mark_document_discovered(self, doc_id: int) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE documents
                SET status = ?, attempts = attempts + 1, updated_at = ?
                WHERE id = ?
                """,
                (
                    DocStatus.FOUND.value,
                    _utc_now_iso(),
                    doc_id,
                ),
            )

    def mark_document_duplicate(self, doc_id: int, local_path: str, sha256: str, size_bytes: int) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE documents
                SET local_path = ?, sha256 = ?, size_bytes = ?, status = ?, last_error = NULL,
                    attempts = attempts + 1, updated_at = ?
                WHERE id = ?
                """,
                (
                    local_path,
                    sha256,
                    size_bytes,
                    DocStatus.SKIPPED_DUPLICATE.value,
                    _utc_now_iso(),
                    doc_id,
                ),
            )

    def mark_document_failed(self, doc_id: int, error: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE documents
                SET status = ?, attempts = attempts + 1, last_error = ?, updated_at = ?
                WHERE id = ?
                """,
                (DocStatus.FAILED.value, error[:500], _utc_now_iso(), doc_id),
            )

    def get_existing_found_document(
        self,
        doc_type: DocType,
        event_date: date,
        pdf_url: str,
    ) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(
                """
                SELECT * FROM documents
                WHERE doc_type = ? AND event_date = ? AND pdf_url = ? AND status IN (?, ?)
                LIMIT 1
                """,
                (
                    doc_type.value,
                    event_date.isoformat(),
                    pdf_url,
                    DocStatus.FOUND.value,
                    DocStatus.SKIPPED_DUPLICATE.value,
                ),
            ).fetchone()

    def ensure_coverage_range(self, start_date: date, end_date: date) -> None:
        current = start_date
        with self._lock, self._conn:
            while current <= end_date:
                iso = current.isoformat()
                self._conn.execute(
                    """
                    INSERT INTO daily_coverage(date, basket_status, swap_end_status, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(date) DO NOTHING
                    """,
                    (
                        iso,
                        CoverageStatus.MISSING.value,
                        CoverageStatus.MISSING.value,
                        _utc_now_iso(),
                    ),
                )
                current = date.fromordinal(current.toordinal() + 1)

    def set_coverage_status(
        self,
        target_date: date,
        doc_type: DocType,
        status: CoverageStatus,
        note: str | None = None,
    ) -> None:
        column = "basket_status" if doc_type == DocType.BASKET_NOTICE else "swap_end_status"
        with self._lock, self._conn:
            self._conn.execute(
                f"""
                INSERT INTO daily_coverage(date, basket_status, swap_end_status, note, updated_at)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(date) DO UPDATE
                    SET {column} = excluded.{column},
                        note = COALESCE(excluded.note, daily_coverage.note),
                        updated_at = excluded.updated_at
                """,
                (
                    target_date.isoformat(),
                    status.value if column == "basket_status" else CoverageStatus.MISSING.value,
                    status.value if column == "swap_end_status" else CoverageStatus.MISSING.value,
                    note,
                    _utc_now_iso(),
                ),
            )

    def get_coverage_status(self, target_date: date, doc_type: DocType) -> CoverageStatus | None:
        column = "basket_status" if doc_type == DocType.BASKET_NOTICE else "swap_end_status"
        with self._lock:
            row = self._conn.execute(
                f"SELECT {column} FROM daily_coverage WHERE date = ?",
                (target_date.isoformat(),),
            ).fetchone()
            if not row:
                return None
            return CoverageStatus(str(row[column]))

    def list_coverage_rows(self, start_date: date, end_date: date) -> list[sqlite3.Row]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT *
                FROM daily_coverage
                WHERE date >= ? AND date <= ?
                ORDER BY date ASC
                """,
                (start_date.isoformat(), end_date.isoformat()),
            ).fetchall()
            return list(rows)

    def list_missing_coverage(self, start_date: date, end_date: date, doc_types: set[DocType]) -> list[tuple[date, DocType]]:
        rows = self.list_coverage_rows(start_date, end_date)
        missing: list[tuple[date, DocType]] = []
        for row in rows:
            target = date.fromisoformat(str(row["date"]))
            if DocType.BASKET_NOTICE in doc_types and str(row["basket_status"]) == CoverageStatus.MISSING.value:
                missing.append((target, DocType.BASKET_NOTICE))
            if DocType.SWAP_END in doc_types and str(row["swap_end_status"]) == CoverageStatus.MISSING.value:
                missing.append((target, DocType.SWAP_END))
        return missing

    def list_failed_documents(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int = 200,
    ) -> list[sqlite3.Row]:
        where_parts = ["status = ?"]
        params: list[Any] = [DocStatus.FAILED.value]
        if start_date is not None:
            where_parts.append("event_date >= ?")
            params.append(start_date.isoformat())
        if end_date is not None:
            where_parts.append("event_date <= ?")
            params.append(end_date.isoformat())
        params.append(limit)
        sql = (
            "SELECT * FROM documents WHERE "
            + " AND ".join(where_parts)
            + " ORDER BY updated_at DESC LIMIT ?"
        )
        with self._lock:
            rows = self._conn.execute(sql, tuple(params)).fetchall()
            return list(rows)

    def list_documents_by_status(
        self,
        start_date: date,
        end_date: date,
        statuses: set[DocStatus],
    ) -> list[sqlite3.Row]:
        placeholders = ",".join("?" for _ in statuses)
        params = [start_date.isoformat(), end_date.isoformat(), *[s.value for s in statuses]]
        sql = f"""
            SELECT * FROM documents
            WHERE event_date >= ? AND event_date <= ? AND status IN ({placeholders})
            ORDER BY event_date ASC
        """
        with self._lock:
            rows = self._conn.execute(sql, tuple(params)).fetchall()
            return list(rows)

    def reset_range(self, start_date: date, end_date: date, doc_types: set[DocType]) -> None:
        placeholders = ",".join("?" for _ in doc_types)
        doc_type_values = [d.value for d in doc_types]
        with self._lock, self._conn:
            self._conn.execute(
                f"""
                DELETE FROM documents
                WHERE event_date >= ? AND event_date <= ? AND doc_type IN ({placeholders})
                """,
                (start_date.isoformat(), end_date.isoformat(), *doc_type_values),
            )
            if DocType.BASKET_NOTICE in doc_types:
                self._conn.execute(
                    """
                    UPDATE daily_coverage
                    SET basket_status = ?, updated_at = ?, note = NULL
                    WHERE date >= ? AND date <= ?
                    """,
                    (
                        CoverageStatus.MISSING.value,
                        _utc_now_iso(),
                        start_date.isoformat(),
                        end_date.isoformat(),
                    ),
                )
            if DocType.SWAP_END in doc_types:
                self._conn.execute(
                    """
                    UPDATE daily_coverage
                    SET swap_end_status = ?, updated_at = ?, note = NULL
                    WHERE date >= ? AND date <= ?
                    """,
                    (
                        CoverageStatus.MISSING.value,
                        _utc_now_iso(),
                        start_date.isoformat(),
                        end_date.isoformat(),
                    ),
                )


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat()
