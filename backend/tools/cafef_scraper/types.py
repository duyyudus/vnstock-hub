from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path


class DocType(StrEnum):
    BASKET_NOTICE = "BASKET_NOTICE"
    SWAP_END = "SWAP_END"


class DocStatus(StrEnum):
    FOUND = "FOUND"
    MISSING = "MISSING"
    FAILED = "FAILED"
    SKIPPED_DUPLICATE = "SKIPPED_DUPLICATE"


class SourceType(StrEnum):
    SITEMAP = "SITEMAP"
    EVENT_FEED = "EVENT_FEED"
    ID_SCAN = "ID_SCAN"
    URL_GUESS = "URL_GUESS"


class CoverageStatus(StrEnum):
    FOUND = "FOUND"
    MISSING = "MISSING"
    FAILED = "FAILED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(slots=True)
class ArticleCandidate:
    url: str
    source_type: SourceType


@dataclass(slots=True)
class ArticleDetail:
    url: str
    source_type: SourceType
    article_id: str | None
    title: str | None
    normalized_title: str | None
    published_at: datetime | None
    pdf_urls: list[str]


@dataclass(slots=True)
class DocumentCandidate:
    article_id: str
    article_url: str
    source_type: SourceType
    doc_type: DocType
    event_date: date
    pdf_url: str
    slug: str
    derived_from_published: bool


@dataclass(slots=True)
class DownloadResult:
    success: bool
    status_code: int | None
    error: str | None
    local_path: Path | None = None
    sha256: str | None = None
    size_bytes: int | None = None
