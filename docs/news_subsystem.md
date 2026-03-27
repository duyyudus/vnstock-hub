# News Subsystem V1

## Purpose

The News subsystem provides a first-class news pipeline for VNStock Hub. In v1 it supports:

- a public default feed available without login
- private user-managed sources for signed-in users
- RSS/Atom discovery and manual feed onboarding
- deterministic non-RSS crawl setup
- article deduplication and ingest-time semantic classification
- per-user blocked-topic filtering
- admin monitoring and repair controls

This document describes the subsystem as implemented in v1.

## Scope

V1 is intentionally not a general web intelligence platform. It is a deterministic ingestion system with LLM-assisted content understanding.

Included in v1:

- background polling of RSS feeds and crawl sources
- public source seeding from YAML
- homepage RSS/Atom discovery
- sitemap-assisted crawl suggestion when RSS is unavailable
- on-demand article summary generation
- admin monitoring and maintenance actions

Explicitly out of scope in v1:

- vector search or embeddings
- semantic reranking or recommendation boosting
- browser-rendered crawling
- LLM-generated crawl rules
- treating sitemap files as permanent feed sources

## Runtime Placement

The subsystem lives under `backend/app/services/news/` and is exposed through `/api/v1/news`.

Key integration points:

- App startup/shutdown:
  - `backend/app/main.py`
  - `news_service.start_background_tasks()`
  - `news_service.stop_background_tasks()`
- API router:
  - `backend/app/api/v1/news.py`
- Service entrypoint:
  - `backend/app/services/news/service.py`
- Discovery and extraction helpers:
  - `backend/app/services/news/discovery.py`
- Semantic classification and summaries:
  - `backend/app/services/news/semantics.py`

## Data Model

The subsystem uses dedicated relational tables instead of the standalone scraper catalog.

Core tables in `backend/app/db/models.py`:

- `news_sites`
  - normalized site/domain metadata
  - public/default vs private ownership marker
- `news_site_feeds`
  - RSS/Atom feeds attached to a site
  - poll interval, validation status, success/failure timestamps
- `news_crawl_sources`
  - deterministic crawl definitions
  - listing URL plus selectors for article links and content
- `news_source_subscriptions`
  - maps a user to private RSS/crawl sources
- `news_articles`
  - canonical deduped article record
  - stores original excerpt, optional `llm_summary`, normalized body text, language, content hash
- `news_article_sources`
  - maps one canonical article to one or more origin sources
- `news_article_semantics`
  - topics, tickers, sectors, importance, sentiment, raw LLM payload
- `news_user_preferences`
  - per-user blocked-topic text and compiled blocked labels
- `news_ingestion_runs`
  - execution history for source polling

Important model behavior:

- canonical URL is unique
- content hash is indexed for secondary dedupe
- one article can originate from multiple feeds/sources
- feed and crawl source polling schedules are source-specific

## Source Types

### Public Default Sources

Public sources are seeded from:

- `backend/news_sources.yaml`

These are loaded at startup by `ensure_public_sources()`.

V1 defaults to validated RSS feeds for the public pack.

### Private RSS Sources

Users can add RSS/Atom feeds through:

- homepage discovery
- manual feed URL entry

Each feed is stored separately even when multiple feeds belong to the same site.

### Private Crawl Sources

Users can add crawl sources for non-RSS sites by providing:

- `listing_url`
- `article_link_selector`
- `content_selector`
- optional `excerpt_selector`
- optional pagination selector config

Validation is deterministic and uses static HTML only.

## Discovery Strategy

### RSS Discovery

RSS discovery is handled by `discover_rss_feeds()` and is deterministic.

It currently tries:

1. `<link rel="alternate" ...>` feed tags on the homepage
2. homepage anchors with feed-like URLs or labels
3. common feed paths such as `/rss`, `/rss.html`, `/feed`, `/atom.xml`
4. feed hub pages that link to real category feeds

Supported result kinds for RSS discovery:

- `rss`
- `atom`

### Sitemap-Assisted Crawl Discovery

If RSS/Atom discovery returns nothing, the subsystem may use sitemap metadata as a discovery aid only.

Important v1 rule:

- sitemap is not treated as a permanent feed source anymore

Instead, `discover_crawl_listings()` uses sitemap files to suggest stable section/listing pages for crawl setup, for example:

- `/kinh-te/`
- `/chung-khoan/`

This is meant to help onboarding for sites that have no real RSS feed but do publish structured category URLs.

### Manual Fallback

Discovery is best-effort. Users can always fall back to:

- manual RSS entry
- manual crawl-source entry

## Validation

### RSS Validation

`validate_feed()`:

- fetches the feed URL
- parses RSS or Atom entries
- returns sample titles and entry counts

Validation succeeds only if the feed contains parseable entries.

### Crawl Validation

`validate_crawl_source()`:

- fetches the listing page
- extracts article links using the provided selector
- samples a small number of article pages
- verifies article body extraction works
- returns heuristic selector suggestions

V1 does not use a headless browser here.

## Ingestion Pipeline

The background worker is implemented by `NewsIngestionService`.

Main loop responsibilities:

1. reconcile stale `running` ingestion rows
2. ensure public sources are present
3. poll due RSS feeds and crawl sources
4. fetch candidates
5. fetch article pages when needed
6. canonicalize URLs
7. dedupe by canonical URL, then by content hash
8. persist article/source mappings
9. classify semantics for new or materially changed content
10. record ingestion runs

Polling uses source-level intervals and `next_poll_at`.

When poll intervals change for a source, the service resets `next_poll_at` so the updated schedule takes effect quickly.

## Content Extraction

Extraction is deterministic and rule-based.

Current behavior includes:

- article body extraction using prioritized selectors
- noise stripping for related content, banners, tags, share blocks, ads, and other non-body sections
- site-agnostic cleanup rules with targeted resilience for known layouts such as VnEconomy

There is also article repair logic:

- opening detail can trigger content repair for suspicious stored bodies
- explicit admin/user-triggered refresh can re-fetch and re-extract one article

## Deduplication

Deduplication happens in two stages:

1. canonical URL match
2. content hash match

This allows:

- one article discovered through multiple feeds to remain a single canonical row
- attribution to all originating sources through `news_article_sources`

## Semantic Layer

LLM usage in v1 is limited to content understanding, not crawling.

### Ingest-Time Classification

For new or changed articles, the subsystem stores:

- `topics`
- `tickers`
- `sectors`
- `importance`
- `sentiment`
- raw payload for debugging/inspection

Classification runs asynchronously during ingestion and is cached in DB.

### Provider Fallback

The subsystem supports multiple OpenAI-compatible providers from `LLM_PROVIDERS`.

The implementation:

- skips obviously unusable providers
- cools down failing providers
- treats empty or malformed “success” responses as failures
- falls back across configured providers

### Blocked Topics

User preference flow:

1. user saves freeform “topics I don’t care about”
2. backend compiles that text into normalized blocked labels
3. personalized feed excludes matching articles

This is exclusion-only in v1.

## Summaries

LLM summaries are on-demand only.

Important v1 behavior:

- `llm_summary` is not generated by default at ingest time
- feed cards show original excerpt unless a summary has been generated
- `Summary` generates and stores the first summary
- `Regenerate` overwrites the stored summary
- summaries are requested in the article’s original language

If article content is refreshed and materially changes, stale `llm_summary` is cleared.

## Feed Behavior

### Anonymous Feed

Anonymous users see:

- public default sources only

### Personalized Feed

Signed-in users see:

- public default sources
- plus their own private subscriptions

Feed-level filtering supports:

- source domain
- topic
- ticker
- date range
- cursor pagination

Current frontend default page size is 20 items per request.

## API Surface

Main public endpoints:

- `GET /api/v1/news/feed`
- `GET /api/v1/news/articles/{article_id}`
- `POST /api/v1/news/articles/{article_id}/summary`
- `POST /api/v1/news/articles/{article_id}/refresh-content`

Authenticated source-management endpoints:

- `POST /api/v1/news/rss/discover`
- `POST /api/v1/news/rss/validate`
- `POST /api/v1/news/crawl/validate`
- `GET /api/v1/news/sources`
- `POST /api/v1/news/sources/rss`
- `POST /api/v1/news/sources/crawl`
- `PATCH /api/v1/news/sources/{source_type}/{source_id}`
- `DELETE /api/v1/news/sources/{source_type}/{source_id}`
- `GET /api/v1/news/preferences`
- `PATCH /api/v1/news/preferences`

Admin/monitoring endpoints:

- `GET /api/v1/news/admin/status`
- `GET /api/v1/news/admin/overview`
- `GET /api/v1/news/admin/sources`
- `GET /api/v1/news/admin/runs`
- `GET /api/v1/news/admin/config`
- `PATCH /api/v1/news/admin/config`
- `POST /api/v1/news/admin/run`
- `POST /api/v1/news/admin/ingest`
- `POST /api/v1/news/admin/refresh`
- `POST /api/v1/news/admin/repair-rss-titles`
- `POST /api/v1/news/admin/apply-default-poll-interval`

## Admin Controls

The admin News monitoring tab currently provides:

- source coverage and ingestion activity
- source validation status and intervals
- recent ingestion runs
- source cache refresh
- manual ingestion trigger
- legacy RSS title repair
- default poll interval setting
- explicit bulk application of the current default poll interval to existing sources

Important behavior:

- saving the default poll interval affects future source creation
- `Apply to existing sources` is a separate explicit action

## Configuration

Environment-backed settings in `backend/app/core/config.py`:

- `NEWS_INGESTION_ENABLED`
- `NEWS_POLL_INTERVAL_SECONDS`
- `NEWS_DEFAULT_POLL_INTERVAL_MINUTES`
- `NEWS_INGESTION_BATCH_SIZE`
- `NEWS_SOURCES_YAML_PATH`
- shared `LLM_PROVIDERS`
- shared `LLM_REQUEST_TIMEOUT_SECONDS`

Persistent admin override:

- `backend/settings.yaml`
- `news.default_poll_interval_minutes`

Public source seed file:

- `backend/news_sources.yaml`

## Time Handling

The subsystem keeps UTC-naive internal timestamps for scheduling/storage consistency and serializes operational timestamps to `Asia/Ho_Chi_Minh` (`GMT+7`) for admin-facing API responses.

This applies to things like:

- ingestion run timestamps
- source status timestamps
- admin monitoring timestamps

Article `published_at` remains source-derived.

## Testing Coverage

There is focused coverage for:

- RSS discovery and feed hub traversal
- sitemap-assisted crawl discovery
- malformed feed parsing fallback
- extraction and cleanup behavior
- dedupe behavior
- semantic storage
- summary generation
- article refresh and repair paths
- admin config and monitoring endpoints

Primary backend tests:

- `backend/tests/test_news_service.py`
- `backend/tests/test_news_api.py`
- `backend/tests/test_news_semantics.py`

## Known Limitations

V1 still has important limits:

- crawl setup remains selector-driven and often site-specific
- no JavaScript-rendered crawling
- no automatic crawl rule generation
- no vector search or semantic retrieval
- RSS discovery is best-effort and not universal
- sitemap-assisted discovery is only a hint for crawl onboarding
- source-specific edge cases may still require extraction or validation refinements

## Operational Notes

When troubleshooting:

1. confirm Alembic schema is up to date
2. restart backend so the News worker starts from FastAPI lifespan
3. check `/api/v1/news/admin/overview`
4. check `/api/v1/news/admin/runs`
5. inspect source validation status in `/api/v1/news/admin/sources`

Useful repair actions:

- refresh source cache
- trigger ingestion
- refresh article content
- repair legacy RSS titles
- apply default poll interval to existing sources

## Recommended Future Work

Likely v2 directions:

- more robust per-site crawl adapters
- better crawl suggestion UX and presets
- more structured source-level analytics
- explicit admin “reclassify article” or “reprocess source” actions
- optional digest or ranking features built on top of the stored semantic layer
