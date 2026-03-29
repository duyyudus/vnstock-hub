# News Subsystem V2

## Purpose

The News subsystem provides a first-class news pipeline for VNStock Hub. In v2 it supports:

- a public latest feed available without login
- private user-managed RSS and crawl sources for signed-in users
- RSS/Atom discovery and deterministic non-RSS crawl onboarding
- article deduplication and ingest-time semantic enrichment
- investor-oriented feed triage through relevance ranking
- portfolio-aware and bookmark-aware feed scopes
- event and catalyst labeling
- story clustering with related-coverage navigation
- per-user blocked-topic filtering
- admin monitoring and repair controls

This document describes the subsystem as currently implemented in v2.

## Scope

V2 is still not a general web intelligence platform. It is a deterministic ingestion system with LLM-assisted content understanding and heuristic investor triage layered on top.

Included in v2:

- background polling of RSS feeds and crawl sources
- public source seeding from YAML
- homepage RSS/Atom discovery
- sitemap-assisted crawl suggestion when RSS is unavailable
- ingest-time semantic classification
- on-demand article summary generation
- portfolio/bookmark-aware relevance scoring
- story grouping and related coverage
- admin monitoring and maintenance actions

Explicitly out of scope in v2:

- vector search or embeddings
- browser-rendered crawling
- LLM-generated crawl rules
- full LLM-first clustering
- push/email/webhook alerts
- a separate watchlist model beyond existing bookmarks and portfolio positions

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
- Dashboard UI:
  - `frontend/src/features/dashboard/news/NewsTab.tsx`
- Shared client types:
  - `frontend/src/api/stockApi.ts`

## Data Model

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
  - topics, tickers, sectors, importance, sentiment, and `raw_payload`
  - `raw_payload` now also carries `event_type`, `event_labels`, and `story_key`
- `news_user_preferences`
  - per-user blocked-topic text and compiled blocked labels
- `news_ingestion_runs`
  - execution history for source polling
- existing cross-subsystem user-interest tables reused by v2:
  - `portfolio_positions`
  - `bookmark_groups`
  - `bookmark_stocks`

Important model behavior:

- canonical URL is unique
- content hash is indexed for secondary dedupe
- one article can originate from multiple feeds/sources
- feed and crawl source polling schedules are source-specific
- story grouping is derived from semantic payload, not stored in a separate table

## Source Types

### Public Default Sources

Public sources are seeded from:

- `backend/news_sources.yaml`

These are loaded at startup by `ensure_public_sources()`.

Current defaults are validated RSS feeds.

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

## Discovery and Validation

### RSS Discovery

`discover_rss_feeds()` is deterministic and currently tries:

1. `<link rel="alternate" ...>` feed tags on the homepage
2. homepage anchors with feed-like URLs or labels
3. common feed paths such as `/rss`, `/rss.html`, `/feed`, `/atom.xml`
4. feed hub pages that link to real category feeds

Supported result kinds:

- `rss`
- `atom`

### Sitemap-Assisted Crawl Discovery

If RSS/Atom discovery returns nothing, `discover_crawl_listings()` may use sitemap metadata as a discovery aid only.

Important rule:

- sitemap is not treated as a permanent feed source

Instead it suggests stable section/listing pages for crawl setup, for example:

- `/kinh-te/`
- `/chung-khoan/`

### Validation

`validate_feed()`:

- fetches the feed URL
- parses RSS or Atom entries
- returns sample titles and entry counts

`validate_crawl_source()`:

- fetches the listing page
- extracts article links using the provided selector
- samples a small number of article pages
- verifies article body extraction works
- returns heuristic selector suggestions

No headless browser is used here.

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

## Content Extraction and Repair

Extraction is deterministic and rule-based.

Current behavior includes:

- article body extraction using prioritized selectors
- noise stripping for related content, banners, tags, share blocks, ads, and other non-body sections
- site-agnostic cleanup rules with targeted resilience for known layouts such as VnEconomy

There is also article repair logic:

- opening detail can trigger content repair for suspicious stored bodies
- explicit refresh can re-fetch and re-extract one article
- when content materially changes, semantic payload is regenerated and stale summaries are cleared

## Deduplication

Deduplication happens in two stages:

1. canonical URL match
2. content hash match

This allows:

- one article discovered through multiple feeds to remain a single canonical row
- attribution to all originating sources through `news_article_sources`

## Semantic Layer

LLM usage in v2 is still limited to content understanding, not crawling or clustering.

### Ingest-Time Classification

For new or changed articles, the subsystem stores:

- `topics`
- `tickers`
- `sectors`
- `importance`
- `sentiment`
- `event_type`
- `event_labels`
- `story_key`
- raw payload for debugging/inspection

Classification is performed by `classify_article()`.

The classifier may use an OpenAI-compatible provider, but it has heuristic fallback behavior when providers are unavailable.

### Provider Fallback

The subsystem supports multiple OpenAI-compatible providers from shared LLM configuration.

Current behavior:

- skips obviously unusable providers
- cools down failing providers
- treats empty or malformed “success” responses as failures
- falls back across configured providers

### Blocked Topics

User preference flow:

1. user saves freeform “topics I don’t care about”
2. backend compiles that text into normalized blocked labels
3. personalized feed excludes matching articles

This remains exclusion-only.

## Summaries

LLM summaries are still on-demand only.

Important v2 behavior:

- `llm_summary` is not generated by default at ingest time
- feed cards show original excerpt unless a summary has been generated
- `Summary` generates and stores the first summary
- `Regenerate` overwrites the stored summary
- summaries are requested in the article’s original language

If article content is refreshed and materially changes, stale `llm_summary` is cleared.

## Investor Triage Layer

V2 adds a read-time investor triage layer on top of stored articles.

### Feed Modes

The frontend exposes these top-level views:

- `Latest`
- `For You`
- `Portfolio`
- `Bookmarks`

Current frontend default:

- `Latest`

### Feed Query Surface

`GET /api/v1/news/feed` now supports:

- `source`
- `topic`
- `ticker`
- `from`
- `to`
- `cursor`
- `limit`
- `sort`
  - `latest`
  - `relevance`
- `scope`
  - `all`
  - `portfolio`
  - `bookmarks`
- `bookmark_group_id`
- `event_type`
- `importance`
- `group_by`
  - `article`
  - `story`

### Relevance Ranking

Relevance ranking is heuristic and request-time only. It does not currently invoke an LLM.

Inputs include:

- matched portfolio tickers
- matched bookmark tickers
- importance
- recency
- source multiplicity

Associated explanatory fields returned in feed/detail payloads:

- `matched_tickers`
- `why_relevant`

Current `why_relevant` reasons are intentionally limited to:

- portfolio matches
- bookmark matches
- multi-source coverage
- cluster size when a grouped story contains multiple related articles

### Event and Catalyst Labels

Feed and detail payloads expose:

- `event_type`
- `event_labels`

These are used for:

- event filtering
- feed-card badges
- detail badges

### Story Clustering

V2 adds lightweight story clustering as a presentation layer.

Important implementation details:

- clustering is heuristic, not LLM-based
- `story_key` is derived from:
  - primary ticker
  - short time bucket
  - normalized title signature
- articles are still stored individually
- grouping only changes how feed items are presented

Returned clustering fields:

- `story_key`
- `story_source_count`
- `related_article_ids`

Article detail also returns:

- `related_articles`

### Scope Integration

V2 reuses existing user-interest sources instead of creating a new watchlist model.

Supported interest sources:

- `portfolio_positions`
- `bookmark_groups` and `bookmark_stocks`

Scope behavior:

- anonymous users always operate effectively in public/latest mode
- signed-in users can request relevance ranking
- `scope=portfolio` restricts to articles matching held tickers
- `scope=bookmarks` restricts to articles matching bookmark tickers
- `bookmark_group_id` can narrow bookmark scope to one group

## Feed Behavior

### Anonymous Feed

Anonymous users see:

- public default sources only
- latest sort only in practice

### Signed-In Feed

Signed-in users see:

- public default sources
- plus their own private subscriptions
- optional relevance ranking against portfolio and bookmarks
- blocked-topic filtering after semantic matching

Current frontend page size is 20 items per request.

Current pagination behavior for the v2 feed is offset-style cursor pagination, returned through `next_cursor`.

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
- shared LLM provider settings
- shared `LLM_REQUEST_TIMEOUT_SECONDS`

Persistent admin override:

- `backend/settings.yaml`
- `news.default_poll_interval_minutes`

Public source seed file:

- `backend/news_sources.yaml`

## Time Handling

The subsystem keeps UTC-naive internal timestamps for scheduling/storage consistency and serializes operational timestamps to `Asia/Ho_Chi_Minh` (`GMT+7`) for admin-facing API responses.

This applies to:

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
- semantic storage and fallback behavior
- summary generation
- event-type classification
- relevance ranking
- portfolio and bookmark scope filtering
- story grouping and related coverage
- article refresh and repair paths
- admin config and monitoring endpoints

Primary backend tests:

- `backend/tests/test_news_service.py`
- `backend/tests/test_news_api.py`
- `backend/tests/test_news_semantics.py`

## Known Limitations

Important current limits:

- crawl setup remains selector-driven and often site-specific
- no JavaScript-rendered crawling
- no automatic crawl rule generation
- no vector search or semantic retrieval
- RSS discovery is best-effort and not universal
- sitemap-assisted discovery is only a hint for crawl onboarding
- story clustering is heuristic and can miss or over-group edge cases
- relevance scoring is heuristic and request-time only
- no proactive alert delivery yet

## Operational Notes

When troubleshooting:

1. confirm Alembic schema is up to date
2. restart backend so the News worker starts from FastAPI lifespan
3. check `/api/v1/news/admin/overview`
4. check `/api/v1/news/admin/runs`
5. inspect source validation status in `/api/v1/news/admin/sources`
6. compare feed behavior in `latest` vs `relevance` mode when triage output looks suspicious

Useful repair actions:

- refresh source cache
- trigger ingestion
- refresh article content
- repair legacy RSS titles
- apply default poll interval to existing sources

## Recommended Future Work

Likely next directions after v2:

- selective LLM adjudication for ambiguous story-clustering cases
- more robust per-site crawl adapters
- better crawl suggestion UX and presets
- more structured source-level analytics
- explicit admin reclassify/reprocess actions
- alert and digest delivery built on top of the stored semantic layer
