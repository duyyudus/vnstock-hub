from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from sqlalchemy import select

from app.db.models import NewsArticle, NewsArticleSemantic, NewsArticleSource, NewsCrawlSource, NewsIngestionRun, NewsQuickGlanceDigest, NewsSite, NewsSiteFeed
from app.services.news import news_service
from app.services.news import service as news_service_module
from app.services.news.service import NEWS_STALE_RUN_MINUTES, utc_naive_to_local_iso
from app.services.news.discovery import (
    discover_crawl_listings,
    discover_rss_feeds,
    extract_article_payload,
    parse_homepage_feed_candidates,
    parse_feed_entries,
    parse_feed_title,
    parse_sitemap_listing_candidates,
)


async def _seed_quick_glance_article(
    db_session,
    *,
    feed: NewsSiteFeed,
    title: str,
    canonical_url: str,
    published_at: datetime,
    excerpt: str = "Article excerpt",
    content_text: str = "Article body content.",
    tickers: list[str] | None = None,
    topics: list[str] | None = None,
    importance: str = "high",
    sentiment: str = "positive",
    event_type: str = "earnings",
    story_key: str | None = None,
) -> NewsArticle:
    article = NewsArticle(
        canonical_url=canonical_url,
        title=title,
        excerpt=excerpt,
        content_text=content_text,
        published_at=published_at,
        language="en",
        content_hash=f"hash-{canonical_url.rsplit('/', 1)[-1]}",
        created_at=published_at,
        updated_at=published_at,
    )
    db_session.add(article)
    await db_session.flush()
    db_session.add(NewsArticleSource(article_id=article.id, site_feed_id=feed.id, article_url=canonical_url))
    db_session.add(
        NewsArticleSemantic(
            article_id=article.id,
            topics=topics or ["earnings"],
            tickers=tickers or ["ABC"],
            sectors=["technology"],
            importance=importance,
            sentiment=sentiment,
            raw_payload={
                "event_type": event_type,
                "event_labels": [event_type.replace("_", " ")],
                "story_key": story_key or f"{tickers[0] if tickers else 'ABC'}|story|{article.id}",
                "display_topics": topics or ["Earnings"],
            },
            classified_at=published_at,
            updated_at=published_at,
        )
    )
    await db_session.flush()
    return article


def test_parse_homepage_feed_candidates_discovers_multiple_feeds():
    html = """
    <html>
      <head>
        <link rel="alternate" type="application/rss+xml" title="Main" href="/rss.xml" />
        <link rel="alternate" type="application/atom+xml" title="Markets" href="https://example.com/markets.xml" />
      </head>
      <body>
        <a href="/feed/economy.xml">Economy feed</a>
      </body>
    </html>
    """

    candidates = parse_homepage_feed_candidates("https://example.com", html)
    urls = {candidate["feed_url"] for candidate in candidates}
    assert "https://example.com/rss.xml" in urls
    assert "https://example.com/markets.xml" in urls
    assert "https://example.com/feed/economy.xml" in urls


def test_parse_homepage_feed_candidates_detects_feed_hub_links_by_label():
    html = """
    <html>
      <body>
        <a href="/rss.html" aria-label="RSS feed directory">
          <img alt="RSS" src="/rss-icon.svg" />
        </a>
      </body>
    </html>
    """

    candidates = parse_homepage_feed_candidates("https://example.com", html)
    urls = {candidate["feed_url"] for candidate in candidates}

    assert "https://example.com/rss.html" in urls


def test_parse_feed_entries_falls_back_for_malformed_xml():
    malformed_feed = """
    <rss><channel>
      <item>
        <title>CafeF headline</title>
        <link>https://cafef.vn/story.chn</link>
        <description><![CDATA[Messy <b>markup</b> &nbsp; text]]></description>
        <pubDate>Wed, 26 Mar 2026 10:30:00 +0700</pubDate>
      </item>
    </channel></rss>
    """

    entries = parse_feed_entries(malformed_feed)
    assert len(entries) == 1
    assert entries[0].title == "CafeF headline"
    assert entries[0].link == "https://cafef.vn/story.chn"


def test_parse_feed_title_prefers_channel_title_over_item_title():
    feed_xml = """
    <rss><channel>
      <title>CafeF - Thi truong chung khoan</title>
      <item>
        <title>CafeF headline</title>
        <link>https://cafef.vn/story.chn</link>
      </item>
    </channel></rss>
    """

    assert parse_feed_title(feed_xml) == "CafeF - Thi truong chung khoan"


def test_parse_sitemap_listing_candidates_returns_stable_section_pages():
    categories_sitemap = """
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url>
        <loc>https://example.com/kinh-te/</loc>
        <lastmod>2026-03-27T10:20:05+07:00</lastmod>
      </url>
      <url>
        <loc>https://example.com/chung-khoan/</loc>
        <lastmod>2026-03-27T10:20:05+07:00</lastmod>
      </url>
      <url>
        <loc>https://example.com/epaper/</loc>
        <lastmod>2026-03-27T10:20:05+07:00</lastmod>
      </url>
    </urlset>
    """

    candidates = parse_sitemap_listing_candidates("https://example.com", categories_sitemap)

    assert [candidate["listing_url"] for candidate in candidates] == [
        "https://example.com/kinh-te/",
        "https://example.com/chung-khoan/",
    ]


def test_extract_article_payload_strips_related_blocks_from_vneconomy_layout():
    html = """
    <html lang="vi">
      <head>
        <meta property="og:title" content="VCCI đề xuất giảm thuế" />
        <meta property="og:description" content="Sapo summary" />
      </head>
      <body>
        <main>
          <h1 class="name-detail">VCCI đề xuất giảm thuế</h1>
          <div class="main-detail-page">
            <div class="ct-edtior-web news-type1">
              <div class="news-sapo"><p>Sapo summary</p></div>
              <div>
                <p>Đoạn 1 của bài viết.</p>
                <p>Đoạn 2 của bài viết.</p>
              </div>
              <div class="list-detail-revert_item">
                <h3>Đọc thêm</h3>
                <p>Bài liên quan không nên xuất hiện.</p>
              </div>
              <div class="box-keyword">
                <h3>Từ khóa:</h3>
                <p>thuế bảo vệ môi trường</p>
              </div>
            </div>
          </div>
        </main>
      </body>
    </html>
    """

    payload = extract_article_payload("https://vneconomy.vn/example.htm", html)

    assert payload["title"] == "VCCI đề xuất giảm thuế"
    assert payload["excerpt"] == "Sapo summary"
    assert payload["content_text"] == "Sapo summary\nĐoạn 1 của bài viết.\nĐoạn 2 của bài viết."
    assert "Bài liên quan không nên xuất hiện." not in payload["content_text"]
    assert "Từ khóa:" not in payload["content_text"]


def test_extract_article_payload_supports_sggp_dttc_article_body_layout():
    html = """
    <html lang="vi">
      <head>
        <meta property="og:title" content="Email của Giám đốc FBI bị hacker xâm nhập" />
        <meta property="og:description" content="(ĐTTCO) - Cục Điều tra liên bang Mỹ (FBI) ngày 29-3 xác nhận tin tặc đã xâm nhập hộp thư điện tử cá nhân của Giám đốc Kash Patel." />
      </head>
      <body>
        <div class="article">
          <div class="article__sapo cms-desc">
            <p>(ĐTTCO) - Cục Điều tra liên bang Mỹ (FBI) ngày 29-3 xác nhận tin tặc đã xâm nhập hộp thư điện tử cá nhân của Giám đốc Kash Patel.</p>
          </div>
          <div class="article__body zce-content-body cms-body" itemprop="articleBody">
            <p>Theo FBI, dữ liệu bị đánh cắp không phải thông tin mới và không bao gồm bất kỳ dữ liệu nào của chính phủ.</p>
            <p>Nhóm này sau đó đăng tải một số ảnh và tài liệu cá nhân, cùng các email từ thời điểm trước khi ông Patel đảm nhiệm chức vụ Giám đốc FBI.</p>
            <div class="related-news">
              <article class="story">
                <h2>Hacker chỉ mất vài giây để xâm nhập tai nghe của bạn</h2>
              </article>
            </div>
          </div>
        </div>
      </body>
    </html>
    """

    payload = extract_article_payload("https://dttc.sggp.org.vn/email-cua-giam-doc-fbi-bi-hacker-xam-nhap-post132660.html", html)

    assert payload["title"] == "Email của Giám đốc FBI bị hacker xâm nhập"
    assert payload["excerpt"].startswith("(ĐTTCO) - Cục Điều tra liên bang Mỹ")
    assert payload["content_text"] == (
        "Theo FBI, dữ liệu bị đánh cắp không phải thông tin mới và không bao gồm bất kỳ dữ liệu nào của chính phủ.\n"
        "Nhóm này sau đó đăng tải một số ảnh và tài liệu cá nhân, cùng các email từ thời điểm trước khi ông Patel đảm nhiệm chức vụ Giám đốc FBI."
    )
    assert "Hacker chỉ mất vài giây" not in payload["content_text"]


def test_matches_topic_filter_supports_case_insensitive_partial_phrases():
    assert news_service_module._matches_topic_filter("bank", ["banking"]) is True
    assert news_service_module._matches_topic_filter("bank", ["macro banking", "rates"]) is True
    assert news_service_module._matches_topic_filter("Bank", ["banking"]) is True
    assert news_service_module._matches_topic_filter("interest rates", ["global interest rates"]) is True
    assert news_service_module._matches_topic_filter("interest rates", ["interest", "rates"]) is False
    assert news_service_module._matches_topic_filter("commodities", ["banking", "interest rates"]) is False


def test_discussion_search_queries_prioritize_subject_queries_for_background_requests():
    article = NewsArticle(
        canonical_url="https://vietnambiz.vn/example.htm",
        title="Chấm dứt thương vụ với đại gia Thái, Home Credit đang kinh doanh ra sao?",
        excerpt="Excerpt",
        content_text="Content",
    )

    queries = news_service._heuristic_discussion_search_queries(
        article,
        None,
        [{"role": "user", "content": "giới thiệu sơ về home credit"}],
        override=None,
    )

    assert queries[0].startswith('"home credit"')
    assert "Chấm dứt thương vụ" not in queries[0]


def test_discussion_search_queries_fall_back_to_article_anchor_for_specific_follow_up():
    article = NewsArticle(
        canonical_url="https://vietnambiz.vn/example.htm",
        title="Chấm dứt thương vụ với đại gia Thái, Home Credit đang kinh doanh ra sao?",
        excerpt="Excerpt",
        content_text="Content",
    )

    queries = news_service._heuristic_discussion_search_queries(
        article,
        None,
        [{"role": "user", "content": "thương vụ này đổ vỡ vì sao"}],
        override=None,
    )

    assert queries[0] == "thương vụ này đổ vỡ vì sao"
    assert any(query.startswith("Chấm dứt thương vụ") for query in queries)


def test_should_auto_search_discussion_for_overview_questions():
    article_context = {
        "title": "Home Credit story",
        "excerpt": "Short excerpt",
        "content_text": "This article body is detailed enough to stand on its own for article recap questions.",
    }

    should_search = news_service._should_auto_search_discussion(
        intent="overview",
        article_context=article_context,
        latest_user_message="giới thiệu sơ về home credit",
    )

    assert should_search is True


def test_should_auto_search_discussion_stays_off_for_recap_when_article_is_strong():
    article_context = {
        "title": "Home Credit story",
        "excerpt": "Short excerpt",
        "content_text": "This article body is detailed enough to stand on its own for article recap questions. " * 12,
    }

    should_search = news_service._should_auto_search_discussion(
        intent="recap",
        article_context=article_context,
        latest_user_message="tóm tắt bài này",
    )

    assert should_search is False


@pytest.mark.asyncio
async def test_discussion_search_queries_merge_llm_and_heuristic_queries(monkeypatch):
    article = NewsArticle(
        canonical_url="https://vietnambiz.vn/example.htm",
        title="Chấm dứt thương vụ với đại gia Thái, Home Credit đang kinh doanh ra sao?",
        excerpt="Excerpt",
        content_text="Content",
    )
    article_context = {"title": article.title, "canonical_url": article.canonical_url}

    async def _fake_generate_discussion_search_queries(*, article_context, messages, fallback_queries):
        assert article_context["title"] == article.title
        assert messages[-1]["content"] == "giới thiệu sơ về home credit"
        assert fallback_queries
        return [
            "\"home credit\" vietnam consumer finance overview",
            "\"home credit\" viet nam la cong ty gi",
        ]

    monkeypatch.setattr(news_service_module, "generate_discussion_search_queries", _fake_generate_discussion_search_queries)

    queries = await news_service._discussion_search_queries(
        article,
        None,
        [{"role": "user", "content": "giới thiệu sơ về home credit"}],
        override=None,
        article_context=article_context,
    )

    assert queries[0] == "\"home credit\" vietnam consumer finance overview"
    assert any(query.startswith('"home credit" Việt Nam') for query in queries)


@pytest.mark.asyncio
async def test_build_web_discussion_evidence_retries_and_promotes_profile_result(monkeypatch):
    article = NewsArticle(
        canonical_url="https://vietnambiz.vn/example.htm",
        title="Chấm dứt thương vụ với đại gia Thái, Home Credit đang kinh doanh ra sao?",
        excerpt="Excerpt",
        content_text="Content",
    )
    article_context = {"title": article.title, "canonical_url": article.canonical_url}
    observed_queries: list[str] = []

    class _FakeSearchProvider:
        async def search(self, query: str, *, limit: int = 5):
            observed_queries.append(query)
            if "company overview official" in query or "giới thiệu công ty" in query:
                return [
                    SimpleNamespace(
                        title="Home Credit Vietnam overview",
                        url="https://homecredit.vn/about",
                        snippet="Company overview and official introduction.",
                        domain="homecredit.vn",
                    )
                ]
            return [
                SimpleNamespace(
                    title="Ngân hàng Thái Lan hủy thương vụ mua Home Credit Việt Nam",
                    url="https://znews.vn/home-credit-deal-post.html",
                    snippet="Deal coverage about the failed transaction.",
                    domain="znews.vn",
                )
            ]

    async def _fake_generate_discussion_search_queries(*, article_context, messages, fallback_queries):
        return ['"home credit" overview']

    async def _fake_fetch_text(client, url: str):
        if url == "https://homecredit.vn/about":
            return """
            <html>
              <head>
                <meta property="og:title" content="Home Credit Vietnam overview" />
                <meta property="og:description" content="Company overview and official introduction." />
              </head>
              <body><article><p>Home Credit Vietnam is a consumer finance company offering lending products and digital services.</p></article></body>
            </html>
            """
        return """
        <html>
          <head>
            <meta property="og:title" content="Ngân hàng Thái Lan hủy thương vụ mua Home Credit Việt Nam" />
            <meta property="og:description" content="Deal coverage about the failed transaction." />
          </head>
          <body><article><p>Deal coverage about the failed transaction.</p></article></body>
        </html>
        """

    monkeypatch.setattr(news_service_module, "generate_discussion_search_queries", _fake_generate_discussion_search_queries)
    monkeypatch.setattr(news_service_module, "get_news_search_provider", lambda client: _FakeSearchProvider())
    monkeypatch.setattr(news_service_module, "fetch_text", _fake_fetch_text)

    evidence_items, result_count = await news_service._build_web_discussion_evidence(
        article,
        None,
        [{"role": "user", "content": "giới thiệu sơ về home credit"}],
        search_query_override=None,
        article_context=article_context,
    )

    assert result_count >= 2
    assert evidence_items[0]["url"] == "https://homecredit.vn/about"
    assert any("company overview official" in query or "giới thiệu công ty" in query for query in observed_queries)


@pytest.mark.asyncio
async def test_discover_rss_feeds_follows_rss_hub_pages():
    homepage_html = """
    <html>
      <body>
        <a href="/rss.html">RSS</a>
      </body>
    </html>
    """
    hub_html = """
    <html>
      <body>
        <a href="/thi-truong/rss">Thi truong</a>
        <a href="/chung-khoan/rss">Chung khoan</a>
      </body>
    </html>
    """
    feed_xml = """
    <rss><channel>
      <item>
        <title>Headline</title>
        <link>https://example.com/articles/1</link>
        <description>Summary</description>
        <pubDate>Wed, 26 Mar 2026 10:30:00 +0700</pubDate>
      </item>
    </channel></rss>
    """

    async def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/":
            return httpx.Response(200, text=homepage_html)
        if request.url.path == "/rss.html":
            return httpx.Response(200, text=hub_html)
        if request.url.path in {"/thi-truong/rss", "/chung-khoan/rss"}:
            return httpx.Response(200, text=feed_xml)
        return httpx.Response(404, text="not found")

    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport, base_url="https://example.com") as client:
        discovered = await discover_rss_feeds(client, "https://example.com/")

    urls = {item["feed_url"] for item in discovered}
    assert "https://example.com/thi-truong/rss" in urls
    assert "https://example.com/chung-khoan/rss" in urls


@pytest.mark.asyncio
async def test_discover_rss_feeds_follows_plain_rss_directory_hub_pages():
    homepage_html = """
    <html>
      <body>
        <p>No explicit feed links on homepage.</p>
      </body>
    </html>
    """
    hub_html = """
    <html>
      <body>
        <a href="/rss/tin-moi-nhat.rss">Tin moi nhat</a>
        <a href="/rss/kinh-doanh.rss">Kinh doanh</a>
      </body>
    </html>
    """
    feed_xml = """
    <rss><channel>
      <item>
        <title>Headline</title>
        <link>https://example.com/articles/1</link>
        <description>Summary</description>
        <pubDate>Wed, 26 Mar 2026 10:30:00 +0700</pubDate>
      </item>
    </channel></rss>
    """

    async def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/":
            return httpx.Response(200, text=homepage_html)
        if request.url.path == "/rss":
            return httpx.Response(200, text=hub_html)
        if request.url.path in {"/rss/tin-moi-nhat.rss", "/rss/kinh-doanh.rss"}:
            return httpx.Response(200, text=feed_xml)
        return httpx.Response(404, text="not found")

    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport, base_url="https://example.com") as client:
        discovered = await discover_rss_feeds(client, "https://example.com/")

    urls = {item["feed_url"] for item in discovered}
    assert "https://example.com/rss/tin-moi-nhat.rss" in urls
    assert "https://example.com/rss/kinh-doanh.rss" in urls


@pytest.mark.asyncio
async def test_discover_crawl_listings_falls_back_to_category_sitemaps():
    homepage_html = """
    <html>
      <body>
        <p>No explicit feed links on homepage.</p>
      </body>
    </html>
    """
    sitemap_index = """
    <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <sitemap>
        <loc>https://example.com/sitemaps/categories.xml</loc>
        <lastmod>2026-03-27</lastmod>
      </sitemap>
    </sitemapindex>
    """
    categories_sitemap = """
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url>
        <loc>https://example.com/kinh-te/</loc>
        <lastmod>2026-03-27T07:43:16+07:00</lastmod>
      </url>
      <url>
        <loc>https://example.com/chung-khoan/</loc>
        <lastmod>2026-03-27T07:43:16+07:00</lastmod>
      </url>
    </urlset>
    """

    async def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/":
            return httpx.Response(200, text=homepage_html)
        if request.url.path in {"/rss", "/rss.xml", "/rss.html", "/rss.htm", "/feed", "/feed.xml", "/atom.xml"}:
            return httpx.Response(404, text="not found")
        if request.url.path == "/sitemap.xml":
            return httpx.Response(200, text=sitemap_index)
        if request.url.path == "/sitemaps/categories.xml":
            return httpx.Response(200, text=categories_sitemap)
        return httpx.Response(404, text="not found")

    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport, base_url="https://example.com") as client:
        discovered = await discover_crawl_listings(client, "https://example.com/")

    assert [item["listing_url"] for item in discovered] == [
        "https://example.com/kinh-te/",
        "https://example.com/chung-khoan/",
    ]
    assert discovered[0]["discovery_method"] == "sitemap"


@pytest.mark.asyncio
async def test_store_article_candidate_dedupes_by_content_hash(db_session, monkeypatch):
    async def _fake_classify_article(title: str, excerpt: str | None, content_text: str | None):
        return {
            "topics": ["earnings"],
            "tickers": ["ABC"],
            "sectors": ["technology"],
            "importance": "high",
            "sentiment": "positive",
            "raw_payload": {"topics": ["earnings"]},
        }

    monkeypatch.setattr("app.services.news.service.classify_article", _fake_classify_article)

    site = NewsSite(
        domain="example.com",
        homepage_url="https://example.com",
        display_name="Example",
        is_public=True,
    )
    db_session.add(site)
    await db_session.flush()

    first_feed = NewsSiteFeed(
        site_id=site.id,
        feed_url="https://example.com/rss.xml",
        title="Feed One",
        kind="rss",
        discovery_method="seed",
        validation_status="valid",
        is_public=True,
    )
    second_feed = NewsSiteFeed(
        site_id=site.id,
        feed_url="https://example.com/markets.xml",
        title="Feed Two",
        kind="rss",
        discovery_method="seed",
        validation_status="valid",
        is_public=True,
    )
    db_session.add(first_feed)
    db_session.add(second_feed)
    await db_session.flush()

    stored_one = await news_service._store_article_candidate(
        db_session,
        source_type="rss",
        site_feed_id=first_feed.id,
        crawl_source_id=None,
        article_url="https://example.com/articles/first",
        title="ABC earnings rise",
        excerpt="Revenue improves",
        published_at=datetime(2026, 3, 26, 12, 0, 0),
        content_text="ABC earnings rise and revenue improves.",
        language="en",
        image_url=None,
    )
    stored_two = await news_service._store_article_candidate(
        db_session,
        source_type="rss",
        site_feed_id=second_feed.id,
        crawl_source_id=None,
        article_url="https://example.com/articles/duplicate",
        title="ABC earnings rise",
        excerpt="Revenue improves",
        published_at=datetime(2026, 3, 26, 12, 1, 0),
        content_text="ABC earnings rise and revenue improves.",
        language="en",
        image_url=None,
    )
    await db_session.commit()

    articles = (await db_session.execute(select(NewsArticle))).scalars().all()
    mappings = (await db_session.execute(select(NewsArticleSource))).scalars().all()
    semantics = (await db_session.execute(select(NewsArticleSemantic))).scalars().all()

    assert stored_one == 1
    assert stored_two == 1
    assert len(articles) == 1
    assert len(mappings) == 2
    assert len(semantics) == 1


@pytest.mark.asyncio
async def test_generate_article_summary_persists_and_prefers_llm_summary(db_session, monkeypatch):
    async def _fake_summarize_article(title: str, excerpt: str | None, content_text: str | None, *, language: str | None = None):
        assert title == "ABC earnings rise"
        assert excerpt == "Original excerpt"
        assert content_text == "Longer body text for the article."
        assert language == "en"
        return "LLM generated summary"

    monkeypatch.setattr("app.services.news.service.summarize_article", _fake_summarize_article)

    site = NewsSite(
        domain="example.com",
        homepage_url="https://example.com",
        display_name="Example",
        is_public=True,
    )
    db_session.add(site)
    await db_session.flush()

    feed = NewsSiteFeed(
        site_id=site.id,
        feed_url="https://example.com/rss.xml",
        title="Feed One",
        kind="rss",
        discovery_method="seed",
        validation_status="valid",
        is_public=True,
    )
    db_session.add(feed)
    await db_session.flush()

    article = NewsArticle(
        canonical_url="https://example.com/articles/first",
        title="ABC earnings rise",
        excerpt="Original excerpt",
        content_text="Longer body text for the article.",
        published_at=datetime(2026, 3, 26, 12, 0, 0),
        language="en",
        content_hash="hash-1",
    )
    db_session.add(article)
    await db_session.flush()
    db_session.add(NewsArticleSource(article_id=article.id, site_feed_id=feed.id, article_url=article.canonical_url))
    db_session.add(
        NewsArticleSemantic(
            article_id=article.id,
            topics=["earnings"],
            tickers=["ABC"],
            sectors=["technology"],
            importance="high",
            sentiment="positive",
            raw_payload={},
        )
    )
    await db_session.commit()

    payload = await news_service.generate_article_summary(article.id, user_id=None)
    await db_session.refresh(article)

    assert payload is not None
    assert payload["excerpt"] == "LLM generated summary"
    assert payload["original_excerpt"] == "Original excerpt"
    assert payload["llm_summary"] == "LLM generated summary"
    assert article.llm_summary == "LLM generated summary"


@pytest.mark.asyncio
async def test_generate_article_summary_force_refresh_overwrites_existing_summary(db_session, monkeypatch):
    calls: list[str] = []

    async def _fake_summarize_article(title: str, excerpt: str | None, content_text: str | None, *, language: str | None = None):
        calls.append("called")
        return "Fresh regenerated summary"

    monkeypatch.setattr("app.services.news.service.summarize_article", _fake_summarize_article)

    site = NewsSite(
        domain="example.com",
        homepage_url="https://example.com",
        display_name="Example",
        is_public=True,
    )
    db_session.add(site)
    await db_session.flush()

    feed = NewsSiteFeed(
        site_id=site.id,
        feed_url="https://example.com/rss.xml",
        title="Feed One",
        kind="rss",
        discovery_method="seed",
        validation_status="valid",
        is_public=True,
    )
    db_session.add(feed)
    await db_session.flush()

    article = NewsArticle(
        canonical_url="https://example.com/articles/force",
        title="ABC earnings rise",
        excerpt="Original excerpt",
        llm_summary="Old stored summary",
        content_text="Longer body text for the article.",
        published_at=datetime(2026, 3, 26, 12, 0, 0),
        language="en",
        content_hash="hash-force",
    )
    db_session.add(article)
    await db_session.flush()
    db_session.add(NewsArticleSource(article_id=article.id, site_feed_id=feed.id, article_url=article.canonical_url))
    await db_session.commit()

    payload = await news_service.generate_article_summary(article.id, user_id=None, force_refresh=True)
    await db_session.refresh(article)

    assert calls == ["called"]
    assert payload is not None
    assert payload["excerpt"] == "Fresh regenerated summary"
    assert payload["llm_summary"] == "Fresh regenerated summary"
    assert article.llm_summary == "Fresh regenerated summary"


@pytest.mark.asyncio
async def test_get_quick_glance_digest_returns_structured_payload_and_persists_cache(db_session, monkeypatch):
    site = NewsSite(
        domain="example.com",
        homepage_url="https://example.com",
        display_name="Example",
        is_public=True,
    )
    db_session.add(site)
    await db_session.flush()

    feed = NewsSiteFeed(
        site_id=site.id,
        feed_url="https://example.com/rss.xml",
        title="Feed One",
        kind="rss",
        discovery_method="seed",
        validation_status="valid",
        is_public=True,
    )
    db_session.add(feed)
    await db_session.flush()

    first_article = await _seed_quick_glance_article(
        db_session,
        feed=feed,
        title="ABC earnings rise",
        canonical_url="https://example.com/articles/quick-1",
        published_at=datetime(2026, 3, 26, 10, 0, 0),
        tickers=["ABC"],
        story_key="ABC|story|1",
    )
    second_article = await _seed_quick_glance_article(
        db_session,
        feed=feed,
        title="XYZ wins new contract",
        canonical_url="https://example.com/articles/quick-2",
        published_at=datetime(2026, 3, 26, 9, 30, 0),
        tickers=["XYZ"],
        topics=["contracts"],
        event_type="other",
        story_key="XYZ|story|2",
    )
    await db_session.commit()

    monkeypatch.setattr(news_service_module, "utc_now", lambda: datetime(2026, 3, 26, 12, 0, 0))

    async def _fake_generate_quick_glance_digest(*, window_hours: int, article_count: int, highlights_target: int, evidence_items: list[dict]):
        assert window_hours == 24
        assert article_count == 2
        assert highlights_target == 5
        assert [item["article_id"] for item in evidence_items[:2]] == [first_article.id, second_article.id]
        return {
            "summary": "ABC earnings and XYZ contract wins dominated the session.",
            "highlights": [
                {
                    "title": "ABC led earnings coverage",
                    "body": "ABC posted the highest-importance story in the window.",
                    "article_ids": [first_article.id],
                }
            ],
        }

    monkeypatch.setattr(news_service_module, "generate_quick_glance_digest", _fake_generate_quick_glance_digest)

    payload = await news_service.get_quick_glance_digest(user_id=None, window_hours=24)

    cached_rows = (await db_session.execute(select(NewsQuickGlanceDigest))).scalars().all()

    assert payload["article_count"] == 2
    assert payload["summary"] == "ABC earnings and XYZ contract wins dominated the session."
    assert payload["highlights"][0]["article_ids"] == [first_article.id]
    assert payload["key_articles"][0]["id"] == first_article.id
    assert payload["cache_hit"] is False
    assert len(cached_rows) == 1


@pytest.mark.asyncio
async def test_get_quick_glance_digest_returns_empty_without_calling_llm(db_session, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(news_service_module, "utc_now", lambda: datetime(2026, 3, 26, 12, 0, 0))

    async def _fake_generate_quick_glance_digest(**kwargs):
        calls.append("called")
        return {"summary": "Should not happen", "highlights": []}

    monkeypatch.setattr(news_service_module, "generate_quick_glance_digest", _fake_generate_quick_glance_digest)

    payload = await news_service.get_quick_glance_digest(user_id=None, window_hours=24)

    assert payload["article_count"] == 0
    assert payload["summary"] is None
    assert payload["highlights"] == []
    assert calls == []


@pytest.mark.asyncio
async def test_get_quick_glance_digest_reuses_cached_result_when_articles_unchanged(db_session, monkeypatch):
    site = NewsSite(
        domain="example.com",
        homepage_url="https://example.com",
        display_name="Example",
        is_public=True,
    )
    db_session.add(site)
    await db_session.flush()

    feed = NewsSiteFeed(
        site_id=site.id,
        feed_url="https://example.com/rss.xml",
        title="Feed One",
        kind="rss",
        discovery_method="seed",
        validation_status="valid",
        is_public=True,
    )
    db_session.add(feed)
    await db_session.flush()

    article = await _seed_quick_glance_article(
        db_session,
        feed=feed,
        title="ABC earnings rise",
        canonical_url="https://example.com/articles/cache-1",
        published_at=datetime(2026, 3, 26, 10, 0, 0),
        tickers=["ABC"],
        story_key="ABC|story|cache",
    )
    await db_session.commit()

    monkeypatch.setattr(news_service_module, "utc_now", lambda: datetime(2026, 3, 26, 12, 0, 0))
    calls: list[int] = []

    async def _fake_generate_quick_glance_digest(**kwargs):
        calls.append(1)
        return {
            "summary": "ABC remained the only high-signal story.",
            "highlights": [
                {
                    "title": "ABC only",
                    "body": "One article carried the window.",
                    "article_ids": [article.id],
                }
            ],
        }

    monkeypatch.setattr(news_service_module, "generate_quick_glance_digest", _fake_generate_quick_glance_digest)

    first_payload = await news_service.get_quick_glance_digest(user_id=None, window_hours=24)
    second_payload = await news_service.get_quick_glance_digest(user_id=None, window_hours=24)

    assert first_payload["cache_hit"] is False
    assert second_payload["cache_hit"] is True
    assert first_payload["summary"] == second_payload["summary"]
    assert calls == [1]


@pytest.mark.asyncio
async def test_get_quick_glance_digest_force_refresh_regenerates_cached_digest(db_session, monkeypatch):
    site = NewsSite(
        domain="example.com",
        homepage_url="https://example.com",
        display_name="Example",
        is_public=True,
    )
    db_session.add(site)
    await db_session.flush()

    feed = NewsSiteFeed(
        site_id=site.id,
        feed_url="https://example.com/rss.xml",
        title="Feed One",
        kind="rss",
        discovery_method="seed",
        validation_status="valid",
        is_public=True,
    )
    db_session.add(feed)
    await db_session.flush()

    article = await _seed_quick_glance_article(
        db_session,
        feed=feed,
        title="ABC earnings rise",
        canonical_url="https://example.com/articles/force-refresh",
        published_at=datetime(2026, 3, 26, 10, 0, 0),
        tickers=["ABC"],
        story_key="ABC|story|force-refresh",
    )
    await db_session.commit()

    monkeypatch.setattr(news_service_module, "utc_now", lambda: datetime(2026, 3, 26, 12, 0, 0))
    calls: list[int] = []

    async def _fake_generate_quick_glance_digest(**kwargs):
        calls.append(1)
        return {
            "summary": f"Digest version {len(calls)}",
            "highlights": [
                {
                    "title": "Top story",
                    "body": "ABC remained the top story.",
                    "article_ids": [article.id],
                }
            ],
        }

    monkeypatch.setattr(news_service_module, "generate_quick_glance_digest", _fake_generate_quick_glance_digest)

    first_payload = await news_service.get_quick_glance_digest(user_id=None, window_hours=24)
    second_payload = await news_service.get_quick_glance_digest(user_id=None, window_hours=24, force_refresh=True)

    cached_rows = (await db_session.execute(select(NewsQuickGlanceDigest))).scalars().all()

    assert first_payload["summary"] == "Digest version 1"
    assert second_payload["summary"] == "Digest version 2"
    assert second_payload["cache_hit"] is False
    assert calls == [1, 1]
    assert len(cached_rows) == 1


@pytest.mark.asyncio
async def test_get_quick_glance_digest_reuses_recent_snapshot_for_single_new_story_within_cooldown(db_session, monkeypatch):
    site = NewsSite(
        domain="example.com",
        homepage_url="https://example.com",
        display_name="Example",
        is_public=True,
    )
    db_session.add(site)
    await db_session.flush()

    feed = NewsSiteFeed(
        site_id=site.id,
        feed_url="https://example.com/rss.xml",
        title="Feed One",
        kind="rss",
        discovery_method="seed",
        validation_status="valid",
        is_public=True,
    )
    db_session.add(feed)
    await db_session.flush()

    await _seed_quick_glance_article(
        db_session,
        feed=feed,
        title="ABC earnings rise",
        canonical_url="https://example.com/articles/cooldown-1",
        published_at=datetime(2026, 3, 26, 10, 0, 0),
        tickers=["ABC"],
        story_key="ABC|story|cooldown-1",
    )
    await db_session.commit()

    current_time = datetime(2026, 3, 26, 12, 0, 0)
    monkeypatch.setattr(news_service_module, "utc_now", lambda: current_time)
    call_counts: list[int] = []

    async def _fake_generate_quick_glance_digest(*, evidence_items: list[dict], **kwargs):
        call_counts.append(len(evidence_items))
        return {
            "summary": f"Digest run {len(call_counts)}",
            "highlights": [
                {
                    "title": "Top story",
                    "body": "The leading story changed with the article set.",
                    "article_ids": [evidence_items[0]["article_id"]],
                }
            ],
        }

    monkeypatch.setattr(news_service_module, "generate_quick_glance_digest", _fake_generate_quick_glance_digest)

    first_payload = await news_service.get_quick_glance_digest(user_id=None, window_hours=24)

    await _seed_quick_glance_article(
        db_session,
        feed=feed,
        title="XYZ opens a new warehouse",
        canonical_url="https://example.com/articles/cooldown-2",
        published_at=datetime(2026, 3, 26, 12, 5, 0),
        tickers=["XYZ"],
        topics=["logistics"],
        importance="medium",
        event_type="other",
        story_key="XYZ|story|cooldown-2",
    )
    await db_session.commit()
    current_time = datetime(2026, 3, 26, 12, 15, 0)

    second_payload = await news_service.get_quick_glance_digest(user_id=None, window_hours=24)

    assert first_payload["cache_hit"] is False
    assert second_payload["cache_hit"] is True
    assert second_payload["article_count"] == 1
    assert second_payload["summary"] == first_payload["summary"]
    assert call_counts == [1]


@pytest.mark.asyncio
async def test_get_quick_glance_digest_reuses_recent_snapshot_when_new_articles_stay_in_one_story_group(db_session, monkeypatch):
    site = NewsSite(
        domain="example.com",
        homepage_url="https://example.com",
        display_name="Example",
        is_public=True,
    )
    db_session.add(site)
    await db_session.flush()

    feed = NewsSiteFeed(
        site_id=site.id,
        feed_url="https://example.com/rss.xml",
        title="Feed One",
        kind="rss",
        discovery_method="seed",
        validation_status="valid",
        is_public=True,
    )
    db_session.add(feed)
    await db_session.flush()

    await _seed_quick_glance_article(
        db_session,
        feed=feed,
        title="ABC earnings rise",
        canonical_url="https://example.com/articles/story-group-1",
        published_at=datetime(2026, 3, 26, 10, 0, 0),
        tickers=["ABC"],
        story_key="ABC|story|shared",
    )
    await db_session.commit()

    current_time = datetime(2026, 3, 26, 12, 0, 0)
    monkeypatch.setattr(news_service_module, "utc_now", lambda: current_time)
    call_counts: list[int] = []

    async def _fake_generate_quick_glance_digest(*, article_count: int, **kwargs):
        call_counts.append(article_count)
        return {
            "summary": f"Digest run {len(call_counts)}",
            "highlights": [],
        }

    monkeypatch.setattr(news_service_module, "generate_quick_glance_digest", _fake_generate_quick_glance_digest)

    first_payload = await news_service.get_quick_glance_digest(user_id=None, window_hours=24)

    await _seed_quick_glance_article(
        db_session,
        feed=feed,
        title="ABC expands the same earnings story",
        canonical_url="https://example.com/articles/story-group-2",
        published_at=datetime(2026, 3, 26, 12, 5, 0),
        tickers=["ABC"],
        importance="medium",
        story_key="ABC|story|shared",
    )
    await _seed_quick_glance_article(
        db_session,
        feed=feed,
        title="ABC follow-up guidance keeps the same angle",
        canonical_url="https://example.com/articles/story-group-3",
        published_at=datetime(2026, 3, 26, 12, 10, 0),
        tickers=["ABC"],
        importance="medium",
        story_key="ABC|story|shared",
    )
    await db_session.commit()
    current_time = datetime(2026, 3, 26, 12, 45, 0)

    second_payload = await news_service.get_quick_glance_digest(user_id=None, window_hours=24)

    assert first_payload["cache_hit"] is False
    assert second_payload["cache_hit"] is True
    assert second_payload["article_count"] == 1
    assert second_payload["summary"] == first_payload["summary"]
    assert call_counts == [1]


@pytest.mark.asyncio
async def test_get_quick_glance_digest_regenerates_after_cooldown_for_two_new_story_groups(db_session, monkeypatch):
    site = NewsSite(
        domain="example.com",
        homepage_url="https://example.com",
        display_name="Example",
        is_public=True,
    )
    db_session.add(site)
    await db_session.flush()

    feed = NewsSiteFeed(
        site_id=site.id,
        feed_url="https://example.com/rss.xml",
        title="Feed One",
        kind="rss",
        discovery_method="seed",
        validation_status="valid",
        is_public=True,
    )
    db_session.add(feed)
    await db_session.flush()

    await _seed_quick_glance_article(
        db_session,
        feed=feed,
        title="ABC earnings rise",
        canonical_url="https://example.com/articles/threshold-1",
        published_at=datetime(2026, 3, 26, 10, 0, 0),
        tickers=["ABC"],
        story_key="ABC|story|threshold-1",
    )
    await db_session.commit()

    current_time = datetime(2026, 3, 26, 12, 0, 0)
    monkeypatch.setattr(news_service_module, "utc_now", lambda: current_time)
    call_counts: list[int] = []

    async def _fake_generate_quick_glance_digest(*, article_count: int, **kwargs):
        call_counts.append(article_count)
        return {
            "summary": f"Digest run {len(call_counts)}",
            "highlights": [],
        }

    monkeypatch.setattr(news_service_module, "generate_quick_glance_digest", _fake_generate_quick_glance_digest)

    await news_service.get_quick_glance_digest(user_id=None, window_hours=24)

    await _seed_quick_glance_article(
        db_session,
        feed=feed,
        title="XYZ wins new contract",
        canonical_url="https://example.com/articles/threshold-2",
        published_at=datetime(2026, 3, 26, 12, 5, 0),
        tickers=["XYZ"],
        topics=["contracts"],
        importance="medium",
        event_type="other",
        story_key="XYZ|story|threshold-2",
    )
    await _seed_quick_glance_article(
        db_session,
        feed=feed,
        title="LMN launches another separate story",
        canonical_url="https://example.com/articles/threshold-3",
        published_at=datetime(2026, 3, 26, 12, 10, 0),
        tickers=["LMN"],
        topics=["launch"],
        importance="medium",
        event_type="other",
        story_key="LMN|story|threshold-3",
    )
    await db_session.commit()
    current_time = datetime(2026, 3, 26, 12, 45, 0)

    second_payload = await news_service.get_quick_glance_digest(user_id=None, window_hours=24)

    assert second_payload["cache_hit"] is False
    assert second_payload["article_count"] == 3
    assert second_payload["summary"] == "Digest run 2"
    assert call_counts == [1, 3]


@pytest.mark.asyncio
async def test_get_quick_glance_digest_regenerates_immediately_for_high_importance_story(db_session, monkeypatch):
    site = NewsSite(
        domain="example.com",
        homepage_url="https://example.com",
        display_name="Example",
        is_public=True,
    )
    db_session.add(site)
    await db_session.flush()

    feed = NewsSiteFeed(
        site_id=site.id,
        feed_url="https://example.com/rss.xml",
        title="Feed One",
        kind="rss",
        discovery_method="seed",
        validation_status="valid",
        is_public=True,
    )
    db_session.add(feed)
    await db_session.flush()

    await _seed_quick_glance_article(
        db_session,
        feed=feed,
        title="ABC earnings rise",
        canonical_url="https://example.com/articles/high-1",
        published_at=datetime(2026, 3, 26, 10, 0, 0),
        tickers=["ABC"],
        importance="medium",
        story_key="ABC|story|high-1",
    )
    await db_session.commit()

    current_time = datetime(2026, 3, 26, 12, 0, 0)
    monkeypatch.setattr(news_service_module, "utc_now", lambda: current_time)
    call_counts: list[int] = []

    async def _fake_generate_quick_glance_digest(*, article_count: int, **kwargs):
        call_counts.append(article_count)
        return {
            "summary": f"Digest run {len(call_counts)}",
            "highlights": [],
        }

    monkeypatch.setattr(news_service_module, "generate_quick_glance_digest", _fake_generate_quick_glance_digest)

    await news_service.get_quick_glance_digest(user_id=None, window_hours=24)

    await _seed_quick_glance_article(
        db_session,
        feed=feed,
        title="XYZ issues a high-priority warning",
        canonical_url="https://example.com/articles/high-2",
        published_at=datetime(2026, 3, 26, 12, 5, 0),
        tickers=["XYZ"],
        topics=["warning"],
        importance="high",
        event_type="regulatory",
        story_key="XYZ|story|high-2",
    )
    await db_session.commit()
    current_time = datetime(2026, 3, 26, 12, 10, 0)

    second_payload = await news_service.get_quick_glance_digest(user_id=None, window_hours=24)

    assert second_payload["cache_hit"] is False
    assert second_payload["summary"] == "Digest run 2"
    assert call_counts == [1, 2]


@pytest.mark.asyncio
async def test_get_quick_glance_digest_regenerates_for_material_update_to_high_importance_story(db_session, monkeypatch):
    site = NewsSite(
        domain="example.com",
        homepage_url="https://example.com",
        display_name="Example",
        is_public=True,
    )
    db_session.add(site)
    await db_session.flush()

    feed = NewsSiteFeed(
        site_id=site.id,
        feed_url="https://example.com/rss.xml",
        title="Feed One",
        kind="rss",
        discovery_method="seed",
        validation_status="valid",
        is_public=True,
    )
    db_session.add(feed)
    await db_session.flush()

    article = await _seed_quick_glance_article(
        db_session,
        feed=feed,
        title="ABC earnings rise",
        canonical_url="https://example.com/articles/update-1",
        published_at=datetime(2026, 3, 26, 10, 0, 0),
        tickers=["ABC"],
        importance="high",
        story_key="ABC|story|update-1",
    )
    await db_session.commit()

    current_time = datetime(2026, 3, 26, 12, 0, 0)
    monkeypatch.setattr(news_service_module, "utc_now", lambda: current_time)
    call_counts: list[int] = []

    async def _fake_generate_quick_glance_digest(*, article_count: int, **kwargs):
        call_counts.append(article_count)
        return {
            "summary": f"Digest run {len(call_counts)}",
            "highlights": [],
        }

    monkeypatch.setattr(news_service_module, "generate_quick_glance_digest", _fake_generate_quick_glance_digest)

    await news_service.get_quick_glance_digest(user_id=None, window_hours=24)

    article.content_text = "Updated body with materially different details."
    article.updated_at = datetime(2026, 3, 26, 12, 5, 0)
    await db_session.commit()
    current_time = datetime(2026, 3, 26, 12, 10, 0)

    second_payload = await news_service.get_quick_glance_digest(user_id=None, window_hours=24)

    assert second_payload["cache_hit"] is False
    assert second_payload["summary"] == "Digest run 2"
    assert call_counts == [1, 1]


@pytest.mark.asyncio
async def test_load_article_sources_map_groups_sources_by_article(db_session):
    site = NewsSite(
        domain="example.com",
        homepage_url="https://example.com",
        display_name="Example",
        is_public=True,
    )
    db_session.add(site)
    await db_session.flush()

    feed = NewsSiteFeed(
        site_id=site.id,
        feed_url="https://example.com/rss.xml",
        title="Feed One",
        kind="rss",
        discovery_method="seed",
        validation_status="valid",
        is_public=True,
    )
    crawl_source = NewsCrawlSource(
        site_id=site.id,
        listing_url="https://example.com/news",
        article_link_selector="a.story-link",
        content_selector="article",
        validation_status="valid",
        is_public=True,
    )
    db_session.add_all([feed, crawl_source])
    await db_session.flush()

    first_article = NewsArticle(
        canonical_url="https://example.com/articles/first",
        title="First article",
        excerpt="First excerpt",
        content_text="First content",
        published_at=datetime(2026, 3, 26, 12, 0, 0),
        language="en",
        content_hash="first-hash",
    )
    second_article = NewsArticle(
        canonical_url="https://example.com/articles/second",
        title="Second article",
        excerpt="Second excerpt",
        content_text="Second content",
        published_at=datetime(2026, 3, 26, 11, 0, 0),
        language="en",
        content_hash="second-hash",
    )
    db_session.add_all([first_article, second_article])
    await db_session.flush()

    db_session.add_all(
        [
            NewsArticleSource(article_id=first_article.id, site_feed_id=feed.id, article_url=first_article.canonical_url),
            NewsArticleSource(article_id=first_article.id, crawl_source_id=crawl_source.id, article_url=first_article.canonical_url),
            NewsArticleSource(article_id=second_article.id, site_feed_id=feed.id, article_url=second_article.canonical_url),
        ]
    )
    await db_session.commit()

    source_map = await news_service._load_article_sources_map(
        db_session,
        [first_article.id, second_article.id],
        user_id=None,
    )

    assert [item["source_type"] for item in source_map[first_article.id]] == ["rss", "crawl"]
    assert [item["domain"] for item in source_map[first_article.id]] == ["example.com", "example.com"]
    assert [item["source_type"] for item in source_map[second_article.id]] == ["rss"]


@pytest.mark.asyncio
async def test_get_article_detail_repairs_stored_noisy_content(db_session, monkeypatch):
    html = """
    <html lang="vi">
      <head>
        <meta property="og:title" content="VCCI đề xuất giảm thuế" />
        <meta property="og:description" content="Sapo summary" />
      </head>
      <body>
        <main>
          <h1 class="name-detail">VCCI đề xuất giảm thuế</h1>
          <div class="main-detail-page">
            <div class="ct-edtior-web news-type1">
              <div class="news-sapo"><p>Sapo summary</p></div>
              <div>
                <p>Đoạn 1 của bài viết.</p>
                <p>Đoạn 2 của bài viết.</p>
              </div>
              <div class="list-detail-revert_item">
                <p>Bài liên quan không nên xuất hiện.</p>
              </div>
              <div class="box-keyword">
                <p>Từ khóa: thuế bảo vệ môi trường</p>
              </div>
            </div>
          </div>
        </main>
      </body>
    </html>
    """

    async def _fake_fetch_text(client, url: str) -> str:
        del client, url
        return html

    async def _fake_classify_article(title: str, excerpt: str | None, content_text: str | None):
        assert title == "VCCI đề xuất giảm thuế"
        assert excerpt == "Sapo summary"
        assert content_text == "Sapo summary\nĐoạn 1 của bài viết.\nĐoạn 2 của bài viết."
        return {
            "topics": ["tax"],
            "tickers": ["PLX"],
            "sectors": ["energy"],
            "importance": "high",
            "sentiment": "neutral",
            "raw_payload": {"topics": ["tax"]},
        }

    monkeypatch.setattr("app.services.news.service.fetch_text", _fake_fetch_text)
    monkeypatch.setattr("app.services.news.service.classify_article", _fake_classify_article)

    site = NewsSite(
        domain="vneconomy.vn",
        homepage_url="https://vneconomy.vn",
        display_name="VnEconomy",
        is_public=True,
    )
    db_session.add(site)
    await db_session.flush()

    feed = NewsSiteFeed(
        site_id=site.id,
        feed_url="https://vneconomy.vn/tai-chinh/rss",
        title="Tài chính",
        kind="rss",
        discovery_method="seed",
        validation_status="valid",
        is_public=True,
    )
    db_session.add(feed)
    await db_session.flush()

    article = NewsArticle(
        canonical_url="https://vneconomy.vn/example.htm",
        title="VCCI đề xuất giảm thuế",
        excerpt="Old excerpt",
        content_text="Đảng Cộng sản Việt Nam - Đại hội XIV\nVới phương châm Đoàn kết - Dân chủ - Kỷ cương - Đột phá - Phát triển...",
        llm_summary="Old generated summary",
        published_at=datetime(2026, 3, 26, 18, 43, 0),
        language="vi",
        content_hash="noisy-hash",
    )
    db_session.add(article)
    await db_session.flush()
    db_session.add(NewsArticleSource(article_id=article.id, site_feed_id=feed.id, article_url=article.canonical_url))
    db_session.add(
        NewsArticleSemantic(
            article_id=article.id,
            topics=["politics"],
            tickers=[],
            sectors=["government"],
            importance="low",
            sentiment="neutral",
            raw_payload={},
        )
    )
    await db_session.commit()

    payload = await news_service.get_article_detail(article.id, user_id=None)
    await db_session.refresh(article)
    semantic = (await db_session.execute(select(NewsArticleSemantic).where(NewsArticleSemantic.article_id == article.id))).scalar_one()

    assert payload is not None
    assert payload["content_text"] == "Sapo summary\nĐoạn 1 của bài viết.\nĐoạn 2 của bài viết."
    assert "Bài liên quan" not in payload["content_text"]
    assert article.content_text == payload["content_text"]
    assert article.excerpt == "Sapo summary"
    assert article.llm_summary is None
    assert payload["topics"] == ["tax"]
    assert payload["tickers"] == ["PLX"]
    assert semantic.topics == ["tax"]
    assert semantic.sectors == ["energy"]


@pytest.mark.asyncio
async def test_get_article_detail_repairs_excerpt_only_content_for_sggp_layout(db_session, monkeypatch):
    html = """
    <html lang="vi">
      <head>
        <meta property="og:title" content="Email của Giám đốc FBI bị hacker xâm nhập" />
        <meta property="og:description" content="(ĐTTCO) - Cục Điều tra liên bang Mỹ (FBI) ngày 29-3 xác nhận tin tặc đã xâm nhập hộp thư điện tử cá nhân của Giám đốc Kash Patel." />
      </head>
      <body>
        <div class="article">
          <div class="article__sapo cms-desc">
            <p>(ĐTTCO) - Cục Điều tra liên bang Mỹ (FBI) ngày 29-3 xác nhận tin tặc đã xâm nhập hộp thư điện tử cá nhân của Giám đốc Kash Patel.</p>
          </div>
          <div class="article__body zce-content-body cms-body" itemprop="articleBody">
            <p>Theo FBI, dữ liệu bị đánh cắp không phải thông tin mới và không bao gồm bất kỳ dữ liệu nào của chính phủ.</p>
            <p>Nhóm này sau đó đăng tải một số ảnh và tài liệu cá nhân, cùng các email từ thời điểm trước khi ông Patel đảm nhiệm chức vụ Giám đốc FBI.</p>
            <div class="related-news">
              <article class="story">
                <h2>Hacker chỉ mất vài giây để xâm nhập tai nghe của bạn</h2>
              </article>
            </div>
          </div>
        </div>
      </body>
    </html>
    """

    async def _fake_fetch_text(client, url: str) -> str:
        del client, url
        return html

    async def _fake_classify_article(title: str, excerpt: str | None, content_text: str | None):
        assert title == "Email của Giám đốc FBI bị hacker xâm nhập"
        assert excerpt and excerpt.startswith("(ĐTTCO) - Cục Điều tra liên bang Mỹ")
        assert content_text == (
            "Theo FBI, dữ liệu bị đánh cắp không phải thông tin mới và không bao gồm bất kỳ dữ liệu nào của chính phủ.\n"
            "Nhóm này sau đó đăng tải một số ảnh và tài liệu cá nhân, cùng các email từ thời điểm trước khi ông Patel đảm nhiệm chức vụ Giám đốc FBI."
        )
        return {
            "topics": ["cybersecurity"],
            "tickers": [],
            "sectors": ["technology"],
            "importance": "medium",
            "sentiment": "neutral",
            "raw_payload": {"topics": ["cybersecurity"]},
        }

    monkeypatch.setattr("app.services.news.service.fetch_text", _fake_fetch_text)
    monkeypatch.setattr("app.services.news.service.classify_article", _fake_classify_article)

    site = NewsSite(
        domain="dttc.sggp.org.vn",
        homepage_url="https://dttc.sggp.org.vn",
        display_name="DTTC",
        is_public=True,
    )
    db_session.add(site)
    await db_session.flush()

    feed = NewsSiteFeed(
        site_id=site.id,
        feed_url="https://saigondautu.com.vn/rss/home.rss",
        title="DTTC RSS",
        kind="rss",
        discovery_method="seed",
        validation_status="valid",
        is_public=True,
    )
    db_session.add(feed)
    await db_session.flush()

    excerpt = "(ĐTTCO) - Cục Điều tra liên bang Mỹ (FBI) ngày 29-3 xác nhận tin tặc đã xâm nhập hộp thư điện tử cá nhân của Giám đốc Kash Patel."
    article = NewsArticle(
        canonical_url="https://dttc.sggp.org.vn/email-cua-giam-doc-fbi-bi-hacker-xam-nhap-post132660.html",
        title="Email của Giám đốc FBI bị hacker xâm nhập",
        excerpt=excerpt,
        content_text=excerpt,
        llm_summary="Old generated summary",
        published_at=datetime(2026, 3, 30, 9, 42, 49),
        language="vi",
        content_hash="old-sggp-hash",
    )
    db_session.add(article)
    await db_session.flush()
    db_session.add(NewsArticleSource(article_id=article.id, site_feed_id=feed.id, article_url=article.canonical_url))
    db_session.add(
        NewsArticleSemantic(
            article_id=article.id,
            topics=["old-topic"],
            tickers=[],
            sectors=["old-sector"],
            importance="low",
            sentiment="neutral",
            raw_payload={},
        )
    )
    await db_session.commit()

    payload = await news_service.get_article_detail(article.id, user_id=None)
    await db_session.refresh(article)
    semantic = (await db_session.execute(select(NewsArticleSemantic).where(NewsArticleSemantic.article_id == article.id))).scalar_one()

    assert payload is not None
    assert payload["content_text"] == article.content_text
    assert article.content_text == (
        "Theo FBI, dữ liệu bị đánh cắp không phải thông tin mới và không bao gồm bất kỳ dữ liệu nào của chính phủ.\n"
        "Nhóm này sau đó đăng tải một số ảnh và tài liệu cá nhân, cùng các email từ thời điểm trước khi ông Patel đảm nhiệm chức vụ Giám đốc FBI."
    )
    assert article.llm_summary is None
    assert payload["topics"] == ["cybersecurity"]
    assert semantic.topics == ["cybersecurity"]
    assert semantic.sectors == ["technology"]


@pytest.mark.asyncio
async def test_refresh_article_content_returns_updated_detail(db_session, monkeypatch):
    html = """
    <html lang="vi">
      <head>
        <meta property="og:title" content="VCCI đề xuất giảm thuế" />
        <meta property="og:description" content="Sapo summary" />
      </head>
      <body>
        <main>
          <h1 class="name-detail">VCCI đề xuất giảm thuế</h1>
          <div class="main-detail-page">
            <div class="ct-edtior-web news-type1">
              <div class="news-sapo"><p>Sapo summary</p></div>
              <div>
                <p>Đoạn 1 của bài viết.</p>
                <p>Đoạn 2 của bài viết.</p>
              </div>
            </div>
          </div>
        </main>
      </body>
    </html>
    """

    async def _fake_fetch_text(client, url: str) -> str:
        del client, url
        return html

    async def _fake_classify_article(title: str, excerpt: str | None, content_text: str | None):
        assert excerpt == "Sapo summary"
        assert content_text == "Sapo summary\nĐoạn 1 của bài viết.\nĐoạn 2 của bài viết."
        return {
            "topics": ["tax"],
            "tickers": ["PLX"],
            "sectors": ["energy"],
            "importance": "high",
            "sentiment": "neutral",
            "raw_payload": {"topics": ["tax"]},
        }

    monkeypatch.setattr("app.services.news.service.fetch_text", _fake_fetch_text)
    monkeypatch.setattr("app.services.news.service.classify_article", _fake_classify_article)

    site = NewsSite(
        domain="vneconomy.vn",
        homepage_url="https://vneconomy.vn",
        display_name="VnEconomy",
        is_public=True,
    )
    db_session.add(site)
    await db_session.flush()

    feed = NewsSiteFeed(
        site_id=site.id,
        feed_url="https://vneconomy.vn/tai-chinh.rss",
        title="Tài chính",
        kind="rss",
        discovery_method="seed",
        validation_status="valid",
        is_public=True,
    )
    db_session.add(feed)
    await db_session.flush()

    article = NewsArticle(
        canonical_url="https://vneconomy.vn/example-refresh.htm",
        title="VCCI đề xuất giảm thuế",
        excerpt="Old excerpt",
        content_text="Old content",
        llm_summary="Old generated summary",
        published_at=datetime(2026, 3, 26, 18, 43, 0),
        language="vi",
        content_hash="old-hash",
    )
    db_session.add(article)
    await db_session.flush()
    db_session.add(NewsArticleSource(article_id=article.id, site_feed_id=feed.id, article_url=article.canonical_url))
    db_session.add(
        NewsArticleSemantic(
            article_id=article.id,
            topics=["old-topic"],
            tickers=[],
            sectors=["old-sector"],
            importance="low",
            sentiment="neutral",
            raw_payload={},
        )
    )
    await db_session.commit()

    payload = await news_service.refresh_article_content(article.id, user_id=None)
    await db_session.refresh(article)
    semantic = (await db_session.execute(select(NewsArticleSemantic).where(NewsArticleSemantic.article_id == article.id))).scalar_one()

    assert payload is not None
    assert payload["content_text"] == "Sapo summary\nĐoạn 1 của bài viết.\nĐoạn 2 của bài viết."
    assert article.content_text == payload["content_text"]
    assert article.excerpt == "Sapo summary"
    assert article.llm_summary is None
    assert payload["topics"] == ["tax"]
    assert payload["tickers"] == ["PLX"]
    assert semantic.topics == ["tax"]
    assert semantic.sectors == ["energy"]


@pytest.mark.asyncio
async def test_get_article_detail_prefers_display_topics_from_raw_payload(db_session):
    site = NewsSite(
        domain="cafef.vn",
        homepage_url="https://cafef.vn",
        display_name="CafeF",
        is_public=True,
    )
    db_session.add(site)
    await db_session.flush()

    feed = NewsSiteFeed(
        site_id=site.id,
        feed_url="https://cafef.vn/rss",
        title="CafeF RSS",
        kind="rss",
        discovery_method="seed",
        validation_status="valid",
        is_public=True,
    )
    db_session.add(feed)
    await db_session.flush()

    article = NewsArticle(
        canonical_url="https://cafef.vn/example-topic-display.chn",
        title="BID duoc mua manh",
        excerpt="Tom tat",
        content_text="Noi dung bai viet",
        published_at=datetime(2026, 3, 27, 10, 0, 0),
        language="vi",
        content_hash="topic-display-hash",
    )
    db_session.add(article)
    await db_session.flush()
    db_session.add(NewsArticleSource(article_id=article.id, site_feed_id=feed.id, article_url=article.canonical_url))
    db_session.add(
        NewsArticleSemantic(
            article_id=article.id,
            topics=["phat_hanh_co_phieu_rieng_le", "tang_von_dieu_le_ngan_hang"],
            tickers=["BID"],
            sectors=["banking"],
            importance="high",
            sentiment="positive",
            raw_payload={
                "topics": ["phát hành cổ phiếu riêng lẻ", "tăng vốn điều lệ ngân hàng"],
                "display_topics": ["phát hành cổ phiếu riêng lẻ", "tăng vốn điều lệ ngân hàng"],
            },
        )
    )
    await db_session.commit()

    payload = await news_service.get_article_detail(article.id, user_id=None)

    assert payload["topics"] == ["phát hành cổ phiếu riêng lẻ", "tăng vốn điều lệ ngân hàng"]


@pytest.mark.asyncio
async def test_get_article_detail_uses_site_domain_as_source_label(db_session):
    site = NewsSite(
        domain="cafef.vn",
        homepage_url="https://cafef.vn",
        display_name="CafeF",
        is_public=True,
    )
    db_session.add(site)
    await db_session.flush()

    feed = NewsSiteFeed(
        site_id=site.id,
        feed_url="https://cafef.vn/rss.chn",
        title="A sample article title accidentally stored as feed title",
        kind="rss",
        discovery_method="manual",
        validation_status="valid",
        is_public=True,
    )
    db_session.add(feed)
    await db_session.flush()

    article = NewsArticle(
        canonical_url="https://cafef.vn/article-source-label.chn",
        title="Article title",
        excerpt="Excerpt",
        content_text="Body",
        published_at=datetime(2026, 3, 26, 12, 0, 0),
        language="vi",
        content_hash="source-label-hash",
    )
    db_session.add(article)
    await db_session.flush()
    db_session.add(NewsArticleSource(article_id=article.id, site_feed_id=feed.id, article_url=article.canonical_url))
    await db_session.commit()

    payload = await news_service.get_article_detail(article.id, user_id=None)

    assert payload is not None
    assert payload["source_labels"] == ["cafef.vn"]


def test_news_utc_naive_to_local_iso_uses_ho_chi_minh_timezone():
    assert utc_naive_to_local_iso(datetime(2026, 3, 26, 12, 14, 22)) == "2026-03-26T19:14:22+07:00"


def test_should_normalize_legacy_feed_title_flags_article_like_titles():
    site = NewsSite(domain="cafef.vn", homepage_url="https://cafef.vn", display_name="CafeF", is_public=True)

    assert news_service._should_normalize_legacy_feed_title(
        "Công ty chứng khoán sắp kinh doanh vàng phái sinh, lấn sân tài sản số",
        site,
    ) is True
    assert news_service._should_normalize_legacy_feed_title("CafeF RSS", site) is False
    assert news_service._should_normalize_legacy_feed_title("CafeF - Thị trường chứng khoán", site) is False


@pytest.mark.asyncio
async def test_normalize_legacy_rss_titles_updates_old_article_like_feed_titles(db_session):
    site = NewsSite(
        domain="cafef.vn",
        homepage_url="https://cafef.vn",
        display_name="CafeF",
        is_public=False,
    )
    db_session.add(site)
    await db_session.flush()

    bad_feed = NewsSiteFeed(
        site_id=site.id,
        feed_url="https://cafef.vn/rss.chn",
        title="Công ty chứng khoán sắp kinh doanh vàng phái sinh, lấn sân tài sản số",
        kind="rss",
        discovery_method="manual",
        validation_status="valid",
        is_public=False,
    )
    good_feed = NewsSiteFeed(
        site_id=site.id,
        feed_url="https://cafef.vn/thi-truong.rss",
        title="CafeF RSS",
        kind="rss",
        discovery_method="manual",
        validation_status="valid",
        is_public=False,
    )
    db_session.add_all([bad_feed, good_feed])
    await db_session.commit()

    changed_count = await news_service.normalize_legacy_rss_titles()
    await db_session.refresh(bad_feed)
    await db_session.refresh(good_feed)

    assert changed_count == 1
    assert bad_feed.title == "cafef.vn"
    assert good_feed.title == "CafeF RSS"


def test_news_admin_config_reads_and_writes_default_poll_interval(tmp_path, monkeypatch):
    settings_path = Path(tmp_path) / "settings.yaml"
    settings_path.write_text("brokers: []\n", encoding="utf-8")
    monkeypatch.setattr(news_service_module.settings, "settings_yaml_path", str(settings_path))
    monkeypatch.setattr(news_service_module.settings, "news_default_poll_interval_minutes", 30)

    assert news_service.get_default_poll_interval_minutes() == 30

    updated = news_service.update_admin_config(default_poll_interval_minutes=45)

    assert updated["default_poll_interval_minutes"] == 45
    assert news_service.get_default_poll_interval_minutes() == 45
    assert "default_poll_interval_minutes: 45" in settings_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_apply_default_poll_interval_to_existing_sources_updates_mismatched_sources(db_session, tmp_path, monkeypatch):
    settings_path = Path(tmp_path) / "settings.yaml"
    settings_path.write_text("news:\n  default_poll_interval_minutes: 30\n", encoding="utf-8")
    monkeypatch.setattr(news_service_module.settings, "settings_yaml_path", str(settings_path))

    site = NewsSite(
        domain="example.com",
        homepage_url="https://example.com",
        display_name="Example",
        is_public=True,
    )
    db_session.add(site)
    await db_session.flush()

    feed = NewsSiteFeed(
        site_id=site.id,
        feed_url="https://example.com/rss.xml",
        title="Example RSS",
        kind="rss",
        discovery_method="seed",
        validation_status="valid",
        is_public=True,
        poll_interval_minutes=60,
        next_poll_at=datetime(2026, 3, 26, 8, 0, 0),
    )
    source = NewsCrawlSource(
        site_id=site.id,
        listing_url="https://example.com/news",
        article_link_selector="article a",
        content_selector="article",
        excerpt_selector=None,
        pagination_config=None,
        validation_status="valid",
        is_public=False,
        poll_interval_minutes=45,
        next_poll_at=datetime(2026, 3, 26, 9, 0, 0),
    )
    unchanged_feed = NewsSiteFeed(
        site_id=site.id,
        feed_url="https://example.com/markets.xml",
        title="Markets RSS",
        kind="rss",
        discovery_method="seed",
        validation_status="valid",
        is_public=True,
        poll_interval_minutes=30,
        next_poll_at=datetime(2026, 3, 26, 10, 0, 0),
    )
    db_session.add_all([feed, source, unchanged_feed])
    await db_session.commit()

    original_utc_now = news_service_module.utc_now
    news_service_module.utc_now = lambda: datetime(2026, 3, 26, 12, 14, 22)
    try:
        changed_count = await news_service.apply_default_poll_interval_to_existing_sources()
    finally:
        news_service_module.utc_now = original_utc_now

    await db_session.refresh(feed)
    await db_session.refresh(source)
    await db_session.refresh(unchanged_feed)

    assert changed_count == 2
    assert feed.poll_interval_minutes == 30
    assert feed.next_poll_at == datetime(2026, 3, 26, 12, 14, 22)
    assert source.poll_interval_minutes == 30
    assert source.next_poll_at == datetime(2026, 3, 26, 12, 14, 22)
    assert unchanged_feed.poll_interval_minutes == 30
    assert unchanged_feed.next_poll_at == datetime(2026, 3, 26, 10, 0, 0)


@pytest.mark.asyncio
async def test_ensure_public_sources_removes_seeded_public_feeds_missing_from_config(db_session, tmp_path, monkeypatch):
    news_sources_path = Path(tmp_path) / "news_sources.yaml"
    news_sources_path.write_text(
        """
public_sources:
  - homepage_url: "https://cafef.vn"
    display_name: "CafeF"
    feeds:
      - feed_url: "https://cafef.vn/thi-truong-chung-khoan.rss"
        title: "CafeF - Thi truong chung khoan"
        kind: "rss"
        poll_interval_minutes: 30
        """.strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(news_service_module.settings, "news_sources_yaml_path", str(news_sources_path))

    cafef_site = NewsSite(
        domain="cafef.vn",
        homepage_url="https://cafef.vn",
        display_name="CafeF",
        is_public=True,
    )
    stale_site = NewsSite(
        domain="tuoitre.vn",
        homepage_url="https://tuoitre.vn",
        display_name="Tuoi Tre",
        is_public=True,
    )
    private_site = NewsSite(
        domain="private.example.com",
        homepage_url="https://private.example.com",
        display_name="Private",
        is_public=False,
    )
    db_session.add_all([cafef_site, stale_site, private_site])
    await db_session.flush()

    cafef_feed = NewsSiteFeed(
        site_id=cafef_site.id,
        feed_url="https://cafef.vn/thi-truong-chung-khoan.rss",
        title="Old CafeF Title",
        kind="rss",
        discovery_method="seed",
        validation_status="valid",
        is_public=True,
        poll_interval_minutes=15,
    )
    stale_feed = NewsSiteFeed(
        site_id=stale_site.id,
        feed_url="https://tuoitre.vn/rss/thoi-su.rss",
        title="Tuoi Tre - Thoi su",
        kind="rss",
        discovery_method="seed",
        validation_status="valid",
        is_public=True,
        poll_interval_minutes=30,
    )
    private_feed = NewsSiteFeed(
        site_id=private_site.id,
        feed_url="https://private.example.com/rss.xml",
        title="Private Feed",
        kind="rss",
        discovery_method="manual",
        validation_status="valid",
        is_public=False,
        poll_interval_minutes=30,
    )
    db_session.add_all([cafef_feed, stale_feed, private_feed])
    await db_session.commit()

    await news_service.ensure_public_sources()
    await db_session.rollback()

    feed_urls = {
        row.feed_url
        for row in (await db_session.execute(select(NewsSiteFeed).order_by(NewsSiteFeed.feed_url.asc()))).scalars().all()
    }
    site_homepages = {
        row.homepage_url
        for row in (await db_session.execute(select(NewsSite).order_by(NewsSite.homepage_url.asc()))).scalars().all()
    }

    assert "https://tuoitre.vn/rss/thoi-su.rss" not in feed_urls
    assert "https://cafef.vn/thi-truong-chung-khoan.rss" in feed_urls
    assert "https://private.example.com/rss.xml" in feed_urls
    assert "https://tuoitre.vn" not in site_homepages


@pytest.mark.asyncio
async def test_reconcile_stale_runs_marks_abandoned_running_rows_failed(db_session):
    stale_run = NewsIngestionRun(
        source_type="rss",
        status="running",
        started_at=datetime(2026, 3, 26, 12, 0, 0),
    )
    fresh_run = NewsIngestionRun(
        source_type="rss",
        status="running",
        started_at=datetime(2026, 3, 26, 12, NEWS_STALE_RUN_MINUTES, 1),
    )
    db_session.add(stale_run)
    db_session.add(fresh_run)
    await db_session.commit()

    original_utc_now = news_service_module.utc_now
    news_service_module.utc_now = lambda: datetime(2026, 3, 26, 12, NEWS_STALE_RUN_MINUTES, 1)
    try:
        cleaned = await news_service.reconcile_stale_runs()
    finally:
        news_service_module.utc_now = original_utc_now

    await db_session.refresh(stale_run)
    await db_session.refresh(fresh_run)

    assert cleaned == 1
    assert stale_run.status == "failed"
    assert stale_run.finished_at is not None
    assert fresh_run.status == "running"
    assert fresh_run.finished_at is None
