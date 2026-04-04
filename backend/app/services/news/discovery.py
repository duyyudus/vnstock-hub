from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
import re
from typing import Any
from urllib.parse import parse_qsl, urljoin, urlparse, urlunparse
from xml.etree import ElementTree as ET

from bs4 import BeautifulSoup
import httpx


FEED_MIME_TYPES = {
    "application/rss+xml",
    "application/atom+xml",
    "application/xml",
    "text/xml",
}

COMMON_FEED_PATHS = (
    "/rss",
    "/rss.xml",
    "/rss.html",
    "/rss.htm",
    "/feed",
    "/feed.xml",
    "/atom.xml",
)
COMMON_SITEMAP_PATHS = (
    "/sitemap.xml",
    "/sitemap_index.xml",
)
SITEMAP_FEED_HINTS = ("news", "article", "post", "tin")
SITEMAP_LISTING_HINTS = ("categories", "category", "topics", "topic", "sections", "section")
SITEMAP_LISTING_EXCLUDE_SEGMENTS = ("epaper", "video", "longform", "luu-tru", "tag", "tags")
MAX_SITEMAP_LISTING_CANDIDATES = 24

ARTICLE_LINK_HINTS = ("news", "tin", "article", "bai-viet", "story")
IGNORED_QUERY_PREFIXES = ("utm_",)
ARTICLE_BODY_SELECTORS = (
    ".ct-edtior-web",
    ".ct-editor-web",
    ".article__body",
    ".zce-content-body",
    ".cms-body",
    ".article__content",
    ".entry-content",
    ".post-content",
    "[itemprop='articleBody']",
    ".article-content",
    ".content-detail",
    ".news-detail",
    ".detail-content",
    ".detail__content",
    ".detail-body",
    ".detail__body",
    ".main-detail-page",
    "article",
)
ARTICLE_NOISE_SELECTORS = (
    "script",
    "style",
    "noscript",
    "iframe",
    "form",
    "nav",
    "aside",
    "footer",
    ".list-detail-revert_item",
    ".box-keyword",
    ".box-tags",
    ".news-related",
    ".related-news",
    ".related-post",
    ".related-posts",
    ".read-more",
    ".recommend",
    ".recommended",
    ".most-read",
    ".social-share",
    ".share-detail",
    ".banner",
    ".ads",
    ".advertisement",
    "[class*='related']",
    "[class*='recommend']",
    "[class*='most-read']",
    "[class*='keyword']",
    "[class*='tag']",
    "[class*='share']",
    "[class*='social']",
    "[class*='banner']",
    "[class*='advert']",
    "[class*='promo']",
)
ARTICLE_STOP_HEADINGS = (
    "đọc thêm",
    "từ khóa",
    "bài liên quan",
    "bài viết liên quan",
    "bài viết mới nhất",
    "tin liên quan",
    "xem thêm",
)


@dataclass(slots=True)
class FeedEntry:
    title: str
    link: str
    summary: str | None
    published_at: datetime | None


def normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    query_items = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith(IGNORED_QUERY_PREFIXES) and key.lower() not in {"fbclid", "gclid"}
    ]
    cleaned = parsed._replace(fragment="", query="&".join(f"{key}={value}" for key, value in query_items))
    return urlunparse(cleaned)


def extract_domain(url: str) -> str:
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


async def fetch_text(client: httpx.AsyncClient, url: str) -> str:
    response = await client.get(url, follow_redirects=True)
    response.raise_for_status()
    return response.text


def _looks_like_feed_url(url: str) -> bool:
    lowered = url.lower()
    return any(token in lowered for token in ("/rss", "/feed", "atom", ".xml", ".rss"))


def _looks_like_feed_hint(value: str | None) -> bool:
    if not value:
        return False
    lowered = value.lower()
    return any(token in lowered for token in ("rss", "feed", "atom"))


def _looks_like_feed_hub_url(url: str) -> bool:
    parsed = urlparse(url)
    lowered = parsed.path.lower().rstrip("/")
    return lowered.endswith(("/rss", "/feed", "/rss.html", "/rss.htm", "/feed.html", "/feed.htm")) or any(
        token in url.lower() for token in ("/rss?", "/feed?")
    )


def _looks_like_article_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    if not path or path == "/" or path.endswith("/"):
        return False
    leaf = path.rsplit("/", 1)[-1]
    if re.search(r"-post\d+\.(?:html?|php|aspx?)$", leaf):
        return True
    if leaf.endswith((".html", ".htm", ".shtml", ".chn")) and "-" in leaf:
        return True
    return False


def _title_from_url(url: str) -> str | None:
    path = urlparse(url).path.strip("/")
    if not path:
        return None
    leaf = path.rsplit("/", 1)[-1]
    normalized = re.sub(r"\.(?:xml|rss|atom|html?|htm|shtml|chn)$", "", leaf, flags=re.IGNORECASE)
    normalized = re.sub(r"-post\d+$", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"[-_]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.title() if normalized else None


def parse_homepage_feed_candidates(homepage_url: str, html_text: str) -> list[dict[str, str | None]]:
    soup = BeautifulSoup(html_text, "html.parser")
    candidates: dict[str, dict[str, str | None]] = {}
    homepage_domain = extract_domain(homepage_url)

    for link in soup.find_all("link"):
        rel_values = {str(value).lower() for value in (link.get("rel") or [])}
        link_type = str(link.get("type") or "").lower()
        href = str(link.get("href") or "").strip()
        if not href:
            continue
        absolute = normalize_url(urljoin(homepage_url, href))
        domain = extract_domain(absolute)
        if domain != homepage_domain:
            continue
        if "alternate" in rel_values and link_type in FEED_MIME_TYPES:
            candidates[absolute] = {
                "feed_url": absolute,
                "title": str(link.get("title") or "").strip() or None,
                "kind": "atom" if "atom" in link_type else "rss",
                "discovery_method": "homepage",
            }

    for anchor in soup.find_all("a", href=True):
        href = str(anchor.get("href") or "").strip()
        if not href:
            continue
        absolute = normalize_url(urljoin(homepage_url, href))
        domain = extract_domain(absolute)
        title = " ".join(anchor.stripped_strings).strip() or None
        title_hint = str(anchor.get("title") or "").strip() or None
        aria_label = str(anchor.get("aria-label") or "").strip() or None
        if domain != homepage_domain:
            continue
        if not (
            _looks_like_feed_url(absolute)
            or _looks_like_feed_hint(title)
            or _looks_like_feed_hint(title_hint)
            or _looks_like_feed_hint(aria_label)
        ):
            continue
        candidates.setdefault(
            absolute,
            {
                "feed_url": absolute,
                "title": title or title_hint or aria_label,
                "kind": "atom" if "atom" in absolute.lower() else "rss",
                "discovery_method": "anchor",
            },
        )

    for suffix in COMMON_FEED_PATHS:
        candidate = normalize_url(urljoin(homepage_url, suffix))
        if extract_domain(candidate) == homepage_domain:
            candidates.setdefault(
                candidate,
                {
                    "feed_url": candidate,
                    "title": None,
                    "kind": "rss",
                    "discovery_method": "common_path",
                },
            )

    return list(candidates.values())


def _first_text(element: ET.Element | None, *names: str) -> str | None:
    if element is None:
        return None
    for name in names:
        found = element.find(name)
        if found is not None and found.text:
            return found.text.strip()
    return None


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return parsedate_to_datetime(text).replace(tzinfo=None)
    except Exception:
        pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def parse_feed_entries(feed_text: str) -> list[FeedEntry]:
    try:
        root = ET.fromstring(feed_text)
    except ET.ParseError:
        return _parse_feed_entries_with_bs4(feed_text)
    entries: list[FeedEntry] = []

    if root.tag.endswith("rss"):
        channel = root.find("channel")
        if channel is None:
            return entries
        for item in channel.findall("item"):
            title = _first_text(item, "title") or ""
            link = _first_text(item, "link") or ""
            if not title or not link:
                continue
            entries.append(
                FeedEntry(
                    title=title,
                    link=normalize_url(link),
                    summary=_first_text(item, "description"),
                    published_at=_parse_datetime(_first_text(item, "pubDate")),
                )
            )
        return entries

    namespace = ""
    if root.tag.startswith("{"):
        namespace = root.tag.split("}", 1)[0] + "}"
    for entry in root.findall(f"{namespace}entry"):
        title = _first_text(entry, f"{namespace}title") or ""
        link_value = ""
        for link in entry.findall(f"{namespace}link"):
            href = link.attrib.get("href")
            if href:
                link_value = href
                break
        if not title or not link_value:
            continue
        entries.append(
            FeedEntry(
                title=title,
                link=normalize_url(link_value),
                summary=_first_text(entry, f"{namespace}summary", f"{namespace}content"),
                published_at=_parse_datetime(
                    _first_text(entry, f"{namespace}published", f"{namespace}updated")
                ),
            )
        )
    return entries


def _parse_feed_entries_with_bs4(feed_text: str) -> list[FeedEntry]:
    soup = BeautifulSoup(feed_text, "html.parser")
    entries: list[FeedEntry] = []

    rss_items = soup.find_all("item")
    if rss_items:
        for item in rss_items:
            title_node = item.find("title")
            link_node = item.find("link")
            title = title_node.get_text(" ", strip=True) if title_node else ""
            link = link_node.get_text(" ", strip=True) if link_node else ""
            if not title or not link:
                continue
            summary_node = item.find("description")
            pub_node = item.find("pubDate")
            entries.append(
                FeedEntry(
                    title=title,
                    link=normalize_url(link),
                    summary=summary_node.get_text(" ", strip=True) if summary_node else None,
                    published_at=_parse_datetime(pub_node.get_text(" ", strip=True) if pub_node else None),
                )
            )
        return entries

    atom_entries = soup.find_all("entry")
    for entry in atom_entries:
        title_node = entry.find("title")
        link_node = entry.find("link", href=True)
        if not title_node or not link_node or not link_node.get("href"):
            continue
        summary_node = entry.find("summary") or entry.find("content")
        published_node = entry.find("published") or entry.find("updated")
        entries.append(
            FeedEntry(
                title=title_node.get_text(" ", strip=True),
                link=normalize_url(str(link_node.get("href")).strip()),
                summary=summary_node.get_text(" ", strip=True) if summary_node else None,
                published_at=_parse_datetime(published_node.get_text(" ", strip=True) if published_node else None),
            )
        )
    return entries


def parse_feed_title(feed_text: str) -> str | None:
    try:
        root = ET.fromstring(feed_text)
    except ET.ParseError:
        return _parse_feed_title_with_bs4(feed_text)

    if root.tag.endswith("rss"):
        channel = root.find("channel")
        return _first_text(channel, "title")

    namespace = ""
    if root.tag.startswith("{"):
        namespace = root.tag.split("}", 1)[0] + "}"
    return _first_text(root, f"{namespace}title")


def _parse_feed_title_with_bs4(feed_text: str) -> str | None:
    soup = BeautifulSoup(feed_text, "html.parser")
    channel = soup.find("channel")
    if channel:
        title_node = channel.find("title")
        if title_node:
            title = title_node.get_text(" ", strip=True)
            if title:
                return title
    feed = soup.find("feed")
    if feed:
        title_node = feed.find("title")
        if title_node:
            title = title_node.get_text(" ", strip=True)
            if title:
                return title
    return None


async def validate_feed(
    client: httpx.AsyncClient,
    feed_url: str,
    *,
    site_url: str | None = None,
) -> dict[str, Any]:
    normalized_feed_url = normalize_url(feed_url)
    feed_text = await fetch_text(client, normalized_feed_url)
    entries = parse_feed_entries(feed_text)
    if not entries:
        raise ValueError("Feed did not contain any parseable entries")
    sample_entry = entries[0]
    feed_title = parse_feed_title(feed_text)
    return {
        "feed_url": normalized_feed_url,
        "site_url": site_url,
        "kind": "atom" if "<feed" in feed_text[:200].lower() else "rss",
        "title": feed_title or sample_entry.title,
        "feed_title": feed_title,
        "sample_title": sample_entry.title,
        "entry_count": len(entries),
        "sample_entries": [
            {
                "title": entry.title,
                "link": entry.link,
                "published_at": entry.published_at.isoformat() if entry.published_at else None,
            }
            for entry in entries[:5]
        ],
    }


def parse_sitemap_listing_candidates(homepage_url: str, sitemap_text: str) -> list[dict[str, str | None]]:
    try:
        root = ET.fromstring(sitemap_text)
    except ET.ParseError:
        return []

    namespace = ""
    if root.tag.startswith("{"):
        namespace = root.tag.split("}", 1)[0] + "}"
    homepage_domain = extract_domain(homepage_url)

    if root.tag.endswith("urlset"):
        candidates: list[dict[str, str | None]] = []
        seen: set[str] = set()
        for item in root.findall(f"{namespace}url"):
            loc = _first_text(item, f"{namespace}loc")
            if not loc:
                continue
            normalized_loc = normalize_url(loc)
            if normalized_loc in seen:
                continue
            if extract_domain(normalized_loc) != homepage_domain:
                continue
            parsed = urlparse(normalized_loc)
            path = parsed.path
            if not path or path == "/" or not path.endswith("/"):
                continue
            segments = [segment for segment in path.strip("/").split("/") if segment]
            if not segments or len(segments) > 2:
                continue
            if any(segment.lower() in SITEMAP_LISTING_EXCLUDE_SEGMENTS for segment in segments):
                continue
            seen.add(normalized_loc)
            candidates.append(
                {
                    "listing_url": normalized_loc,
                    "title": _title_from_url(normalized_loc),
                    "site_url": homepage_url,
                    "discovery_method": "sitemap",
                }
            )
        return candidates[:MAX_SITEMAP_LISTING_CANDIDATES]

    if not root.tag.endswith("sitemapindex"):
        return []

    ranked_candidates: list[tuple[int, datetime | None, str]] = []
    for sitemap in root.findall(f"{namespace}sitemap"):
        loc = _first_text(sitemap, f"{namespace}loc")
        if not loc:
            continue
        normalized_loc = normalize_url(loc)
        if extract_domain(normalized_loc) != homepage_domain:
            continue
        lowered = normalized_loc.lower()
        if not any(hint in lowered for hint in SITEMAP_LISTING_HINTS):
            continue
        priority = 0 if any(hint in lowered for hint in ("categories", "category", "sections", "section")) else 1
        ranked_candidates.append((priority, _parse_datetime(_first_text(sitemap, f"{namespace}lastmod")), normalized_loc))

    ranked_candidates.sort(key=lambda item: (item[0], -(item[1].timestamp() if item[1] else 0), item[2]))
    return [
        {
            "listing_url": candidate_url,
            "title": _title_from_url(candidate_url),
            "site_url": homepage_url,
            "discovery_method": "sitemap",
        }
        for _, _, candidate_url in ranked_candidates[:4]
    ]


async def discover_rss_feeds(client: httpx.AsyncClient, homepage_url: str) -> list[dict[str, Any]]:
    normalized_homepage = normalize_url(homepage_url)
    html_text = await fetch_text(client, normalized_homepage)
    raw_candidates = parse_homepage_feed_candidates(normalized_homepage, html_text)
    discovered: list[dict[str, Any]] = []
    seen: set[str] = set()
    pending_hub_candidates: list[dict[str, Any]] = []
    for candidate in raw_candidates:
        feed_url = str(candidate["feed_url"])
        if feed_url in seen:
            continue
        try:
            validated = await validate_feed(client, feed_url, site_url=normalized_homepage)
        except Exception:
            if _looks_like_feed_hub_url(feed_url):
                pending_hub_candidates.append(candidate)
            continue
        seen.add(feed_url)
        discovered.append(
            {
                **validated,
                "discovery_method": candidate.get("discovery_method", "homepage"),
                "candidate_title": candidate.get("title"),
            }
        )

    for hub_candidate in pending_hub_candidates:
        hub_url = str(hub_candidate["feed_url"])
        try:
            hub_html = await fetch_text(client, hub_url)
        except Exception:
            continue
        nested_candidates = parse_homepage_feed_candidates(hub_url, hub_html)
        for nested_candidate in nested_candidates:
            feed_url = str(nested_candidate["feed_url"])
            if feed_url in seen or feed_url == hub_url:
                continue
            try:
                validated = await validate_feed(client, feed_url, site_url=normalized_homepage)
            except Exception:
                continue
            seen.add(feed_url)
            discovered.append(
                {
                    **validated,
                    "discovery_method": nested_candidate.get("discovery_method", hub_candidate.get("discovery_method", "homepage")),
                    "candidate_title": nested_candidate.get("title") or hub_candidate.get("title"),
                }
            )

    return discovered


async def discover_crawl_listings(client: httpx.AsyncClient, homepage_url: str) -> list[dict[str, Any]]:
    normalized_homepage = normalize_url(homepage_url)
    discovered: list[dict[str, Any]] = []
    seen: set[str] = set()

    for suffix in COMMON_SITEMAP_PATHS:
        sitemap_url = normalize_url(urljoin(normalized_homepage, suffix))
        try:
            sitemap_text = await fetch_text(client, sitemap_url)
        except Exception:
            continue

        listing_candidates = parse_sitemap_listing_candidates(normalized_homepage, sitemap_text)
        for candidate in listing_candidates:
            listing_url = str(candidate["listing_url"])
            if listing_url in seen:
                continue

            if listing_url.lower().endswith(".xml"):
                seen.add(listing_url)
                try:
                    nested_text = await fetch_text(client, listing_url)
                except Exception:
                    continue
                for nested_candidate in parse_sitemap_listing_candidates(normalized_homepage, nested_text):
                    nested_url = str(nested_candidate["listing_url"])
                    if nested_url in seen:
                        continue
                    seen.add(nested_url)
                    discovered.append(nested_candidate)
                continue

            seen.add(listing_url)
            discovered.append(candidate)
        if discovered:
            break

    return discovered[:MAX_SITEMAP_LISTING_CANDIDATES]


def extract_links_with_selector(base_url: str, html_text: str, selector: str) -> list[str]:
    soup = BeautifulSoup(html_text, "html.parser")
    urls: list[str] = []
    seen: set[str] = set()
    for node in soup.select(selector):
        href = node.get("href")
        if not href:
            nested = node.find("a", href=True)
            href = nested.get("href") if nested else None
        if not href:
            continue
        absolute = normalize_url(urljoin(base_url, href))
        if absolute in seen:
            continue
        seen.add(absolute)
        urls.append(absolute)
    return urls


def suggest_article_link_selectors(html_text: str) -> list[str]:
    soup = BeautifulSoup(html_text, "html.parser")
    suggestions: list[str] = []
    for anchor in soup.find_all("a", href=True):
        href = str(anchor.get("href") or "").lower()
        if any(token in href for token in ARTICLE_LINK_HINTS):
            class_names = " ".join(anchor.get("class") or [])
            if class_names:
                suggestions.append(f"a.{class_names.split()[0]}")
            elif anchor.get("href"):
                suggestions.append("a[href*='news']")
                break
    if not suggestions:
        suggestions.extend(["article a", ".news-item a", "a[href*='article']"])
    deduped: list[str] = []
    for selector in suggestions:
        if selector not in deduped:
            deduped.append(selector)
    return deduped[:5]


def _sanitize_article_node(node) -> BeautifulSoup:
    fragment = BeautifulSoup(str(node), "html.parser")
    for selector in ARTICLE_NOISE_SELECTORS:
        for noisy_node in fragment.select(selector):
            noisy_node.decompose()
    for heading in fragment.select("h2, h3, h4, h5, h6, strong"):
        text = heading.get_text(" ", strip=True).lower().rstrip(":")
        if any(text.startswith(marker) for marker in ARTICLE_STOP_HEADINGS):
            parent = heading.parent
            if parent is not None:
                parent.decompose()
            else:
                heading.decompose()
    return fragment


def _extract_text_blocks(node) -> str:
    fragment = _sanitize_article_node(node)
    blocks: list[str] = []
    seen: set[str] = set()
    for element in fragment.select("p, figcaption, blockquote, li"):
        if element.find_parent("a") is not None:
            continue
        text = element.get_text(" ", strip=True)
        if not text or len(text) < 3 or text in seen:
            continue
        seen.add(text)
        blocks.append(text)
    if blocks:
        return "\n".join(blocks).strip()
    return fragment.get_text("\n", strip=True).strip()


def extract_article_payload(
    article_url: str,
    html_text: str,
    *,
    content_selector: str | None = None,
    excerpt_selector: str | None = None,
) -> dict[str, Any]:
    soup = BeautifulSoup(html_text, "html.parser")
    canonical = soup.find("link", rel=lambda value: value and "canonical" in value)
    og_url = soup.find("meta", attrs={"property": "og:url"})
    canonical_url = normalize_url(
        canonical.get("href")
        if canonical and canonical.get("href")
        else og_url.get("content")
        if og_url and og_url.get("content")
        else article_url
    )

    title = None
    for selector in (
        'meta[property="og:title"]',
        'meta[name="twitter:title"]',
        "h1",
        "title",
    ):
        node = soup.select_one(selector)
        if not node:
            continue
        title = node.get("content") if node.name == "meta" else node.get_text(" ", strip=True)
        if title:
            break
    if not title:
        title = canonical_url

    excerpt = None
    if excerpt_selector:
        node = soup.select_one(excerpt_selector)
        if node:
            excerpt = node.get_text(" ", strip=True)
    if not excerpt:
        for selector in ('meta[name="description"]', 'meta[property="og:description"]'):
            node = soup.select_one(selector)
            if node and node.get("content"):
                excerpt = str(node.get("content")).strip()
                break

    body_text = ""
    if content_selector:
        node = soup.select_one(content_selector)
        if node:
            body_text = _extract_text_blocks(node)
    if not body_text:
        for selector in ARTICLE_BODY_SELECTORS:
            node = soup.select_one(selector)
            if node:
                body_text = _extract_text_blocks(node)
                if body_text:
                    break
    if not body_text:
        body_text = "\n".join(
            paragraph.get_text(" ", strip=True)
            for paragraph in soup.find_all("p")
            if paragraph.get_text(" ", strip=True)
        ).strip()

    image_url = None
    for selector in ('meta[property="og:image"]', 'meta[name="twitter:image"]'):
        node = soup.select_one(selector)
        if node and node.get("content"):
            image_url = normalize_url(urljoin(article_url, str(node.get("content")).strip()))
            break

    published_at = None
    for selector in (
        'meta[property="article:published_time"]',
        'meta[name="article:published_time"]',
        "time[datetime]",
    ):
        node = soup.select_one(selector)
        if not node:
            continue
        value = node.get("content") if node.name == "meta" else node.get("datetime")
        published_at = _parse_datetime(str(value) if value else None)
        if published_at:
            break

    language = soup.html.get("lang") if soup.html else None
    return {
        "canonical_url": canonical_url,
        "title": title.strip(),
        "excerpt": excerpt.strip() if excerpt else None,
        "content_text": body_text.strip() or None,
        "image_url": image_url,
        "published_at": published_at,
        "language": language,
    }


async def validate_crawl_source(
    client: httpx.AsyncClient,
    *,
    listing_url: str,
    article_link_selector: str,
    content_selector: str,
    excerpt_selector: str | None = None,
) -> dict[str, Any]:
    normalized_listing_url = normalize_url(listing_url)
    listing_html = await fetch_text(client, normalized_listing_url)
    article_links = extract_links_with_selector(normalized_listing_url, listing_html, article_link_selector)
    if not article_links:
        raise ValueError("No article links matched the provided selector")

    extracted_samples: list[dict[str, Any]] = []
    for article_url in article_links[:3]:
        html_text = await fetch_text(client, article_url)
        payload = extract_article_payload(
            article_url,
            html_text,
            content_selector=content_selector,
            excerpt_selector=excerpt_selector,
        )
        if payload["content_text"]:
            extracted_samples.append(
                {
                    "article_url": payload["canonical_url"],
                    "title": payload["title"],
                    "excerpt": payload["excerpt"],
                }
            )
    if not extracted_samples:
        raise ValueError("Article content could not be extracted with the provided selectors")

    return {
        "listing_url": normalized_listing_url,
        "matched_article_count": len(article_links),
        "sample_articles": extracted_samples,
        "heuristic_selectors": suggest_article_link_selectors(listing_html),
    }
