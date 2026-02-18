from tools.cafef_scraper.discovery import (
    build_symbol_probe_url,
    parse_event_feed_urls,
    parse_sitemap_locs,
)


def test_parse_event_feed_urls_extracts_relative_and_absolute():
    html = """
    <li><a href="/du-lieu/e1vfvn30-123/event-1.chn">A</a></li>
    <li><a href="https://cafef.vn/du-lieu/e1vfvn30-124/event-2.chn">B</a></li>
    """
    urls = parse_event_feed_urls(html)
    assert "https://cafef.vn/du-lieu/e1vfvn30-123/event-1.chn" in urls
    assert "https://cafef.vn/du-lieu/e1vfvn30-124/event-2.chn" in urls


def test_parse_sitemap_locs_works_with_namespace():
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://cafef.vn/a.chn</loc></url>
      <url><loc>https://cafef.vn/b.chn</loc></url>
    </urlset>
    """
    assert parse_sitemap_locs(xml) == ["https://cafef.vn/a.chn", "https://cafef.vn/b.chn"]


def test_build_symbol_probe_url():
    assert (
        build_symbol_probe_url(863116, "E1VFVN30")
        == "https://cafef.vn/du-lieu/e1vfvn30-863116/e1vfvn30.chn"
    )
