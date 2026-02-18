from tools.cafef_scraper.extractor import parse_article_detail
from tools.cafef_scraper.types import SourceType


def test_parse_article_detail_extracts_title_date_and_pdf_urls():
    html = """
    <html>
      <head>
        <meta property="og:title" content="E1VFVN30: Kết thúc giao dịch hoán đổi ngày 10-02-2026" />
        <meta property="article:published_time" content="2026-02-10T08:30:00+07:00" />
      </head>
      <body>
        <a href="https://cafef1.mediacdn.vn/download/100226/e1vfvn30-ket-thuc-giao-dich-hoan-doi-ngay-10-02-2026-0.pdf">PDF</a>
      </body>
    </html>
    """
    detail = parse_article_detail(
        html_text=html,
        url="https://cafef.vn/du-lieu/e1vfvn30-1/example-2401721.chn",
        source_type=SourceType.EVENT_FEED,
    )
    assert detail.title == "E1VFVN30: Kết thúc giao dịch hoán đổi ngày 10-02-2026"
    assert detail.published_at is not None
    assert detail.article_id == "2401721"
    assert len(detail.pdf_urls) == 1
    assert detail.pdf_urls[0].startswith("https://cafef1.mediacdn.vn/download/")


def test_parse_article_detail_uses_canonical_url_and_extracts_path_article_id():
    html = """
    <html>
      <head>
        <meta property="og:title" content="E1VFVN30: Thông báo về danh mục chứng khoán cơ cấu hoán đổi" />
        <meta property="og:url" content="https://cafef.vn/du-lieu/e1vfvn30-863116/e1vfvn30-thong-bao-ve-danh-muc-ty-le-chung-khoan-co-cau-hoan-doi.chn" />
      </head>
      <body>
      </body>
    </html>
    """
    detail = parse_article_detail(
        html_text=html,
        url="https://cafef.vn/du-lieu/e1vfvn30-863116/e1vfvn30.chn",
        source_type=SourceType.ID_SCAN,
    )
    assert detail.url.endswith("co-cau-hoan-doi.chn")
    assert detail.article_id == "863116"
