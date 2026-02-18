from datetime import datetime

from tools.cafef_scraper.classifier import (
    classify_doc_type,
    extract_event_date,
    likely_target_article_url,
    normalize_text,
)
from tools.cafef_scraper.types import DocType


def test_normalize_text_removes_accents():
    raw = "E1VFVN30: Thông báo về danh mục chứng khoán cơ cấu hoán đổi"
    assert normalize_text(raw) == "e1vfvn30 thong bao ve danh muc chung khoan co cau hoan doi"


def test_classify_doc_type_basket_notice():
    value = "E1VFVN30: Thông báo về danh mục chứng khoán cơ cấu hoán đổi ngày 13/02/2026"
    assert classify_doc_type(value) == DocType.BASKET_NOTICE


def test_classify_doc_type_swap_end():
    value = "E1VFVN30: Kết thúc giao dịch hoán đổi ngày 10-02-2026"
    assert classify_doc_type(value) == DocType.SWAP_END


def test_extract_event_date_from_slug():
    event_date, derived = extract_event_date(
        "e1vfvn30-ket-thuc-giao-dich-hoan-doi-ngay-10102016.pdf",
        published_at=None,
    )
    assert event_date is not None
    assert event_date.isoformat() == "2016-10-10"
    assert derived is False


def test_extract_event_date_falls_back_to_published():
    published = datetime(2026, 2, 13, 8, 30)
    event_date, derived = extract_event_date("E1VFVN30", published_at=published)
    assert event_date is not None
    assert event_date.isoformat() == "2026-02-13"
    assert derived is True


def test_extract_event_date_from_legacy_dot_format():
    value = "https://cafef1.mediacdn.vn/download/211014/20141021 - E1VFVN30 - DM CK co cau ngay 21.10.2014.pdf"
    event_date, derived = extract_event_date(value, published_at=None)
    assert event_date is not None
    assert event_date.isoformat() == "2014-10-21"
    assert derived is False


def test_likely_target_article_url_true_for_target_slug():
    url = "https://cafef.vn/du-lieu/e1vfvn30-863116/e1vfvn30-thong-bao-ve-danh-muc-ty-le-chung-khoan-co-cau-hoan-doi.chn"
    assert likely_target_article_url(url)


def test_likely_target_article_url_false_for_irrelevant_slug():
    url = "https://cafef.vn/du-lieu/e1vfvn30-123456/e1vfvn30-dong-cua-tang-giam-trong-phien.chn"
    assert not likely_target_article_url(url)
