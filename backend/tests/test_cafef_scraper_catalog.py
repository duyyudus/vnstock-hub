from datetime import date

from tools.cafef_scraper.catalog import Catalog
from tools.cafef_scraper.types import DocType, SourceType


def test_article_queue_resume_and_reset(tmp_path):
    catalog = Catalog(tmp_path / "catalog.sqlite")
    scope_key = "backfill:2026-02-10:2026-02-11"

    inserted = catalog.enqueue_article_urls(
        scope_key=scope_key,
        source_type=SourceType.EVENT_FEED,
        urls=[
            "https://cafef.vn/du-lieu/e1vfvn30-a-1.chn",
            "https://cafef.vn/du-lieu/e1vfvn30-b-2.chn",
        ],
    )
    assert inserted == 2

    pending = catalog.list_pending_article_queue(scope_key)
    assert len(pending) == 2

    catalog.mark_article_queue_fetched(int(pending[0]["id"]))
    pending_after = catalog.list_pending_article_queue(scope_key)
    assert len(pending_after) == 1

    catalog.set_state(f"discovery:{scope_key}:event_done", "1")
    catalog.set_state(f"discovery:{scope_key}:event_next_page", "12")
    catalog.reset_discovery_scope(scope_key)

    assert catalog.list_pending_article_queue(scope_key) == []
    assert catalog.get_state(f"discovery:{scope_key}:event_done") is None

    catalog.close()


def test_reset_range_resets_document_statuses(tmp_path):
    catalog = Catalog(tmp_path / "catalog.sqlite")
    target_date = date(2026, 2, 10)
    catalog.ensure_coverage_range(target_date, target_date)
    doc_id = catalog.upsert_document(
        article_id="abc",
        article_url="https://cafef.vn/doc.chn",
        source_type=SourceType.EVENT_FEED,
        doc_type=DocType.SWAP_END,
        event_date=target_date,
        pdf_url="https://cafef1.mediacdn.vn/download/100226/file.pdf",
        derived_from_published=False,
    )
    assert doc_id > 0
    catalog.reset_range(target_date, target_date, {DocType.SWAP_END})
    failed = catalog.list_failed_documents(start_date=target_date, end_date=target_date)
    assert failed == []
    catalog.close()

