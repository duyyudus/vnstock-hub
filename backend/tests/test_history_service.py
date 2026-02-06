from datetime import date

from app.services.vnstock_service.history import HistoryService


def test_check_prices_staleness_allows_late_first_point_if_latest_is_fresh():
    service = HistoryService()
    stocks_data = {
        "AAA": [
            {"date": "2025-01-10", "close": 10.0},
            {"date": "2026-02-06", "close": 12.0},
        ]
    }

    is_stale = service._check_prices_staleness(
        stocks_data=stocks_data,
        requested_symbols=["AAA"],
        end_date=date(2026, 2, 6),
    )

    assert is_stale is False


def test_check_prices_staleness_marks_missing_symbol_as_stale():
    service = HistoryService()
    stocks_data = {
        "AAA": [{"date": "2026-02-06", "close": 12.0}],
    }

    is_stale = service._check_prices_staleness(
        stocks_data=stocks_data,
        requested_symbols=["AAA", "BBB"],
        end_date=date(2026, 2, 6),
    )

    assert is_stale is True


def test_check_prices_staleness_marks_old_latest_date_as_stale():
    service = HistoryService()
    stocks_data = {
        "AAA": [{"date": "2026-01-20", "close": 12.0}],
    }

    is_stale = service._check_prices_staleness(
        stocks_data=stocks_data,
        requested_symbols=["AAA"],
        end_date=date(2026, 2, 6),
    )

    assert is_stale is True
