def test_stock_index_model_exists():
    """Test that StockIndex model exists."""
    # This import should fail if the model is not defined
    from app.db.models import StockIndex
    assert StockIndex is not None

def test_daily_history_and_history_sync_models_expose_renamed_schema():
    from app.db.models import StockDailyHistory, StockHistorySyncState

    daily_columns = set(StockDailyHistory.__table__.columns.keys())
    sync_indexes = {index.name for index in StockHistorySyncState.__table__.indexes}
    sync_constraints = {
        constraint.name
        for constraint in StockHistorySyncState.__table__.constraints
        if getattr(constraint, "name", None)
    }

    assert StockDailyHistory.__tablename__ == "stock_daily_history"
    assert StockHistorySyncState.__tablename__ == "stock_history_sync_state"
    assert "foreign_buy_value" in daily_columns
    assert "foreign_net_value" in daily_columns
    assert "matched_value" in daily_columns
    assert "total_value" in daily_columns
    assert "prop_buy_volume" in daily_columns
    assert "prop_sell_value" in daily_columns
    assert "ix_stock_history_sync_state_symbol" in sync_indexes
    assert "uq_stock_history_sync_state_symbol" in sync_constraints
