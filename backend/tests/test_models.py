import pytest

def test_stock_index_model_exists():
    """Test that StockIndex model exists."""
    # This import should fail if the model is not defined
    from app.db.models import StockIndex
    assert StockIndex is not None
