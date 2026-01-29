import pytest
from unittest.mock import patch, MagicMock
from sqlalchemy import select, delete
from app.services.vnstock_service import vnstock_service
from app.db.models import StockIndex
from app.db.database import async_session
import pandas as pd

@pytest.mark.asyncio
async def test_sync_indices():
    # Clean up before test
    async with async_session() as session:
        await session.execute(delete(StockIndex))
        await session.commit()

    # Mock data
    mock_data = pd.DataFrame([
        {
            'symbol': 'TEST_INDEX', 
            'name': 'TEST_INDEX', 
            'full_name': 'Test Index Full Name', 
            'group': 'Test Group',
            'index_id': 999
        }
    ])

    # Patch the method that fetches indices from vnstock
    # We haven't implemented it yet, but we will call it _fetch_all_indices_from_lib
    with patch.object(vnstock_service.indices, '_fetch_all_indices_from_lib', return_value=mock_data):
        await vnstock_service.sync_indices()

    # Verify DB
    async with async_session() as session:
        result = await session.execute(select(StockIndex).where(StockIndex.symbol == 'TEST_INDEX'))
        index = result.scalar_one_or_none()
        
        assert index is not None
        assert index.symbol == 'TEST_INDEX'
        assert index.name == 'Test Index Full Name'
        assert index.group == 'Test Group'
