import pytest
from unittest.mock import patch
from sqlalchemy import select, delete
from app.services.vnstock_service import vnstock_service
from app.db.models import StockIndex
import pandas as pd

@pytest.mark.asyncio
async def test_sync_indices(db_session):
    # Clean up before test
    await db_session.execute(delete(StockIndex))
    await db_session.commit()

    # Mock data using a valid group symbol (VN30) so it's not filtered out
    mock_data = pd.DataFrame([
        {
            'symbol': 'VN30', 
            'name': 'VN30 Index', 
            'full_name': 'VN30 Index Full Name', 
            'group': 'HOSE',
            'index_id': 999
        }
    ])

    # Patch the method that fetches indices from vnstock
    with patch.object(vnstock_service.indices, '_fetch_all_indices_from_lib', return_value=mock_data):
        await vnstock_service.sync_indices()

    # Verify DB
    result = await db_session.execute(select(StockIndex).where(StockIndex.symbol == 'VN30'))
    index = result.scalar_one_or_none()
    
    assert index is not None
    assert index.symbol == 'VN30'
    # The sync logic prioritizes full_name
    assert index.name == 'VN30 Index Full Name' 
    assert index.group == 'HOSE'
