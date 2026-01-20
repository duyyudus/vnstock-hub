import asyncio
import sys
import os

# Add the backend directory to sys.path to allow imports from app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.vnstock_service import vnstock_service
from app.core.config import settings

async def test_vn100_fetch():
    print(f"Testing VN100 fetch with limit: {settings.vn100_limit}")
    
    # Mock _enrich_with_price_changes to avoid DB connection issues
    from unittest.mock import MagicMock
    vnstock_service._enrich_with_price_changes = MagicMock(side_effect=lambda x: x)
    
    # We call _fetch_vn100_data directly
    loop = asyncio.get_event_loop()
    stocks = await loop.run_in_executor(None, vnstock_service._fetch_vn100_data)
    
    print(f"Total stocks fetched: {len(stocks)}")
    
    if len(stocks) == 0:
        print("Error: No stocks fetched!")
        sys.exit(1)
        
    print("\nTop 10 stocks by market cap:")
    for i, stock in enumerate(stocks[:10]):
        print(f"{i+1}. {stock.ticker}: Price={stock.price}, Market Cap={stock.market_cap}, Name={stock.company_name}")
        
    # Check for expected major stocks
    major_stocks = ['VCB', 'VIC', 'VNM', 'FPT', 'HPG']
    found_majors = [s.ticker for s in stocks if s.ticker in major_stocks]
    
    print(f"\nMajor stocks found: {found_majors}")
    
    if len(stocks) >= 50:
        print("\nSuccess: Fetched more than 50 stocks (likely full VN100 list depending on data availability)")
    else:
        print(f"\nWarning: Only fetched {len(stocks)} stocks. Expected closer to 100.")
        
    if len(found_majors) < 3:
        print("Error: Missing too many major stocks!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_vn100_fetch())
