import React, { useState, useEffect } from 'react';
import { stockApi } from '../../api/stockApi';
import { FundSelector, type FundInfo } from './FundSelector';
import { FundInfoCard } from './FundInfoCard';
import { NavReportChart } from './NavReportChart';
import { TopHoldingChart } from './TopHoldingChart';
import { IndustryHoldingChart } from './IndustryHoldingChart';
import { AssetHoldingChart } from './AssetHoldingChart';

/**
 * Funds Tab - displays Vietnamese open-end fund data with selector and charts.
 */
export const FundsTab: React.FC = () => {
    const [funds, setFunds] = useState<FundInfo[]>([]);
    const [selectedFund, setSelectedFund] = useState<string | null>(null);
    const [fundInfo, setFundInfo] = useState<any | null>(null);
    const [navData, setNavData] = useState<any[]>([]);
    const [topHoldings, setTopHoldings] = useState<any[]>([]);
    const [industryHoldings, setIndustryHoldings] = useState<any[]>([]);
    const [assetHoldings, setAssetHoldings] = useState<any[]>([]);
    const [loadingFunds, setLoadingFunds] = useState(true);
    const [loadingData, setLoadingData] = useState(false);

    // Fetch fund listing on mount
    useEffect(() => {
        const fetchFunds = async () => {
            setLoadingFunds(true);
            try {
                const response = await stockApi.getFunds();
                const fundList = response.data.map((f: any) => ({
                    symbol: f.symbol || f.fund_code,
                    name: f.fund_name || f.name || f.symbol || f.fund_code,
                    fund_type: f.fund_type || f.type,
                    fund_owner: f.fund_owner || f.owner || f.management_company,
                })).sort((a: any, b: any) => a.name.localeCompare(b.name));

                setFunds(fundList);

                // Auto-select first fund if available
                if (fundList.length > 0) {
                    setSelectedFund(fundList[0].symbol);
                }
            } catch (error) {
                console.error('Error fetching funds:', error);
            } finally {
                setLoadingFunds(false);
            }
        };

        fetchFunds();
    }, []);

    // Fetch fund data when selected fund changes
    useEffect(() => {
        if (!selectedFund) return;

        const fetchFundData = async () => {
            setLoadingData(true);
            try {
                // Find fund info from the list
                const fund = funds.find(f => f.symbol === selectedFund);
                setFundInfo(fund || { symbol: selectedFund });

                // Fetch all fund data in parallel
                const [navResponse, holdingsResponse, industryResponse, assetResponse] = await Promise.all([
                    stockApi.getFundNavReport(selectedFund),
                    stockApi.getFundTopHolding(selectedFund),
                    stockApi.getFundIndustryHolding(selectedFund),
                    stockApi.getFundAssetHolding(selectedFund),
                ]);

                setNavData(navResponse.data);
                setTopHoldings(holdingsResponse.data);
                setIndustryHoldings(industryResponse.data);
                setAssetHoldings(assetResponse.data);
            } catch (error) {
                console.error(`Error fetching data for fund ${selectedFund}:`, error);
            } finally {
                setLoadingData(false);
            }
        };

        fetchFundData();
    }, [selectedFund, funds]);

    return (
        <div className="space-y-4 p-4">
            {/* Fund Selector */}
            <div className="card bg-base-100 shadow-md border border-base-300">
                <div className="card-body p-4">
                    <FundSelector
                        funds={funds}
                        selectedFund={selectedFund}
                        onFundChange={setSelectedFund}
                        loading={loadingFunds}
                    />
                </div>
            </div>

            {/* Fund Info Card */}
            <FundInfoCard fundInfo={fundInfo} loading={loadingData && !fundInfo} />

            {/* Charts Grid (2x2) */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {/* NAV Report Chart */}
                <div className="card bg-base-100 shadow-md border border-base-300">
                    <div className="card-body p-4">
                        <h3 className="card-title text-base mb-2">NAV Report</h3>
                        <div className="h-80">
                            <NavReportChart data={navData} loading={loadingData} />
                        </div>
                    </div>
                </div>

                {/* Top Holdings Chart */}
                <div className="card bg-base-100 shadow-md border border-base-300">
                    <div className="card-body p-4">
                        <h3 className="card-title text-base mb-2">Top Holdings</h3>
                        <div className="h-80">
                            <TopHoldingChart data={topHoldings} loading={loadingData} />
                        </div>
                    </div>
                </div>

                {/* Industry Allocation Chart */}
                <div className="card bg-base-100 shadow-md border border-base-300">
                    <div className="card-body p-4">
                        <h3 className="card-title text-base mb-2">Industry Allocation</h3>
                        <div className="h-80">
                            <IndustryHoldingChart data={industryHoldings} loading={loadingData} />
                        </div>
                    </div>
                </div>

                {/* Asset Allocation Chart */}
                <div className="card bg-base-100 shadow-md border border-base-300">
                    <div className="card-body p-4">
                        <h3 className="card-title text-base mb-2">Asset Allocation</h3>
                        <div className="h-80">
                            <AssetHoldingChart data={assetHoldings} loading={loadingData} />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default FundsTab;
