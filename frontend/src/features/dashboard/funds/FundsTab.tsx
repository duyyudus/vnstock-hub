import React, { useState, useEffect, useMemo } from 'react';
import { stockApi, type FundPerformanceData } from '../../../api/stockApi';
import { FundSelector, type FundInfo } from './FundSelector';
import { FundInfoCard } from './FundInfoCard';
import { NavReportChart } from './NavReportChart';
import { TopHoldingChart } from './TopHoldingChart';
import { IndustryHoldingChart } from './IndustryHoldingChart';
import { AssetHoldingChart } from './AssetHoldingChart';
import { CumulativeGrowthChart } from './CumulativeGrowthChart';
import { RiskReturnScatterPlot } from './RiskReturnScatterPlot';
import { PeriodicReturnHeatmap } from './PeriodicReturnHeatmap';

type ChartType = 'growth' | 'scatter' | 'heatmap';
type Benchmark = 'VNINDEX' | 'VN30';

/**
 * Funds Tab - displays aggregate performance charts and individual fund data.
 */
export const FundsTab: React.FC = () => {
    // --- Aggregate Performance State ---
    const [performanceData, setPerformanceData] = useState<FundPerformanceData | null>(null);
    const [loadingPerformance, setLoadingPerformance] = useState(true);
    const [performanceError, setPerformanceError] = useState<string | null>(null);
    const [performanceWarning, setPerformanceWarning] = useState<string | null>(null);
    const [chartType, setChartType] = useState<ChartType>('growth');
    const [benchmark, setBenchmark] = useState<Benchmark>('VNINDEX');
    const [startYear, setStartYear] = useState<number>(new Date().getFullYear() - 3);
    const [isSyncing, setIsSyncing] = useState(false);
    const [rateLimitUntil, setRateLimitUntil] = useState<number | null>(null);

    // --- Individual Fund State ---
    const [funds, setFunds] = useState<FundInfo[]>([]);
    const [selectedFund, setSelectedFund] = useState<string | null>(null);
    const [fundInfo, setFundInfo] = useState<any | null>(null);
    const [navData, setNavData] = useState<any[]>([]);
    const [topHoldings, setTopHoldings] = useState<any[]>([]);
    const [industryHoldings, setIndustryHoldings] = useState<any[]>([]);
    const [assetHoldings, setAssetHoldings] = useState<any[]>([]);
    const [loadingFunds, setLoadingFunds] = useState(true);
    const [loadingData, setLoadingData] = useState(false);

    // --- Fetch Aggregate Performance Data ---
    useEffect(() => {
        const fetchPerformanceData = async () => {
            setLoadingPerformance(true);
            setPerformanceError(null);
            setPerformanceWarning(null);
            try {
                const response = await stockApi.getFundPerformance();
                setPerformanceData(response);
                setIsSyncing(response.is_syncing || response.is_stale || false);
                setRateLimitUntil(null);

                if (response.funds.length > 0) {
                    const years = new Set<number>();
                    response.funds.forEach(fund => {
                        Object.keys(fund.yearly_returns).forEach(y => years.add(parseInt(y)));
                    });
                    const sortedYears = Array.from(years).sort((a, b) => a - b);
                    if (sortedYears.length > 0) {
                        const defaultStart = new Date().getFullYear() - 3;
                        if (sortedYears.includes(defaultStart)) {
                            setStartYear(defaultStart);
                        } else {
                            setStartYear(sortedYears[0]);
                        }
                    }
                }
            } catch (err) {
                const anyErr = err as any;
                const status = anyErr?.response?.status;
                const retryAfter = anyErr?.response?.data?.retry_after;

                if (status === 429 || status === 503) {
                    const delayMs = (typeof retryAfter === 'number' ? retryAfter : 30) * 1000;
                    setPerformanceWarning('Rate limit reached. Retrying automatically...');
                    setIsSyncing(false);
                    setRateLimitUntil(Date.now() + delayMs);
                    setTimeout(fetchPerformanceData, delayMs);
                } else {
                    console.error('Error fetching fund performance:', err);
                    setPerformanceError('Failed to load fund performance data.');
                }
            } finally {
                setLoadingPerformance(false);
            }
        };

        fetchPerformanceData();
    }, []);



    // Poll for fresh data when syncing
    useEffect(() => {
        if (!isSyncing) return;
        if (rateLimitUntil && Date.now() < rateLimitUntil) return;

        const pollForFreshData = async () => {
            try {
                const response = await stockApi.getFundPerformance();
                if (!response.is_syncing && !response.is_stale) {
                    setPerformanceData(response);
                    setIsSyncing(false);
                }
            } catch (err) {
                console.error('Error polling for fresh performance data:', err);
            }
        };

        const interval = setInterval(pollForFreshData, 5000);
        return () => clearInterval(interval);
    }, [isSyncing, rateLimitUntil]);

    // Performance Memos
    const availableYears = useMemo(() => {
        if (!performanceData?.funds) return [];
        const years = new Set<number>();
        performanceData.funds.forEach(fund => {
            Object.keys(fund.yearly_returns).forEach(y => years.add(parseInt(y)));
        });
        if (performanceData.benchmarks) {
            Object.values(performanceData.benchmarks).forEach(b => {
                Object.keys(b.yearly_returns).forEach(y => years.add(parseInt(y)));
            });
        }
        return Array.from(years).sort((a, b) => a - b);
    }, [performanceData]);

    const performanceFunds = useMemo(() => {
        if (!performanceData?.funds) return [];
        const cutoffStr = `${startYear}-12-31`;
        return performanceData.funds.filter(fund => {
            if (!fund.data_start_date) return false;
            return fund.data_start_date <= cutoffStr;
        });
    }, [performanceData, startYear]);

    const selectedBenchmarkData = useMemo(() => {
        return performanceData?.benchmarks?.[benchmark] || null;
    }, [performanceData, benchmark]);

    // --- Fetch Fund Listing for Selector ---
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
                if (fundList.length > 0 && !selectedFund) {
                    setSelectedFund(fundList[0].symbol);
                }
            } catch (error) {
                console.error('Error fetching funds list:', error);
            } finally {
                setLoadingFunds(false);
            }
        };

        fetchFunds();
    }, []);

    // --- Fetch Selected Fund Details ---
    useEffect(() => {
        if (!selectedFund) return;

        const fetchFundData = async () => {
            setLoadingData(true);
            try {
                const fund = funds.find(f => f.symbol === selectedFund);
                setFundInfo(fund || { symbol: selectedFund });

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
        <div className="space-y-6 p-4">
            {/* --- Aggregate Performance Section --- */}
            <div className="space-y-4">
                <div className="flex items-center justify-between border-b border-base-300 pb-2">
                    <h2 className="text-xl font-bold">Fund Performance Comparison</h2>
                </div>

                {loadingPerformance && !performanceData ? (
                    <div className="flex flex-col items-center justify-center h-64 card bg-base-100 shadow-md border border-base-300">
                        <span className="loading loading-spinner loading-lg text-primary"></span>
                        <p className="mt-4 text-base-content/70">Crunching performance data...</p>
                    </div>
                ) : performanceError ? (
                    <div className="alert alert-error shadow-lg">
                        <span>{performanceError}</span>
                    </div>
                ) : (
                    <>
                        {performanceWarning && (
                            <div className="alert alert-warning shadow-lg">
                                <span>{performanceWarning}</span>
                            </div>
                        )}
                        {/* Performance Controls */}
                        <div className="card bg-base-100 shadow-md border border-base-300">
                            <div className="card-body p-4">
                                <div className="flex flex-wrap items-center gap-4">
                                    <div className="flex items-center gap-2">
                                        <span className="text-sm font-medium text-base-content/70">Chart:</span>
                                        <div className="btn-group">
                                            <button
                                                className={`btn btn-sm ${chartType === 'growth' ? 'btn-primary' : 'btn-ghost'}`}
                                                onClick={() => setChartType('growth')}
                                            >
                                                📈 Growth
                                            </button>
                                            <button
                                                className={`btn btn-sm ${chartType === 'scatter' ? 'btn-primary' : 'btn-ghost'}`}
                                                onClick={() => setChartType('scatter')}
                                            >
                                                ⚖️ Risk/Return
                                            </button>
                                            <button
                                                className={`btn btn-sm ${chartType === 'heatmap' ? 'btn-primary' : 'btn-ghost'}`}
                                                onClick={() => setChartType('heatmap')}
                                            >
                                                🗓️ Heatmap
                                            </button>
                                        </div>
                                    </div>

                                    {availableYears.length > 0 && (
                                        <div className="flex items-center gap-2">
                                            <span className="text-sm font-medium text-base-content/70">Start:</span>
                                            <select
                                                className="select select-sm select-bordered"
                                                value={startYear}
                                                onChange={(e) => setStartYear(parseInt(e.target.value))}
                                            >
                                                {availableYears.map(year => (
                                                    <option key={year} value={year}>{year}</option>
                                                ))}
                                            </select>
                                        </div>
                                    )}

                                    <div className="flex items-center gap-2">
                                        <span className="text-sm font-medium text-base-content/70">Vs:</span>
                                        <button
                                            className={`btn btn-sm ${benchmark === 'VN30' ? 'btn-secondary' : 'btn-accent'}`}
                                            onClick={() => setBenchmark(prev => prev === 'VNINDEX' ? 'VN30' : 'VNINDEX')}
                                            title={`Click to switch to ${benchmark === 'VN30' ? 'VN-Index' : 'VN30'}`}
                                        >
                                            {benchmark === 'VNINDEX' ? 'VN-Index' : 'VN30'}
                                        </button>
                                    </div>

                                    <div className="ml-auto flex items-center gap-4 text-sm text-base-content/50">
                                        <span>{performanceFunds.length} funds</span>
                                        {isSyncing && (
                                            <span className="text-warning flex items-center gap-1">
                                                <span className="loading loading-spinner loading-xs"></span>
                                                Syncing...
                                            </span>
                                        )}
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Performance Chart */}
                        <div className="card bg-base-100 shadow-md border border-base-300">
                            <div className="card-body p-4">
                                <h3 className="card-title text-base mb-4">
                                    {chartType === 'growth' && '📈 Cumulative Growth (Normalized NAV)'}
                                    {chartType === 'scatter' && '⚖️ Risk vs Return'}
                                    {chartType === 'heatmap' && '🗓️ Yearly Performance Heatmap'}
                                </h3>
                                <div className={chartType === 'heatmap' ? 'w-full' : 'w-full h-[750px]'}>
                                    {chartType === 'growth' && (
                                        <CumulativeGrowthChart
                                            funds={performanceFunds}
                                            benchmark={selectedBenchmarkData}
                                            startYear={startYear}
                                        />
                                    )}
                                    {chartType === 'scatter' && (
                                        <RiskReturnScatterPlot
                                            funds={performanceFunds}
                                            benchmark={selectedBenchmarkData}
                                            startYear={startYear}
                                        />
                                    )}
                                    {chartType === 'heatmap' && (
                                        <PeriodicReturnHeatmap
                                            funds={performanceFunds}
                                            benchmark={selectedBenchmarkData}
                                            startYear={startYear}
                                        />
                                    )}
                                </div>
                            </div>
                        </div>
                    </>
                )}
            </div>

            <div className="divider opacity-50"></div>

            {/* --- Individual Fund Details Section --- */}
            <div className="space-y-4">
                <div className="flex items-center justify-between border-b border-base-300 pb-2">
                    <h2 className="text-xl font-bold">Individual Fund Details</h2>
                </div>

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

                <FundInfoCard fundInfo={fundInfo} loading={loadingData && !fundInfo} />

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    <div className="card bg-base-100 shadow-md border border-base-300">
                        <div className="card-body p-4">
                            <h3 className="card-title text-base mb-2">NAV Report</h3>
                            <div className="h-80">
                                <NavReportChart data={navData} loading={loadingData} />
                            </div>
                        </div>
                    </div>

                    <div className="card bg-base-100 shadow-md border border-base-300">
                        <div className="card-body p-4">
                            <h3 className="card-title text-base mb-2">Top Holdings</h3>
                            <div className="h-80">
                                <TopHoldingChart data={topHoldings} loading={loadingData} />
                            </div>
                        </div>
                    </div>

                    <div className="card bg-base-100 shadow-md border border-base-300">
                        <div className="card-body p-4">
                            <h3 className="card-title text-base mb-2">Industry Allocation</h3>
                            <div className="h-80">
                                <IndustryHoldingChart data={industryHoldings} loading={loadingData} />
                            </div>
                        </div>
                    </div>

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
        </div>
    );
};

export default FundsTab;
