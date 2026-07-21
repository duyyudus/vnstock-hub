import React, { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { stockApi, type FundOverviewResponse, type FundPerformanceData, type Stock } from '../../../api/stockApi';
import { FundSelector, type FundInfo } from './FundSelector';
import { FundInfoCard } from './FundInfoCard';
import { FundOverview } from './FundOverview';
import { NavReportChart } from './NavReportChart';
import { TopHoldingChart } from './TopHoldingChart';
import { IndustryHoldingChart } from '../components/IndustryHoldingChart';
import { AssetHoldingChart } from './AssetHoldingChart';
import { CumulativeGrowthChart } from './CumulativeGrowthChart';
import { RiskReturnScatterPlot } from './RiskReturnScatterPlot';
import { PeriodicReturnHeatmap } from './PeriodicReturnHeatmap';

type ChartType = 'growth' | 'scatter' | 'heatmap';
type Benchmark = 'VNINDEX' | 'VN30';
type ViewMode = 'overview' | 'details';
type FundApiRecord = Record<string, unknown>;

interface IndustryHoldingStock {
    ticker: string;
    companyName?: string;
    marketValue?: number;
    allocation?: number;
}

interface EnrichedIndustryHoldingRecord extends FundApiRecord {
    stocks?: IndustryHoldingStock[];
}

interface EnrichedTopHoldingRecord extends FundApiRecord {
    company_name?: string;
}

const getStringValue = (value: unknown): string | null => {
    return typeof value === 'string' && value.trim() ? value.trim() : null;
};

const getNumberValue = (value: unknown): number | null => {
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
};

const normalizeIndustryKey = (value: string | null): string | null => {
    return value ? value.trim().toLocaleLowerCase() : null;
};

/**
 * Funds Tab - displays aggregate performance charts and individual fund data.
 */
export const FundsTab: React.FC = () => {
    const [viewMode, setViewMode] = useState<ViewMode>('overview');

    // --- Overview State ---
    const [overviewData, setOverviewData] = useState<FundOverviewResponse | null>(null);
    const [loadingOverview, setLoadingOverview] = useState(true);
    const [overviewError, setOverviewError] = useState<string | null>(null);

    // --- Aggregate Performance State ---
    const [performanceData, setPerformanceData] = useState<FundPerformanceData | null>(null);
    const [loadingPerformance, setLoadingPerformance] = useState(true);
    const [performanceError, setPerformanceError] = useState<string | null>(null);
    const [chartType, setChartType] = useState<ChartType>('growth');
    const [benchmark, setBenchmark] = useState<Benchmark>('VNINDEX');
    const [startYear, setStartYear] = useState<number>(() => new Date().getFullYear());

    // --- Individual Fund State ---
    const [funds, setFunds] = useState<FundInfo[]>([]);
    const [selectedFund, setSelectedFund] = useState<string | null>(null);
    const [fundInfo, setFundInfo] = useState<FundInfo | null>(null);
    const [navData, setNavData] = useState<FundApiRecord[]>([]);
    const [topHoldings, setTopHoldings] = useState<FundApiRecord[]>([]);
    const [industryHoldings, setIndustryHoldings] = useState<FundApiRecord[]>([]);
    const [assetHoldings, setAssetHoldings] = useState<FundApiRecord[]>([]);
    const [topHoldingCompanyNames, setTopHoldingCompanyNames] = useState<Record<string, string>>({});
    const [loadingFunds, setLoadingFunds] = useState(true);
    const [loadingData, setLoadingData] = useState(false);
    const fundDetailsSectionRef = useRef<HTMLDivElement | null>(null);

    const fetchOverviewData = useCallback(async () => {
        setLoadingOverview(true);
        setOverviewError(null);
        try {
            const response = await stockApi.getFundOverview();
            setOverviewData(response);
        } catch (err) {
            console.error('Error fetching fund overview:', err);
            setOverviewError('Failed to load fund holdings overview.');
        } finally {
            setLoadingOverview(false);
        }
    }, []);

    // --- Fetch Aggregate Holdings Overview ---
    useEffect(() => {
        fetchOverviewData();
    }, [fetchOverviewData]);

    // --- Fetch Aggregate Performance Data ---
    useEffect(() => {
        const fetchPerformanceData = async () => {
            setLoadingPerformance(true);
            setPerformanceError(null);
            try {
                const response = await stockApi.getFundPerformance();
                setPerformanceData(response);

                if (response.funds.length > 0) {
                    const currentYear = new Date().getFullYear();
                    const years = new Set<number>();
                    response.funds.forEach((fund) => {
                        Object.keys(fund.yearly_returns).forEach((y) => years.add(parseInt(y, 10)));
                    });
                    const sortedYears = Array.from(years).sort((a, b) => a - b);
                    if (sortedYears.length > 0) {
                        const defaultStart = currentYear;
                        if (sortedYears.includes(defaultStart)) {
                            setStartYear(defaultStart);
                        } else {
                            setStartYear(sortedYears[0]);
                        }
                    }
                }
            } catch (err) {
                console.error('Error fetching fund performance:', err);
                setPerformanceError('Failed to load fund performance data.');
            } finally {
                setLoadingPerformance(false);
            }
        };

        fetchPerformanceData();
    }, []);
    // Performance Memos
    const availableYears = useMemo(() => {
        if (!performanceData?.funds) return [];
        const years = new Set<number>();
        performanceData.funds.forEach(fund => {
            Object.keys(fund.yearly_returns).forEach((y) => years.add(parseInt(y, 10)));
        });
        if (performanceData.benchmarks) {
            Object.values(performanceData.benchmarks).forEach(b => {
                Object.keys(b.yearly_returns).forEach((y) => years.add(parseInt(y, 10)));
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

    const fundTypesBySymbol = useMemo(() => {
        return funds.reduce<Record<string, string | undefined>>((lookup, fund) => {
            lookup[fund.symbol] = fund.fund_type;
            lookup[fund.symbol.toUpperCase()] = fund.fund_type;
            return lookup;
        }, {});
    }, [funds]);

    const selectedBenchmarkData = useMemo(() => {
        return performanceData?.benchmarks?.[benchmark] || null;
    }, [performanceData, benchmark]);

    const enrichedTopHoldings = useMemo<EnrichedTopHoldingRecord[]>(() => {
        return topHoldings.map((holding) => {
            const ticker = getStringValue(holding.ticker)
                || getStringValue(holding.stock_code)
                || getStringValue(holding.symbol);
            const normalizedTicker = ticker?.toUpperCase();
            const companyName = getStringValue(holding.company_name)
                || getStringValue(holding.companyName)
                || (normalizedTicker ? topHoldingCompanyNames[normalizedTicker] : null);

            return companyName
                ? {
                    ...holding,
                    company_name: companyName,
                }
                : holding;
        });
    }, [topHoldings, topHoldingCompanyNames]);

    const enrichedIndustryHoldings = useMemo<EnrichedIndustryHoldingRecord[]>(() => {
        if (industryHoldings.length === 0) {
            return [];
        }

        const stocksByIndustry = new Map<string, IndustryHoldingStock[]>();

        enrichedTopHoldings.forEach((holding) => {
            const industry = getStringValue(holding.industry);
            const industryKey = normalizeIndustryKey(industry);
            const ticker = getStringValue(holding.ticker)
                || getStringValue(holding.stock_code)
                || getStringValue(holding.symbol);

            if (!industryKey || !ticker) {
                return;
            }

            const stock: IndustryHoldingStock = {
                ticker,
                companyName: getStringValue(holding.company_name)
                    || getStringValue(holding.companyName)
                    || undefined,
                marketValue: getNumberValue(holding.market_value)
                    ?? getNumberValue(holding.marketValue)
                    ?? undefined,
                allocation: getNumberValue(holding.allocation)
                    ?? getNumberValue(holding.net_asset_percent)
                    ?? getNumberValue(holding.weight)
                    ?? getNumberValue(holding.percentage)
                    ?? undefined,
            };

            const industryStocks = stocksByIndustry.get(industryKey) || [];
            industryStocks.push(stock);
            stocksByIndustry.set(industryKey, industryStocks);
        });

        stocksByIndustry.forEach((stocks, key) => {
            stocksByIndustry.set(
                key,
                [...stocks].sort((a, b) => (b.allocation || 0) - (a.allocation || 0)),
            );
        });

        return industryHoldings.map((industryHolding) => {
            const industry = getStringValue(industryHolding.industry)
                || getStringValue(industryHolding.sector)
                || getStringValue(industryHolding.industry_name);
            const industryKey = normalizeIndustryKey(industry);

            return {
                ...industryHolding,
                stocks: industryKey ? stocksByIndustry.get(industryKey) : undefined,
            };
        });
    }, [enrichedTopHoldings, industryHoldings]);

    // --- Fetch Fund Listing for Selector ---
    useEffect(() => {
        const fetchFunds = async () => {
            setLoadingFunds(true);
            try {
                const response = await stockApi.getFunds();
                const toStringValue = (value: string | number | boolean | null | undefined) => {
                    return typeof value === 'string' ? value : '';
                };
                const fundList = response.data.map((fundRecord): FundInfo => {
                    const symbol = toStringValue(fundRecord.symbol) || toStringValue(fundRecord.fund_code);
                    const name = toStringValue(fundRecord.fund_name)
                        || toStringValue(fundRecord.name)
                        || symbol;

                    return {
                        symbol,
                        name,
                        fund_type: toStringValue(fundRecord.fund_type) || toStringValue(fundRecord.type) || undefined,
                        fund_owner: toStringValue(fundRecord.fund_owner)
                            || toStringValue(fundRecord.owner)
                            || toStringValue(fundRecord.management_company)
                            || undefined,
                    };
                }).filter((fund) => fund.symbol).sort((a, b) => a.name.localeCompare(b.name));

                setFunds(fundList);
                setSelectedFund((current) => {
                    if (!current) {
                        return null;
                    }

                    return fundList.some((fund) => fund.symbol === current) ? current : null;
                });
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
        if (!selectedFund) {
            setFundInfo(null);
            setNavData([]);
            setTopHoldings([]);
            setIndustryHoldings([]);
            setAssetHoldings([]);
            setTopHoldingCompanyNames({});
            setLoadingData(false);
            return;
        }

        const fetchFundData = async () => {
            setLoadingData(true);
            try {
                const fund = funds.find(f => f.symbol === selectedFund);
                setFundInfo(fund || { symbol: selectedFund, name: selectedFund });

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
                await fetchOverviewData();
            } catch (error) {
                console.error(`Error fetching data for fund ${selectedFund}:`, error);
            } finally {
                setLoadingData(false);
            }
        };

        fetchFundData();
    }, [selectedFund, funds, fetchOverviewData]);

    useEffect(() => {
        const tickers = Array.from(new Set(
            topHoldings
                .map((holding) => getStringValue(holding.ticker)
                    || getStringValue(holding.stock_code)
                    || getStringValue(holding.symbol))
                .filter((ticker): ticker is string => Boolean(ticker))
                .map((ticker) => ticker.toUpperCase()),
        ));

        if (tickers.length === 0) {
            setTopHoldingCompanyNames({});
            return;
        }

        let cancelled = false;

        const fetchTopHoldingCompanyNames = async () => {
            try {
                const response = await stockApi.getStockQuotes(tickers);
                if (cancelled) {
                    return;
                }

                const companyNames = response.stocks.reduce<Record<string, string>>((acc, stock: Stock) => {
                    const companyName = stock.company_name.trim();
                    if (companyName) {
                        acc[stock.ticker.toUpperCase()] = companyName;
                    }
                    return acc;
                }, {});

                setTopHoldingCompanyNames(companyNames);
            } catch (error) {
                console.error('Error fetching top holding company names:', error);
                if (!cancelled) {
                    setTopHoldingCompanyNames({});
                }
            }
        };

        fetchTopHoldingCompanyNames();

        return () => {
            cancelled = true;
        };
    }, [topHoldings]);

    const handleFundSelectFromGrowthChart = (symbol: string) => {
        if (!funds.some((fund) => fund.symbol === symbol)) {
            return;
        }

        setSelectedFund(symbol);
        setViewMode('details');
        requestAnimationFrame(() => {
            fundDetailsSectionRef.current?.scrollIntoView({
                behavior: 'smooth',
                block: 'start',
            });
        });
    };

    return (
        <div className="space-y-6 p-4">
            <div className="overflow-x-auto">
                <div className="join min-w-max">
                    <button
                        type="button"
                        className={`join-item btn btn-sm ${viewMode === 'overview' ? 'btn-primary' : 'btn-ghost'}`}
                        onClick={() => setViewMode('overview')}
                    >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                        </svg>
                        Overview
                    </button>
                    <button
                        type="button"
                        className={`join-item btn btn-sm ${viewMode === 'details' ? 'btn-primary' : 'btn-ghost'}`}
                        onClick={() => setViewMode('details')}
                    >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 17v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm6 0V7a2 2 0 00-2-2h-2a2 2 0 00-2 2v10a2 2 0 002 2h2a2 2 0 002-2zm6 0v-4a2 2 0 00-2-2h-2a2 2 0 00-2 2v4a2 2 0 002 2h2a2 2 0 002-2z" />
                        </svg>
                        Details
                    </button>
                </div>
            </div>

            {viewMode === 'overview' ? (
                <FundOverview
                    data={overviewData}
                    loading={loadingOverview}
                    error={overviewError}
                />
            ) : (
                <>

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
                                        {performanceData?.is_syncing && (
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
                                            fundTypesBySymbol={fundTypesBySymbol}
                                            onFundSelect={handleFundSelectFromGrowthChart}
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
            <div ref={fundDetailsSectionRef} className="space-y-4">
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

                {selectedFund && (
                    <>
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
                                        <TopHoldingChart data={enrichedTopHoldings} loading={loadingData} />
                                    </div>
                                </div>
                            </div>

                            <div className="card bg-base-100 shadow-md border border-base-300">
                                <div className="card-body p-4">
                                    <h3 className="card-title text-base mb-2">Industry Allocation</h3>
                                    <div className="h-80">
                                        <IndustryHoldingChart data={enrichedIndustryHoldings} loading={loadingData} />
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
                    </>
                )}
            </div>
                </>
            )}
        </div>
    );
};

export default FundsTab;
