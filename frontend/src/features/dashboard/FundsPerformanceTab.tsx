import React, { useState, useEffect, useMemo } from 'react';
import { stockApi, type FundPerformanceData } from '../../api/stockApi';
import { CumulativeGrowthChart } from './CumulativeGrowthChart';
import { RiskReturnScatterPlot } from './RiskReturnScatterPlot';
import { PeriodicReturnHeatmap } from './PeriodicReturnHeatmap';

type ChartType = 'growth' | 'scatter' | 'heatmap';
type Benchmark = 'VNINDEX' | 'VN30';

/**
 * Funds Performance Tab - Compare performance of all open-end funds.
 */
export const FundsPerformanceTab: React.FC = () => {
    const [data, setData] = useState<FundPerformanceData | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [chartType, setChartType] = useState<ChartType>('growth');
    const [benchmark, setBenchmark] = useState<Benchmark>('VNINDEX');
    const [startYear, setStartYear] = useState<number>(new Date().getFullYear() - 3); // Default to last 3 years
    const [isSyncing, setIsSyncing] = useState(false);

    useEffect(() => {
        const fetchData = async () => {
            setLoading(true);
            setError(null);
            try {
                const response = await stockApi.getFundPerformance();
                setData(response);
                // Use is_syncing for indicator - more accurate than is_stale
                setIsSyncing(response.is_syncing || response.is_stale || false);

                // Set initial start year to something reasonable like 3 years ago if available
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
                console.error('Error fetching fund performance:', err);
                setError('Failed to load fund performance data. Please try again later.');
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, []);

    // Dispatch sync status to Dashboard for header indicator
    useEffect(() => {
        window.dispatchEvent(new CustomEvent('fundSyncStatusChange', {
            detail: { isSyncing: isSyncing }
        }));
    }, [isSyncing]);

    // Poll for fresh data when syncing
    useEffect(() => {
        if (!isSyncing) return;

        const pollForFreshData = async () => {
            try {
                const response = await stockApi.getFundPerformance();
                // Stop polling when sync is complete
                if (!response.is_syncing && !response.is_stale) {
                    setData(response);
                    setIsSyncing(false);
                    console.log('Fresh fund performance data received');
                }
            } catch (err) {
                console.error('Error polling for fresh data:', err);
            }
        };

        // Poll every 5 seconds while syncing
        const interval = setInterval(pollForFreshData, 5000);
        return () => clearInterval(interval);
    }, [isSyncing]);

    // Get all unique years available in the data for the start year selector
    const availableYears = useMemo(() => {
        if (!data?.funds) return [];
        const years = new Set<number>();
        data.funds.forEach(fund => {
            Object.keys(fund.yearly_returns).forEach(y => years.add(parseInt(y)));
        });
        if (data.benchmarks) {
            Object.values(data.benchmarks).forEach(b => {
                Object.keys(b.yearly_returns).forEach(y => years.add(parseInt(y)));
            });
        }
        return Array.from(years).sort((a, b) => a - b);
    }, [data]);

    // Filter funds based on selected startYear
    const filteredFunds = useMemo(() => {
        if (!data?.funds) return [];

        const cutoffStr = `${startYear}-12-31`; // Fund must have started by the end of the selected start year

        return data.funds.filter(fund => {
            // Fund must have data starting before the cutoff date
            if (!fund.data_start_date) return false;
            return fund.data_start_date <= cutoffStr;
        });
    }, [data, startYear]);

    const selectedBenchmark = useMemo(() => {
        return data?.benchmarks?.[benchmark] || null;
    }, [data, benchmark]);

    if (loading) {
        // ... existing loading state ...
        return (
            <div className="flex flex-col items-center justify-center h-96">
                <span className="loading loading-spinner loading-lg text-primary"></span>
                <p className="mt-4 text-base-content/70">Loading fund performance data...</p>
                <p className="mt-2 text-sm text-base-content/50">This may take a moment as we crunch the numbers.</p>
            </div>
        );
    }

    if (error) {
        // ... existing error state ...
        return (
            <div className="flex flex-col items-center justify-center h-96">
                <div className="text-error text-lg mb-2">⚠️ Error</div>
                <p className="text-base-content/70">{error}</p>
            </div>
        );
    }

    if (!data || data.funds.length === 0) {
        return (
            <div className="flex flex-col items-center justify-center h-96">
                <p className="text-base-content/60">No fund performance data available.</p>
            </div>
        );
    }

    return (
        <div className="space-y-4 p-4">
            {/* Controls Row */}
            <div className="card bg-base-100 shadow-md border border-base-300">
                <div className="card-body p-4">
                    <div className="flex flex-wrap items-center gap-4">
                        {/* Chart Type Selector */}
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

                        {/* Start Year Selector */}
                        {availableYears.length > 0 && (
                            <div className="flex items-center gap-2">
                                <span className="text-sm font-medium text-base-content/70">Start Year:</span>
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

                        {/* Benchmark Toggle */}
                        <div className="flex items-center gap-2">
                            <span className="text-sm font-medium text-base-content/70">Benchmark:</span>
                            <label className="swap swap-flip">
                                <input
                                    type="checkbox"
                                    checked={benchmark === 'VN30'}
                                    onChange={(e) => setBenchmark(e.target.checked ? 'VN30' : 'VNINDEX')}
                                />
                                <div className="swap-on btn btn-sm btn-secondary">VN30</div>
                                <div className="swap-off btn btn-sm btn-accent">VN-Index</div>
                            </label>
                        </div>

                        {/* Stats */}
                        <div className="ml-auto flex items-center gap-4 text-sm text-base-content/50">
                            <span>{filteredFunds.length} funds</span>
                            {isSyncing && (
                                <span className="text-warning flex items-center gap-1">
                                    <span className="loading loading-spinner loading-xs"></span>
                                    Syncing NAV data...
                                </span>
                            )}
                            {data.last_updated && (
                                <span>Updated: {new Date(data.last_updated).toLocaleDateString()}</span>
                            )}
                        </div>
                    </div>
                </div>
            </div>

            {/* Chart Container */}
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
                                funds={filteredFunds}
                                benchmark={selectedBenchmark}
                                startYear={startYear}
                            />
                        )}
                        {chartType === 'scatter' && (
                            <RiskReturnScatterPlot
                                funds={filteredFunds}
                                benchmark={selectedBenchmark}
                                startYear={startYear}
                            />
                        )}
                        {chartType === 'heatmap' && (
                            <PeriodicReturnHeatmap
                                funds={filteredFunds}
                                benchmark={selectedBenchmark}
                                startYear={startYear}
                            />
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default FundsPerformanceTab;
