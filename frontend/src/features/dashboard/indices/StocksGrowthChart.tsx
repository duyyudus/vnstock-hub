import React, { useEffect, useState, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { stockApi } from '../../../api/stockApi';
import type { Stock, StocksWeeklyPricesResponse } from '../../../api/stockApi';

interface StocksGrowthChartProps {
    stocks: Stock[];
}

type Benchmark = 'VNINDEX' | 'VN30';

// Color palette for top stocks
const TOP_COLORS = [
    '#3b82f6', // blue
    '#10b981', // green
    '#f59e0b', // amber
    '#ef4444', // red
    '#8b5cf6', // purple
    '#ec4899', // pink
    '#06b6d4', // cyan
    '#84cc16', // lime
    '#f97316', // orange
    '#6366f1', // indigo
];

const GRAY_COLOR = '#6b7280';
const VNINDEX_COLOR = '#fbbf24'; // amber
const VN30_COLOR = '#22d3d3'; // cyan

const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
        // Sort payload by value descending
        const sorted = [...payload].sort((a, b) => (b.value || 0) - (a.value || 0));
        return (
            <div className="bg-base-100 border border-base-300 p-3 rounded-lg shadow-lg max-h-64 overflow-y-auto">
                <p className="text-sm font-semibold mb-2">{label}</p>
                {sorted.slice(0, 10).map((entry: any, index: number) => {
                    const isBenchmark = entry.dataKey === 'VNINDEX' || entry.dataKey === 'VN30';
                    return (
                        <p key={index} className="text-xs" style={{ color: entry.color }}>
                            {isBenchmark ? '📊 ' : ''}{entry.dataKey}: {entry.value?.toFixed(1) || 'N/A'}
                        </p>
                    );
                })}
                {sorted.length > 10 && (
                    <p className="text-xs text-base-content/50">...and {sorted.length - 10} more</p>
                )}
            </div>
        );
    }
    return null;
};

export const StocksGrowthChart: React.FC<StocksGrowthChartProps> = ({
    stocks,
}) => {
    // Local State
    const [startYear, setStartYear] = useState<number>(new Date().getFullYear() - 3);
    const [benchmark, setBenchmark] = useState<Benchmark>('VNINDEX');

    // Data State
    const [priceData, setPriceData] = useState<StocksWeeklyPricesResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [isSyncing, setIsSyncing] = useState(false);

    // Extract symbols from stocks
    const symbols = useMemo(() => stocks.map(s => s.ticker), [stocks]);

    // Generate year options (last 10 years)
    const yearOptions = useMemo(() => {
        const currentYear = new Date().getFullYear();
        return Array.from({ length: 10 }, (_, i) => currentYear - i);
    }, []);

    // Fetch price data when stocks or startYear change
    useEffect(() => {
        const fetchData = async () => {
            if (symbols.length === 0) {
                setPriceData(null);
                return;
            }

            setLoading(true);
            setError(null);

            try {
                // Always request benchmarks
                const response = await stockApi.getStocksWeeklyPrices(
                    symbols,
                    startYear,
                    true
                );
                setPriceData(response);
                setIsSyncing(response.is_syncing || response.is_stale || false);
            } catch (err) {
                setError('Failed to load price data. Please try again.');
                console.error('Error fetching weekly prices:', err);
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, [symbols, startYear]);

    // Poll for fresh data when syncing
    useEffect(() => {
        if (!isSyncing) return;

        const pollForFreshData = async () => {
            try {
                const response = await stockApi.getStocksWeeklyPrices(
                    symbols,
                    startYear,
                    true
                );
                if (!response.is_syncing && !response.is_stale) {
                    setPriceData(response);
                    setIsSyncing(false);
                }
            } catch (err) {
                console.error('Error polling for fresh stock performance data:', err);
            }
        };

        const interval = setInterval(pollForFreshData, 5000);
        return () => clearInterval(interval);
    }, [isSyncing, symbols, startYear]);

    // Process data for chart - normalize to base 100
    const chartData = useMemo(() => {
        if (!priceData || !priceData.stocks.length) return [];

        // Get all unique dates
        const dateMap = new Map<string, Record<string, number>>();

        // Add stock data
        priceData.stocks.forEach((stock) => {
            const prices = stock.prices;
            if (prices.length > 0) {
                // Filter to ensure we respect startYear (robustness against backend extra points)
                const startOfSelectedYear = `${startYear}-01-01`;
                const filteredPrices = prices.filter(p => p.date >= startOfSelectedYear);

                if (filteredPrices.length > 0) {
                    // Normalize to base 100 at start of FILTERED range
                    const basePrice = filteredPrices[0].close;
                    filteredPrices.forEach((point) => {
                        if (!dateMap.has(point.date)) {
                            dateMap.set(point.date, { date: point.date as unknown as number });
                        }
                        const record = dateMap.get(point.date)!;
                        record[stock.symbol] = basePrice > 0 ? (point.close / basePrice) * 100 : 100;
                    });
                }
            }
        });

        // Add benchmark data (only the selected one)
        const benchmarkKey = benchmark;
        if (priceData.benchmarks && priceData.benchmarks[benchmarkKey]) {
            const prices = priceData.benchmarks[benchmarkKey];
            if (prices && prices.length > 0) {
                const startOfSelectedYear = `${startYear}-01-01`;
                const filteredPrices = prices.filter(p => p.date >= startOfSelectedYear);

                if (filteredPrices.length > 0) {
                    const basePrice = filteredPrices[0].close;
                    filteredPrices.forEach((point) => {
                        if (!dateMap.has(point.date)) {
                            dateMap.set(point.date, { date: point.date as unknown as number });
                        }
                        const record = dateMap.get(point.date)!;
                        record[benchmarkKey] = basePrice > 0 ? (point.close / basePrice) * 100 : 100;
                    });
                }
            }
        }

        // Sort by date
        return Array.from(dateMap.values()).sort((a, b) =>
            String(a.date).localeCompare(String(b.date))
        );
    }, [priceData, benchmark, startYear]);

    // Sort stocks by final performance to determine colors
    const sortedStocks = useMemo(() => {
        if (!priceData || chartData.length === 0) return [];

        const lastRecord = chartData[chartData.length - 1];
        return [...priceData.stocks]
            .filter(s => s.prices.length > 0)
            .sort((a, b) => {
                const aVal = lastRecord[a.symbol] ?? 0;
                const bVal = lastRecord[b.symbol] ?? 0;
                return bVal - aVal;
            });
    }, [priceData, chartData]);

    const formatDate = (dateStr: string) => {
        const date = new Date(dateStr);
        return `${date.getMonth() + 1}/${date.getFullYear().toString().slice(2)}`;
    };

    const formatValue = (value: number) => {
        return value.toFixed(0);
    };

    return (
        <div className="w-full h-full flex flex-col space-y-4">
            {/* Controls Bar */}
            <div className="flex flex-wrap items-center gap-4 border-b border-base-300 pb-2">
                <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-base-content/70">Start:</span>
                    <select
                        className="select select-sm select-bordered"
                        value={startYear}
                        onChange={(e) => setStartYear(parseInt(e.target.value))}
                    >
                        {yearOptions.map(year => (
                            <option key={year} value={year}>From {year}</option>
                        ))}
                    </select>
                </div>

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

                {isSyncing && (
                    <div className="ml-auto flex items-center gap-1 text-xs text-warning">
                        <span className="loading loading-spinner loading-xs"></span>
                        Syncing...
                    </div>
                )}
            </div>

            {loading ? (
                <div className="flex flex-col items-center justify-center h-96 gap-4">
                    <span className="loading loading-spinner loading-lg text-primary"></span>
                    <p className="text-base-content/70">Loading price data...</p>
                </div>
            ) : error ? (
                <div className="flex flex-col items-center justify-center h-96 gap-4">
                    <div className="alert alert-error max-w-md">
                        <svg xmlns="http://www.w3.org/2000/svg" className="stroke-current shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <span>{error}</span>
                    </div>
                </div>
            ) : chartData.length === 0 ? (
                <div className="flex items-center justify-center h-96 text-base-content/50">
                    No price data available for the selected timeframe
                </div>
            ) : (
                <div className="flex-1 min-h-0 relative">
                    <ResponsiveContainer width="100%" height={500} debounce={50}>
                        <LineChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 30 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} />
                            <XAxis
                                dataKey="date"
                                tickFormatter={formatDate}
                                tick={{ fontSize: 11 }}
                                stroke="currentColor"
                                opacity={0.5}
                                interval="preserveStartEnd"
                            />
                            <YAxis
                                tickFormatter={formatValue}
                                tick={{ fontSize: 11 }}
                                stroke="currentColor"
                                opacity={0.5}
                                domain={['auto', 'auto']}
                                label={{ value: 'Growth (Base=100)', angle: -90, position: 'insideLeft', style: { fontSize: 11 } }}
                            />
                            <Tooltip content={<CustomTooltip />} isAnimationActive={false} />

                            {/* Render stocks - top 10 get colors, rest are gray */}
                            {sortedStocks.map((stock, idx) => (
                                <Line
                                    key={stock.symbol}
                                    type="monotone"
                                    dataKey={stock.symbol}
                                    stroke={idx < 10 ? TOP_COLORS[idx] : GRAY_COLOR}
                                    strokeWidth={idx < 3 ? 2.5 : idx < 10 ? 1.5 : 0.8}
                                    strokeOpacity={idx < 10 ? 1 : 0.3}
                                    dot={false}
                                    name={stock.company_name || stock.symbol}
                                    connectNulls={true}
                                />
                            ))}

                            {/* Benchmark Line (Only one) */}
                            <Line
                                type="monotone"
                                dataKey={benchmark}
                                stroke={benchmark === 'VNINDEX' ? VNINDEX_COLOR : VN30_COLOR}
                                strokeWidth={3}
                                strokeDasharray="5 5"
                                dot={false}
                                name={benchmark === 'VNINDEX' ? "VN-Index" : "VN30"}
                                connectNulls={true}
                            />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            )}

            {/* Legend for top stocks and benchmarks */}
            <div className="flex flex-wrap justify-center gap-3 mt-4 mb-2 text-xs">
                {sortedStocks.slice(0, 5).map((stock, idx) => (
                    <div key={stock.symbol} className="flex items-center gap-1">
                        <div className="w-3 h-3 rounded" style={{ backgroundColor: TOP_COLORS[idx] }}></div>
                        <span>{stock.symbol}</span>
                    </div>
                ))}

                {/* Benchmark Legend */}
                {priceData?.benchmarks && priceData.benchmarks[benchmark] && (
                    <div className="flex items-center gap-1">
                        <div className="w-4 h-0.5 border-t-2 border-dashed"
                            style={{ borderColor: benchmark === 'VNINDEX' ? VNINDEX_COLOR : VN30_COLOR }}>
                        </div>
                        <span>{benchmark === 'VNINDEX' ? "VN-Index" : "VN30"}</span>
                    </div>
                )}
            </div>
        </div>
    );
};

export default StocksGrowthChart;
