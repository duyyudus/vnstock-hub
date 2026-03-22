import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
    CartesianGrid,
    Legend,
    Line,
    LineChart,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from 'recharts';
import { stockApi } from '../../../api/stockApi';
import type { Stock, StocksVolumeSeriesResponse } from '../../../api/stockApi';
import type { DateRange } from './dateRange';

interface StocksVolumeChartProps {
    stocks: Stock[];
    dateRange: DateRange;
}

type TopNOption = 10 | 15 | 20 | 30 | 'all';
type ActiveFilter = 'topn' | 'manual';

interface RankedSeries {
    symbol: string;
    ticker: string;
    company_name: string;
    latestValue: number;
    data: Array<{ date: string; value: number | null }>;
}

interface ChartRow {
    date: string;
    [key: string]: string | number | null;
}

interface TooltipEntry {
    dataKey?: string | number;
    value?: string | number | null;
    color?: string;
}

interface CustomTooltipProps {
    active?: boolean;
    payload?: TooltipEntry[];
    label?: string;
    symbolToCompany: Map<string, string>;
}

const TOP_N_OPTIONS: TopNOption[] = [10, 15, 20, 30, 'all'];
const TOP_COLORS = [
    '#3b82f6',
    '#10b981',
    '#f59e0b',
    '#ef4444',
    '#8b5cf6',
    '#ec4899',
    '#06b6d4',
    '#84cc16',
    '#f97316',
    '#6366f1',
];

const formatDateShort = (dateStr: string): string => {
    const date = new Date(`${dateStr}T00:00:00`);
    if (Number.isNaN(date.getTime())) {
        return dateStr;
    }
    return `${date.getMonth() + 1}/${date.getDate()}`;
};

const formatBilVnd = (value: number): string => {
    if (value >= 1000) return `${(value / 1000).toFixed(1)}K`;
    if (value >= 100) return value.toFixed(0);
    if (value >= 10) return value.toFixed(1);
    return value.toFixed(2);
};

const formatTopNLabel = (option: TopNOption): string => {
    if (option === 'all') {
        return 'All';
    }
    return `Top ${option}`;
};

const CustomTooltip = ({ active, payload, label, symbolToCompany }: CustomTooltipProps) => {
    if (!active || !payload || payload.length === 0) {
        return null;
    }

    const sortedPayload = [...payload]
        .filter((entry) => entry.value !== null && entry.value !== undefined)
        .sort((a, b) => Number(b.value) - Number(a.value));

    return (
        <div className="bg-base-100 border border-base-300 p-3 rounded-lg shadow-lg max-h-64 overflow-y-auto">
            <p className="text-sm font-semibold mb-2">{label || '-'}</p>
            {sortedPayload.map((entry, index) => {
                const symbol = String(entry.dataKey ?? '');
                const companyName = symbolToCompany.get(symbol) || symbol;
                const value = Number(entry.value);
                return (
                    <p key={`${symbol}-${index}`} className="text-xs" style={{ color: entry.color || 'currentColor' }}>
                        {symbol} ({companyName}): {Number.isFinite(value) ? `${formatBilVnd(value)} Bil VND` : 'N/A'}
                    </p>
                );
            })}
        </div>
    );
};

export const StocksVolumeChart: React.FC<StocksVolumeChartProps> = ({ stocks, dateRange }) => {
    const requestIdRef = useRef(0);
    const stockPickerRef = useRef<HTMLDivElement | null>(null);
    const [topN, setTopN] = useState<TopNOption>(15);
    const [activeFilter, setActiveFilter] = useState<ActiveFilter>('topn');
    const [selectedSymbols, setSelectedSymbols] = useState<string[]>([]);
    const [isStockPickerOpen, setIsStockPickerOpen] = useState(false);

    const [volumeResponse, setVolumeResponse] = useState<StocksVolumeSeriesResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const symbols = useMemo(() => {
        return Array.from(new Set(stocks.map((stock) => stock.ticker.toUpperCase())));
    }, [stocks]);

    useEffect(() => {
        if (symbols.length === 0) {
            return;
        }

        const requestId = requestIdRef.current + 1;
        requestIdRef.current = requestId;
        window.setTimeout(() => {
            if (requestId !== requestIdRef.current) {
                return;
            }
            setLoading(true);
            setError(null);
        }, 0);

        stockApi.getStocksVolumeSeries(symbols, dateRange.startDate, dateRange.endDate)
            .then((response) => {
                if (requestId !== requestIdRef.current) {
                    return;
                }
                setVolumeResponse(response);
            })
            .catch((err) => {
                if (requestId !== requestIdRef.current) {
                    return;
                }
                console.error('Error fetching stocks volume series:', err);
                setError('Failed to load volume series. Please try again.');
            })
            .finally(() => {
                if (requestId !== requestIdRef.current) {
                    return;
                }
                setLoading(false);
            });
    }, [dateRange.endDate, dateRange.startDate, symbols]);

    useEffect(() => {
        const handlePointerDown = (event: MouseEvent | TouchEvent) => {
            const picker = stockPickerRef.current;
            if (!picker) return;
            const target = event.target as Node | null;
            if (!target) return;
            if (!picker.contains(target)) {
                setIsStockPickerOpen(false);
            }
        };

        document.addEventListener('mousedown', handlePointerDown);
        document.addEventListener('touchstart', handlePointerDown);
        return () => {
            document.removeEventListener('mousedown', handlePointerDown);
            document.removeEventListener('touchstart', handlePointerDown);
        };
    }, []);

    const symbolToCompany = useMemo(() => {
        const mapping = new Map<string, string>();
        if (!volumeResponse) {
            return mapping;
        }

        volumeResponse.stocks.forEach((stockSeries) => {
            mapping.set(stockSeries.symbol, stockSeries.company_name);
        });

        return mapping;
    }, [volumeResponse]);

    const rankedSeries = useMemo<RankedSeries[]>(() => {
        if (!volumeResponse) {
            return [];
        }

        return volumeResponse.stocks
            .map((stockSeries) => {
                let latestValue: number | null = null;
                for (let i = stockSeries.data.length - 1; i >= 0; i -= 1) {
                    const point = stockSeries.data[i];
                    if (point.value !== null && Number.isFinite(point.value)) {
                        latestValue = point.value;
                        break;
                    }
                }

                return {
                    symbol: stockSeries.symbol,
                    ticker: stockSeries.ticker,
                    company_name: stockSeries.company_name,
                    latestValue,
                    data: stockSeries.data,
                };
            })
            .filter((stockSeries): stockSeries is RankedSeries => stockSeries.latestValue !== null)
            .sort((a, b) => b.latestValue - a.latestValue);
    }, [volumeResponse]);

    const selectedSymbolsSet = useMemo(() => new Set(selectedSymbols), [selectedSymbols]);
    const selectedActiveCount = useMemo(
        () => rankedSeries.reduce((count, series) => (selectedSymbolsSet.has(series.symbol) ? count + 1 : count), 0),
        [rankedSeries, selectedSymbolsSet]
    );
    const pickerSeries = useMemo(
        () => [...rankedSeries].sort((a, b) => a.symbol.localeCompare(b.symbol)),
        [rankedSeries]
    );

    const visibleSeries = useMemo(() => {
        if (activeFilter === 'manual') {
            return rankedSeries.filter((stockSeries) => selectedSymbolsSet.has(stockSeries.symbol));
        }
        if (topN === 'all') {
            return rankedSeries;
        }
        return rankedSeries.slice(0, topN);
    }, [activeFilter, rankedSeries, selectedSymbolsSet, topN]);

    const chartData = useMemo<ChartRow[]>(() => {
        if (visibleSeries.length === 0) {
            return [];
        }

        const dateMap = new Map<string, ChartRow>();
        visibleSeries.forEach((stockSeries) => {
            stockSeries.data.forEach((point) => {
                if (!dateMap.has(point.date)) {
                    dateMap.set(point.date, { date: point.date });
                }
                const row = dateMap.get(point.date);
                if (!row) {
                    return;
                }
                row[stockSeries.symbol] = point.value;
            });
        });

        return Array.from(dateMap.values()).sort((a, b) => String(a.date).localeCompare(String(b.date)));
    }, [visibleSeries]);

    const handleTopNChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
        const value = event.target.value;
        if (value === 'all') {
            setTopN('all');
        } else {
            setTopN(Number(value) as TopNOption);
        }
        setSelectedSymbols([]);
        setActiveFilter('topn');
        setIsStockPickerOpen(false);
    };

    const handleToggleSymbol = (symbol: string) => {
        const next = selectedSymbols.includes(symbol)
            ? selectedSymbols.filter((item) => item !== symbol)
            : [...selectedSymbols, symbol];
        setSelectedSymbols(next);
        if (next.length === 0) {
            setActiveFilter('topn');
        } else {
            setActiveFilter('manual');
        }
    };

    const handleSelectAllSymbols = () => {
        setSelectedSymbols(pickerSeries.map((item) => item.symbol));
        setActiveFilter('manual');
    };

    const handleClearSymbols = () => {
        setSelectedSymbols([]);
        setActiveFilter('topn');
    };

    if (symbols.length === 0) {
        return (
            <div className="flex items-center justify-center h-64 text-base-content/50">
                No stocks available for volume comparison.
            </div>
        );
    }

    return (
        <div className="w-full h-full flex flex-col space-y-4">
            <div className="flex flex-col gap-3 border-b border-base-300 pb-3">
                <div className="flex flex-wrap items-center gap-3">
                    <label className="flex items-center gap-2 text-sm">
                        Show
                        <select
                            className="select select-sm select-bordered"
                            value={String(topN)}
                            onChange={handleTopNChange}
                        >
                            {TOP_N_OPTIONS.map((option) => (
                                <option key={String(option)} value={String(option)}>
                                    {formatTopNLabel(option)}
                                </option>
                            ))}
                        </select>
                    </label>

                    <div ref={stockPickerRef} className="relative">
                        <button
                            type="button"
                            className="btn btn-sm btn-outline"
                            onClick={() => setIsStockPickerOpen((prev) => !prev)}
                        >
                            Stocks ({selectedActiveCount})
                        </button>
                        {isStockPickerOpen ? (
                            <div className="absolute left-0 z-20 mt-2 w-72 rounded-box border border-base-300 bg-base-100 p-3 shadow-lg">
                                <div className="mb-2 flex items-center justify-between gap-2">
                                    <span className="text-xs text-base-content/70">Specific stocks</span>
                                    <div className="flex items-center gap-1">
                                        <button
                                            type="button"
                                            className="btn btn-ghost btn-xs"
                                            onClick={handleSelectAllSymbols}
                                        >
                                            All
                                        </button>
                                        <button
                                            type="button"
                                            className="btn btn-ghost btn-xs"
                                            onClick={handleClearSymbols}
                                        >
                                            Clear
                                        </button>
                                    </div>
                                </div>
                                <div className="max-h-56 overflow-y-auto rounded border border-base-300">
                                {pickerSeries.length === 0 ? (
                                    <div className="p-2 text-xs text-base-content/60">No stocks available</div>
                                ) : (
                                    pickerSeries.map((series) => (
                                        <label
                                            key={series.symbol}
                                            className="flex cursor-pointer items-center gap-2 px-2 py-1.5 text-sm hover:bg-base-200/60"
                                        >
                                                <input
                                                    type="checkbox"
                                                    className="checkbox checkbox-xs"
                                                    checked={selectedSymbolsSet.has(series.symbol)}
                                                    onChange={() => handleToggleSymbol(series.symbol)}
                                                />
                                                <span className="font-semibold">{series.symbol}</span>
                                                <span className="truncate text-xs text-base-content/60">
                                                    {series.company_name}
                                                </span>
                                            </label>
                                        ))
                                    )}
                                </div>
                                <div className="mt-2 text-[11px] text-base-content/60">
                                    Selecting stocks switches filter mode to manual.
                                </div>
                            </div>
                        ) : null}
                    </div>

                    {volumeResponse?.is_syncing ? (
                        <div className="ml-auto flex items-center gap-1 text-xs text-warning">
                            <span className="loading loading-spinner loading-xs"></span>
                            Syncing...
                        </div>
                    ) : null}
                </div>

                <div className="text-xs text-base-content/60">
                    Range: {dateRange.startDate} to {dateRange.endDate} | {visibleSeries.length} of {rankedSeries.length} stocks shown | Mode: {activeFilter === 'topn' ? `Show ${formatTopNLabel(topN)}` : 'Specific stocks'}
                </div>
            </div>

            {volumeResponse?.is_stale && !volumeResponse.is_syncing ? (
                <div className="alert alert-info py-2">
                    <span className="text-xs">Showing cached data. Fresh data sync is pending.</span>
                </div>
            ) : null}

            {loading ? (
                <div className="flex flex-col items-center justify-center h-96 gap-4">
                    <span className="loading loading-spinner loading-lg text-primary"></span>
                    <p className="text-base-content/70">Loading volume series...</p>
                </div>
            ) : error ? (
                <div className="flex flex-col items-center justify-center h-96 gap-4">
                    <div className="alert alert-error max-w-md">
                        <span>{error}</span>
                    </div>
                </div>
            ) : chartData.length === 0 ? (
                <div className="flex items-center justify-center h-96 text-base-content/50">
                    No volume value data available for the selected range.
                </div>
            ) : (
                <div className="flex-1 min-h-0">
                    <ResponsiveContainer width="100%" height={560} debounce={50}>
                        <LineChart
                            data={chartData}
                            margin={{ top: 10, right: 30, left: 10, bottom: 30 }}
                        >
                            <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} />
                            <XAxis
                                dataKey="date"
                                tickFormatter={formatDateShort}
                                tick={{ fontSize: 11 }}
                                stroke="currentColor"
                                opacity={0.5}
                            />
                            <YAxis
                                tickFormatter={formatBilVnd}
                                tick={{ fontSize: 11 }}
                                stroke="currentColor"
                                opacity={0.5}
                                label={{ value: 'Value (Bil VND)', angle: -90, position: 'insideLeft', style: { fontSize: 11 } }}
                            />
                            <Tooltip
                                content={<CustomTooltip symbolToCompany={symbolToCompany} />}
                                isAnimationActive={false}
                            />
                            <Legend verticalAlign="top" height={30} wrapperStyle={{ fontSize: 11 }} />

                            {visibleSeries.map((stockSeries, index) => (
                                <Line
                                    key={stockSeries.symbol}
                                    type="monotone"
                                    dataKey={stockSeries.symbol}
                                    stroke={TOP_COLORS[index % TOP_COLORS.length]}
                                    strokeWidth={index < 5 ? 2 : 1.2}
                                    dot={false}
                                    connectNulls={true}
                                    name={stockSeries.symbol}
                                />
                            ))}
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            )}
        </div>
    );
};

export default StocksVolumeChart;
