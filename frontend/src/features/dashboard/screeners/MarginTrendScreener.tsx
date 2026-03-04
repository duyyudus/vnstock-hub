import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
    Bar,
    CartesianGrid,
    ComposedChart,
    Legend,
    Line,
    LineChart,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from 'recharts';
import { stockApi } from '../../../api/stockApi';
import type { IndustryInfo, Stock } from '../../../api/stockApi';
import {
    buildMarginTrendScreen,
    type MarginTrendRow,
    type MarginTrendTickerBundle,
    type MarginTrendUniverseStats,
    type TrajectorySignal,
} from './marginTrendEngine';

export interface MarginTrendScreenerProps {
    benchmarkStocks: Stock[];
    displayStocks: Stock[];
    industries: IndustryInfo[];
    portfolioTickers?: string[];
    benchmarkLabel: string;
    sourceLoading: boolean;
    benchmarkLoading: boolean;
    sourceError: string | null;
    searchQuery: string;
}

interface FetchCacheEntry {
    bundle: MarginTrendTickerBundle;
    fetchedAt: number;
    hasErrors: boolean;
}

interface ScreenerProgress {
    total: number;
    loaded: number;
    failed: number;
    running: boolean;
}

type SortDirection = 'asc' | 'desc';
type SortKey =
    | 'ticker'
    | 'industry'
    | 'revenue'
    | 'net_margin'
    | 'roe'
    | 'revenue_yoy'
    | 'margin_change'
    | 'signal';

type TimeWindow = '1Y' | '2Y' | '3Y' | 'All';

interface SortConfig {
    key: SortKey;
    direction: SortDirection;
}

interface SortableHeader {
    key: SortKey;
    label: string;
}

interface SparklineProps {
    data: MarginTrendRow['series'];
    window: TimeWindow;
    metric: 'revenue' | 'net_margin' | 'roe';
    color: string;
}

const CONCURRENCY_LIMIT = 6;
const WINDOW_QUARTERS: Record<TimeWindow, number | null> = {
    '1Y': 4,
    '2Y': 8,
    '3Y': 12,
    'All': null,
};

const SORTABLE_HEADERS: SortableHeader[] = [
    { key: 'ticker', label: 'Ticker' },
    { key: 'industry', label: 'Industry' },
    { key: 'revenue', label: 'Revenue (Bn VND)' },
    { key: 'net_margin', label: 'Net Margin %' },
    { key: 'roe', label: 'ROE %' },
    { key: 'revenue_yoy', label: 'Revenue YoY %' },
    { key: 'margin_change', label: 'Margin Δ YoY (pp)' },
    { key: 'signal', label: 'Signal' },
];

const SORTABLE_HEADER_BY_KEY: Record<SortKey, SortableHeader> = SORTABLE_HEADERS.reduce((acc, header) => {
    acc[header.key] = header;
    return acc;
}, {} as Record<SortKey, SortableHeader>);

const buildErrorMessage = (error: unknown, fallback: string): string => {
    if (error && typeof error === 'object' && 'response' in error) {
        const response = (error as { response?: { data?: { detail?: string } } }).response;
        if (response?.data?.detail) {
            return response.data.detail;
        }
    }
    if (error instanceof Error) {
        return error.message;
    }
    return fallback;
};

const formatNumber = (value: number | null, digits = 2): string => {
    if (value === null || !Number.isFinite(value)) {
        return '-';
    }
    return new Intl.NumberFormat('en-US', {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
    }).format(value);
};

const formatPercent = (value: number | null, digits = 1): string => {
    if (value === null || !Number.isFinite(value)) {
        return '-';
    }
    return `${(value * 100).toFixed(digits)}%`;
};

const formatPercentPoint = (value: number | null, digits = 1): string => {
    if (value === null || !Number.isFinite(value)) {
        return '-';
    }
    const signed = value * 100;
    const prefix = signed > 0 ? '+' : '';
    return `${prefix}${signed.toFixed(digits)}pp`;
};

const signalBadgeClass = (signal: TrajectorySignal): string => {
    switch (signal) {
        case 'Strong':
            return 'badge-success';
        case 'Improving':
            return 'badge-info';
        case 'Neutral':
            return 'badge-ghost';
        case 'Compressing':
            return 'badge-warning';
        case 'Weak':
            return 'badge-error';
        default:
            return 'badge-ghost';
    }
};

const metricTextClass = (value: number | null): string => {
    if (value === null || !Number.isFinite(value)) {
        return 'text-base-content/50';
    }
    if (value > 0) {
        return 'text-success';
    }
    if (value < 0) {
        return 'text-error';
    }
    return 'text-base-content';
};

const sourceBadgeClass = (source: MarginTrendRow['benchmark']['source']): string => {
    switch (source) {
        case 'industry':
            return 'badge-success';
        case 'family':
            return 'badge-info';
        case 'market':
            return 'badge-warning';
        default:
            return 'badge-ghost';
    }
};

const sourceLabel = (source: MarginTrendRow['benchmark']['source']): string => {
    switch (source) {
        case 'industry':
            return 'Industry';
        case 'family':
            return 'Family';
        case 'market':
            return 'Market';
        default:
            return 'N/A';
    }
};

const signalDescription = (signal: TrajectorySignal): string => {
    switch (signal) {
        case 'Strong':
            return 'Strong: Net margin is expanding YoY and revenue YoY growth is above 10%.';
        case 'Improving':
            return 'Improving: Net margin is expanding YoY and revenue YoY is flat or positive.';
        case 'Neutral':
            return 'Neutral: Mixed/flat signals, or not enough data for a directional call.';
        case 'Compressing':
            return 'Compressing: Net margin is shrinking YoY while revenue YoY is still flat or positive.';
        case 'Weak':
            return 'Weak: Net margin is shrinking YoY and revenue YoY is declining.';
        default:
            return 'Signal based on margin YoY change and revenue YoY growth.';
    }
};

const formatTooltipMetric = (value: unknown, digits: number): string => {
    const numeric = typeof value === 'number'
        ? value
        : (typeof value === 'string' ? Number(value) : NaN);
    if (!Number.isFinite(numeric)) {
        return '-';
    }
    return numeric.toFixed(digits);
};

const runWithConcurrency = async <T,>(
    items: T[],
    limit: number,
    worker: (item: T) => Promise<void>,
): Promise<void> => {
    if (items.length === 0) {
        return;
    }
    let cursor = 0;
    const workers = Array.from({ length: Math.min(limit, items.length) }).map(async () => {
        while (cursor < items.length) {
            const index = cursor;
            cursor += 1;
            await worker(items[index]);
        }
    });
    await Promise.all(workers);
};

const sliceByWindow = (series: MarginTrendRow['series'], window: TimeWindow): MarginTrendRow['series'] => {
    const limit = WINDOW_QUARTERS[window];
    if (limit === null) {
        return series;
    }
    return series.slice(-limit);
};

const Sparkline: React.FC<SparklineProps> = ({ data, window, metric, color }) => {
    const sliced = useMemo(() => sliceByWindow(data, window), [data, window]);

    const chartData = useMemo(() => {
        return sliced
            .map((point) => {
                let value: number | null = null;
                if (metric === 'revenue') {
                    value = point.revenueBn;
                } else if (metric === 'net_margin') {
                    value = point.netMargin !== null ? point.netMargin * 100 : null;
                } else {
                    value = point.roe !== null ? point.roe * 100 : null;
                }
                return {
                    periodKey: point.periodKey,
                    value,
                };
            })
            .filter((item) => item.value !== null) as Array<{ periodKey: string; value: number }>;
    }, [metric, sliced]);

    if (chartData.length < 2) {
        return <span className="text-xs text-base-content/50">-</span>;
    }

    return (
        <div className="w-28 h-8">
            <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 2, right: 2, left: 2, bottom: 2 }}>
                    <Line
                        type="monotone"
                        dataKey="value"
                        stroke={color}
                        strokeWidth={1.8}
                        dot={false}
                        isAnimationActive={false}
                    />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
};

export const MarginTrendScreener: React.FC<MarginTrendScreenerProps> = ({
    benchmarkStocks,
    displayStocks,
    industries,
    portfolioTickers = [],
    benchmarkLabel,
    sourceLoading,
    benchmarkLoading,
    sourceError,
    searchQuery,
}) => {
    const [rows, setRows] = useState<MarginTrendRow[]>([]);
    const [stats, setStats] = useState<MarginTrendUniverseStats | null>(null);
    const [progress, setProgress] = useState<ScreenerProgress>({
        total: 0,
        loaded: 0,
        failed: 0,
        running: false,
    });
    const [sortConfig, setSortConfig] = useState<SortConfig>({
        key: 'signal',
        direction: 'desc',
    });
    const [timeWindow, setTimeWindow] = useState<TimeWindow>('2Y');
    const [selectedRow, setSelectedRow] = useState<MarginTrendRow | null>(null);

    const cacheRef = useRef<Map<string, FetchCacheEntry>>(new Map());
    const inflightRef = useRef<Map<string, Promise<FetchCacheEntry>>>(new Map());
    const runIdRef = useRef(0);
    const detailDialogRef = useRef<HTMLDialogElement>(null);

    const portfolioTickerSet = useMemo(() => {
        return new Set(portfolioTickers.map((ticker) => ticker.toUpperCase()));
    }, [portfolioTickers]);

    const fetchTickerBundle = useCallback(async (ticker: string): Promise<FetchCacheEntry> => {
        const normalizedTicker = ticker.toUpperCase();
        const cached = cacheRef.current.get(normalizedTicker);
        if (cached) {
            return cached;
        }

        const inflight = inflightRef.current.get(normalizedTicker);
        if (inflight) {
            return inflight;
        }

        const promise = (async () => {
            const [incomeResult, ratiosResult] = await Promise.allSettled([
                stockApi.getIncomeStatement(normalizedTicker),
                stockApi.getFinancialRatios(normalizedTicker),
            ]);

            const errors: string[] = [];

            const unwrapDataRows = (
                result: PromiseSettledResult<{ data: unknown[] }>,
                label: string,
            ): Record<string, unknown>[] => {
                if (result.status === 'fulfilled') {
                    return result.value.data
                        .filter((row): row is object => Boolean(row) && typeof row === 'object')
                        .map((row) => ({ ...(row as Record<string, unknown>) }));
                }
                errors.push(`${label}: ${buildErrorMessage(result.reason, 'request failed')}`);
                return [];
            };

            const bundle: MarginTrendTickerBundle = {
                income: unwrapDataRows(incomeResult, 'income'),
                ratios: unwrapDataRows(ratiosResult, 'ratios'),
                errors,
            };

            const entry: FetchCacheEntry = {
                bundle,
                fetchedAt: Date.now(),
                hasErrors: errors.length > 0,
            };

            cacheRef.current.set(normalizedTicker, entry);
            inflightRef.current.delete(normalizedTicker);
            return entry;
        })();

        inflightRef.current.set(normalizedTicker, promise);
        return promise;
    }, []);

    useEffect(() => {
        const benchmarkTickerSet = new Set(benchmarkStocks.map((stock) => stock.ticker.toUpperCase()));
        const displayTickerSet = new Set(displayStocks.map((stock) => stock.ticker.toUpperCase()));
        const requiredTickers = Array.from(new Set([...benchmarkTickerSet, ...displayTickerSet]));

        const activeRunId = runIdRef.current + 1;
        runIdRef.current = activeRunId;
        let cancelled = false;

        if (requiredTickers.length === 0) {
            const resetTimeout = window.setTimeout(() => {
                if (cancelled || runIdRef.current !== activeRunId) {
                    return;
                }
                setRows([]);
                setStats(null);
                setProgress({ total: 0, loaded: 0, failed: 0, running: false });
            }, 0);
            return () => {
                cancelled = true;
                window.clearTimeout(resetTimeout);
            };
        }

        const recompute = () => {
            if (cancelled || runIdRef.current !== activeRunId) {
                return;
            }

            const bundlesByTicker = new Map<string, MarginTrendTickerBundle>();
            const failedTickers = new Set<string>();

            requiredTickers.forEach((ticker) => {
                const cached = cacheRef.current.get(ticker);
                if (!cached) {
                    return;
                }
                bundlesByTicker.set(ticker, cached.bundle);
                if (cached.hasErrors) {
                    failedTickers.add(ticker);
                }
            });

            const computed = buildMarginTrendScreen({
                benchmarkStocks,
                displayStocks,
                industries,
                bundlesByTicker,
                failedTickers,
            });

            setRows(computed.rows);
            setStats(computed.stats);
            setProgress((prev) => ({
                ...prev,
                total: requiredTickers.length,
                loaded: bundlesByTicker.size,
                failed: failedTickers.size,
            }));
        };

        const run = async () => {
            setProgress({
                total: requiredTickers.length,
                loaded: requiredTickers.filter((ticker) => cacheRef.current.has(ticker)).length,
                failed: requiredTickers.filter((ticker) => cacheRef.current.get(ticker)?.hasErrors).length,
                running: true,
            });

            recompute();

            const missingTickers = requiredTickers.filter((ticker) => !cacheRef.current.has(ticker));
            await runWithConcurrency(missingTickers, CONCURRENCY_LIMIT, async (ticker) => {
                await fetchTickerBundle(ticker);
                recompute();
            });

            if (!cancelled && runIdRef.current === activeRunId) {
                setProgress((prev) => ({ ...prev, running: false }));
            }
        };

        run().catch((error) => {
            console.error('Failed to compute margin trend screener rows:', error);
            if (!cancelled && runIdRef.current === activeRunId) {
                setProgress((prev) => ({ ...prev, running: false }));
            }
        });

        return () => {
            cancelled = true;
        };
    }, [benchmarkStocks, displayStocks, fetchTickerBundle, industries]);

    const filteredRows = useMemo(() => {
        if (!searchQuery.trim()) {
            return rows;
        }
        const query = searchQuery.trim().toLowerCase();
        return rows.filter((row) => row.ticker.toLowerCase().includes(query));
    }, [rows, searchQuery]);

    const sortedRows = useMemo(() => {
        const direction = sortConfig.direction === 'asc' ? 1 : -1;
        const valueFor = (row: MarginTrendRow): string | number => {
            switch (sortConfig.key) {
                case 'ticker':
                    return row.ticker;
                case 'industry':
                    return row.industry;
                case 'revenue':
                    return row.latest_revenue_bn ?? -Infinity;
                case 'net_margin':
                    return row.latest_net_margin ?? -Infinity;
                case 'roe':
                    return row.latest_roe ?? -Infinity;
                case 'revenue_yoy':
                    return row.latest_revenue_yoy ?? -Infinity;
                case 'margin_change':
                    return row.margin_change_yoy ?? -Infinity;
                case 'signal':
                    return row.signal_strength;
                default:
                    return row.signal_strength;
            }
        };

        return [...filteredRows].sort((a, b) => {
            const va = valueFor(a);
            const vb = valueFor(b);
            if (typeof va === 'number' && typeof vb === 'number') {
                return (va - vb) * direction;
            }
            return String(va).localeCompare(String(vb)) * direction;
        });
    }, [filteredRows, sortConfig]);

    const toggleSort = (key: SortKey) => {
        setSortConfig((prev) => {
            if (prev.key !== key) {
                return { key, direction: 'desc' };
            }
            return { key, direction: prev.direction === 'desc' ? 'asc' : 'desc' };
        });
    };

    const openDetailModal = (row: MarginTrendRow) => {
        setSelectedRow(row);
        detailDialogRef.current?.showModal();
    };

    const closeDetailModal = () => {
        detailDialogRef.current?.close();
        setSelectedRow(null);
    };

    const detailSeries = useMemo(() => {
        if (!selectedRow) {
            return [] as MarginTrendRow['series'];
        }
        return sliceByWindow(selectedRow.series, timeWindow);
    }, [selectedRow, timeWindow]);

    const detailChartData = useMemo(() => {
        if (!selectedRow) {
            return [] as Array<Record<string, number | string | null>>;
        }
        const benchmarkMap = new Map(
            selectedRow.benchmark.series.map((point) => [point.periodKey, point]),
        );

        return detailSeries.map((point) => {
            const benchmarkPoint = benchmarkMap.get(point.periodKey);
            return {
                periodLabel: point.periodLabel,
                revenueBn: point.revenueBn,
                revenueYoyPct: point.revenueYoy !== null ? point.revenueYoy * 100 : null,
                revenueQoqPct: point.revenueQoq !== null ? point.revenueQoq * 100 : null,
                netMarginPct: point.netMargin !== null ? point.netMargin * 100 : null,
                roePct: point.roe !== null ? point.roe * 100 : null,
                benchmarkNetMarginPct: benchmarkPoint?.netMarginMedian !== null && benchmarkPoint?.netMarginMedian !== undefined
                    ? benchmarkPoint.netMarginMedian * 100
                    : null,
                benchmarkRoePct: benchmarkPoint?.roeMedian !== null && benchmarkPoint?.roeMedian !== undefined
                    ? benchmarkPoint.roeMedian * 100
                    : null,
            };
        });
    }, [detailSeries, selectedRow]);

    const headerButtonClass = (key: SortKey): string => {
        if (sortConfig.key !== key) {
            return 'text-base-content/60';
        }
        return 'text-primary';
    };

    const renderSortableHeader = (key: SortKey) => {
        const header = SORTABLE_HEADER_BY_KEY[key];
        const button = (
            <button
                className={`btn btn-ghost btn-xs h-auto min-h-0 px-1 py-0.5 normal-case ${headerButtonClass(key)}`}
                onClick={() => toggleSort(key)}
            >
                {header.label}
            </button>
        );
        if (key !== 'signal') {
            return button;
        }
        return (
            <div
                className="tooltip tooltip-bottom"
                data-tip="Signal is derived from net margin YoY change and revenue YoY growth."
            >
                {button}
            </div>
        );
    };

    return (
        <div className="space-y-4">
            <div className="card bg-base-100 shadow-md border border-base-300">
                <div className="card-body p-4 gap-4">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                        <h3 className="text-lg font-semibold">Margin Trend Screener</h3>
                        <div className="text-xs text-base-content/60 max-w-2xl">
                            Earnings and margin trajectory across quarters. Signal uses margin YoY change and revenue YoY growth.
                        </div>
                    </div>

                    <div className="flex flex-wrap items-center gap-2">
                        <span className={`badge ${progress.running ? 'badge-info' : 'badge-ghost'}`}>
                            {progress.running ? 'Computing...' : 'Ready'}
                        </span>
                        <span className="badge badge-outline">Benchmark: {benchmarkLabel}</span>
                        <span className="badge badge-outline">Universe: {stats?.benchmark_size ?? benchmarkStocks.length}</span>
                        <span className="badge badge-outline">Loaded: {progress.loaded}/{progress.total}</span>
                        <span className="badge badge-outline">Displayed: {stats?.display_size ?? displayStocks.length}</span>
                        <span className="badge badge-outline">Strong: {stats?.signal_counts.strong ?? 0}</span>
                        <span className="badge badge-outline">Weak: {stats?.signal_counts.weak ?? 0}</span>
                    </div>

                    <div className="flex flex-wrap items-center gap-2">
                        <span className="text-xs font-medium text-base-content/70">Period:</span>
                        <div className="join">
                            {(['1Y', '2Y', '3Y', 'All'] as TimeWindow[]).map((window) => (
                                <button
                                    key={window}
                                    className={`join-item btn btn-xs ${timeWindow === window ? 'btn-primary' : 'btn-outline'}`}
                                    onClick={() => setTimeWindow(window)}
                                >
                                    {window}
                                </button>
                            ))}
                        </div>
                    </div>

                    {sourceError ? (
                        <div className="alert alert-error">
                            <span>{sourceError}</span>
                        </div>
                    ) : null}

                    {progress.failed > 0 ? (
                        <div className="alert alert-warning">
                            <span>
                                {progress.failed} ticker(s) had partial endpoint failures. Signals are computed with best available data.
                            </span>
                        </div>
                    ) : null}

                    {(sourceLoading || benchmarkLoading) ? (
                        <div className="flex flex-col items-center justify-center h-24 gap-2">
                            <span className="loading loading-spinner loading-md text-primary"></span>
                            <p className="text-sm text-base-content/70">Loading source universe...</p>
                        </div>
                    ) : null}
                </div>
            </div>

            <div className="card bg-base-100 shadow-md border border-base-300">
                <div className="card-body p-0">
                    <div className="dashboard-adaptive-table-wrap border border-base-300 rounded-lg">
                        <table className="table table-xs table-pin-rows w-max min-w-full">
                            <thead>
                                <tr>
                                    {SORTABLE_HEADERS.map((header) => (
                                        <th key={header.key} className="align-top">
                                            {renderSortableHeader(header.key)}
                                        </th>
                                    ))}
                                    <th className="text-base-content/60 text-xs">Revenue Trend</th>
                                    <th className="text-base-content/60 text-xs">Margin Trend</th>
                                    <th className="text-base-content/60 text-xs">ROE Trend</th>
                                </tr>
                            </thead>
                            <tbody>
                                {sortedRows.length === 0 ? (
                                    <tr>
                                        <td colSpan={SORTABLE_HEADERS.length + 3} className="text-center text-base-content/60 py-6">
                                            {progress.running ? 'Computing margin trend rows...' : 'No stocks match current filters.'}
                                        </td>
                                    </tr>
                                ) : sortedRows.map((row) => {
                                    const isInPortfolio = portfolioTickerSet.has(row.ticker.toUpperCase());
                                    return (
                                        <tr
                                            key={row.ticker}
                                            className="hover cursor-pointer"
                                            onClick={() => openDetailModal(row)}
                                            title={`Open ${row.ticker} detail view`}
                                        >
                                            <td className={`font-semibold whitespace-nowrap ${isInPortfolio ? 'text-accent' : ''}`}>{row.ticker}</td>
                                            <td className="max-w-52 truncate" title={row.industry}>{row.industry}</td>
                                            <td className="whitespace-nowrap">{formatNumber(row.latest_revenue_bn, 1)}</td>
                                            <td className="whitespace-nowrap">{formatPercent(row.latest_net_margin, 1)}</td>
                                            <td className="whitespace-nowrap">{formatPercent(row.latest_roe, 1)}</td>
                                            <td className={`whitespace-nowrap ${metricTextClass(row.latest_revenue_yoy)}`}>
                                                {formatPercent(row.latest_revenue_yoy, 1)}
                                            </td>
                                            <td className={`whitespace-nowrap ${metricTextClass(row.margin_change_yoy)}`}>
                                                {formatPercentPoint(row.margin_change_yoy, 1)}
                                            </td>
                                            <td className="whitespace-nowrap">
                                                <div
                                                    className="tooltip tooltip-left"
                                                    data-tip={signalDescription(row.signal)}
                                                >
                                                    <span className={`badge badge-sm ${signalBadgeClass(row.signal)}`}>
                                                        {row.signal}
                                                    </span>
                                                </div>
                                            </td>
                                            <td>
                                                <Sparkline data={row.series} window={timeWindow} metric="revenue" color="#0284c7" />
                                            </td>
                                            <td>
                                                <Sparkline data={row.series} window={timeWindow} metric="net_margin" color="#16a34a" />
                                            </td>
                                            <td>
                                                <Sparkline data={row.series} window={timeWindow} metric="roe" color="#f97316" />
                                            </td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            <dialog
                ref={detailDialogRef}
                className="modal"
                onClose={() => setSelectedRow(null)}
            >
                <div className="modal-box max-w-6xl">
                    {selectedRow ? (
                        <>
                            <div className="flex flex-wrap items-start justify-between gap-3">
                                <div>
                                    <h3 className="font-bold text-xl">{selectedRow.ticker} Margin & Earnings Trend</h3>
                                    <p className="text-sm text-base-content/70 mt-1">
                                        {selectedRow.company_name || selectedRow.ticker} · {selectedRow.industry}
                                    </p>
                                </div>
                                <div className="flex flex-wrap items-center gap-2">
                                    <span className={`badge ${signalBadgeClass(selectedRow.signal)}`}>
                                        <span
                                            className="tooltip tooltip-left"
                                            data-tip={signalDescription(selectedRow.signal)}
                                        >
                                            Signal: {selectedRow.signal}
                                        </span>
                                    </span>
                                    <span className={`badge ${sourceBadgeClass(selectedRow.benchmark.source)}`}>
                                        Benchmark: {sourceLabel(selectedRow.benchmark.source)} ({selectedRow.benchmark.peer_count})
                                    </span>
                                    <span className="badge badge-outline">
                                        Latest: {selectedRow.latest_quarter ?? 'N/A'}
                                    </span>
                                </div>
                            </div>

                            <div className="mt-4 mb-2 flex items-center gap-2">
                                <span className="text-xs font-medium text-base-content/70">Range:</span>
                                <div className="join">
                                    {(['1Y', '2Y', '3Y', 'All'] as TimeWindow[]).map((window) => (
                                        <button
                                            key={`detail-${window}`}
                                            className={`join-item btn btn-xs ${timeWindow === window ? 'btn-primary' : 'btn-outline'}`}
                                            onClick={() => setTimeWindow(window)}
                                        >
                                            {window}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            {detailChartData.length === 0 ? (
                                <div className="h-40 flex items-center justify-center text-base-content/60">
                                    Not enough quarterly data to render charts.
                                </div>
                            ) : (
                                <div className="space-y-6 mt-4">
                                    <div>
                                        <h4 className="font-semibold text-sm mb-2">Revenue (Bn VND) with YoY Growth Overlay</h4>
                                        <div className="w-full h-72">
                                            <ResponsiveContainer width="100%" height="100%">
                                                <ComposedChart data={detailChartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                                                    <CartesianGrid strokeDasharray="3 3" />
                                                    <XAxis dataKey="periodLabel" />
                                                    <YAxis yAxisId="left" tickFormatter={(value) => `${value}`} />
                                                    <YAxis yAxisId="right" orientation="right" tickFormatter={(value) => `${value}%`} />
                                                    <Tooltip
                                                        formatter={(value: unknown, name: string | undefined) => {
                                                            const seriesName = name ?? '';
                                                            if (value === null || value === undefined) {
                                                                return ['-', seriesName];
                                                            }
                                                            if (seriesName === 'Revenue (Bn VND)') {
                                                                const numeric = typeof value === 'number'
                                                                    ? value
                                                                    : Number(value);
                                                                return [formatNumber(Number.isFinite(numeric) ? numeric : null, 1), seriesName];
                                                            }
                                                            return [`${formatTooltipMetric(value, 1)}%`, seriesName];
                                                        }}
                                                    />
                                                    <Legend />
                                                    <Bar yAxisId="left" dataKey="revenueBn" name="Revenue (Bn VND)" fill="#0284c7" />
                                                    <Line
                                                        yAxisId="right"
                                                        type="monotone"
                                                        dataKey="revenueYoyPct"
                                                        name="Revenue YoY %"
                                                        stroke="#ef4444"
                                                        strokeWidth={2}
                                                        dot={false}
                                                    />
                                                </ComposedChart>
                                            </ResponsiveContainer>
                                        </div>
                                    </div>

                                    <div>
                                        <h4 className="font-semibold text-sm mb-2">Net Margin and ROE vs Benchmark Median</h4>
                                        <div className="w-full h-72">
                                            <ResponsiveContainer width="100%" height="100%">
                                                <ComposedChart data={detailChartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                                                    <CartesianGrid strokeDasharray="3 3" />
                                                    <XAxis dataKey="periodLabel" />
                                                    <YAxis tickFormatter={(value) => `${value}%`} />
                                                    <Tooltip
                                                        formatter={(value: unknown, name: string | undefined) => {
                                                            const seriesName = name ?? '';
                                                            if (value === null || value === undefined) {
                                                                return ['-', seriesName];
                                                            }
                                                            return [`${formatTooltipMetric(value, 1)}%`, seriesName];
                                                        }}
                                                    />
                                                    <Legend />
                                                    <Line
                                                        type="monotone"
                                                        dataKey="netMarginPct"
                                                        name="Net Margin %"
                                                        stroke="#16a34a"
                                                        strokeWidth={2}
                                                        dot={false}
                                                    />
                                                    <Line
                                                        type="monotone"
                                                        dataKey="roePct"
                                                        name="ROE %"
                                                        stroke="#f97316"
                                                        strokeWidth={2}
                                                        dot={false}
                                                    />
                                                    <Line
                                                        type="monotone"
                                                        dataKey="benchmarkNetMarginPct"
                                                        name="Benchmark Net Margin %"
                                                        stroke="#16a34a"
                                                        strokeDasharray="5 4"
                                                        strokeWidth={1.6}
                                                        dot={false}
                                                    />
                                                    <Line
                                                        type="monotone"
                                                        dataKey="benchmarkRoePct"
                                                        name="Benchmark ROE %"
                                                        stroke="#f97316"
                                                        strokeDasharray="5 4"
                                                        strokeWidth={1.6}
                                                        dot={false}
                                                    />
                                                </ComposedChart>
                                            </ResponsiveContainer>
                                        </div>
                                    </div>
                                </div>
                            )}

                            <div className="modal-action">
                                <button type="button" className="btn btn-ghost" onClick={closeDetailModal}>Close</button>
                            </div>
                        </>
                    ) : (
                        <div className="py-8 text-center text-base-content/60">No stock selected.</div>
                    )}
                </div>
                <form method="dialog" className="modal-backdrop">
                    <button onClick={closeDetailModal}>close</button>
                </form>
            </dialog>
        </div>
    );
};

export default MarginTrendScreener;
