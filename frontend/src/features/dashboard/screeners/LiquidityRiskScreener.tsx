import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { stockApi } from '../../../api/stockApi';
import type { OhlcvDataPoint, Stock } from '../../../api/stockApi';
import {
    buildLiquidityRiskScreen,
    type LiquidityRiskRow,
    type LiquidityRiskTickerBundle,
    type LiquidityRiskUniverseStats,
    type LiquidityTier,
    type RiskTier,
} from './liquidityRiskEngine';

export interface LiquidityRiskScreenerProps {
    benchmarkStocks: Stock[];
    displayStocks: Stock[];
    portfolioTickers?: string[];
    benchmarkLabel: string;
    sourceLoading: boolean;
    benchmarkLoading: boolean;
    sourceError: string | null;
    searchQuery: string;
}

interface FetchCacheEntry {
    bundle: LiquidityRiskTickerBundle;
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
type SortKey = 'ticker' | 'beta' | 'vol' | 'max_dd' | 'turnover' | 'atr' | 'risk' | 'max_pos';
type LiquidityFilter = 'All' | 'Very High' | 'High' | 'Medium' | 'Low' | 'Very Low';
type RiskFilter = 'All' | 'High' | 'Medium' | 'Low';

interface SortConfig {
    key: SortKey;
    direction: SortDirection;
}

interface SortableHeader {
    key: SortKey;
    label: string;
}

const CONCURRENCY_LIMIT = 6;

const SORTABLE_HEADERS: SortableHeader[] = [
    { key: 'ticker', label: 'Ticker / Industry' },
    { key: 'beta', label: 'Beta' },
    { key: 'vol', label: 'Vol 1Y' },
    { key: 'max_dd', label: 'Max DD 3Y' },
    { key: 'turnover', label: 'Turnover/day' },
    { key: 'atr', label: 'ATR%' },
    { key: 'risk', label: 'Risk Score' },
    { key: 'max_pos', label: 'Max Pos %' },
];

const LIQUIDITY_FILTER_OPTIONS: LiquidityFilter[] = ['All', 'Very High', 'High', 'Medium', 'Low', 'Very Low'];
const RISK_FILTER_OPTIONS: RiskFilter[] = ['All', 'High', 'Medium', 'Low'];

const runWithConcurrency = async <T,>(
    items: T[],
    limit: number,
    worker: (item: T) => Promise<void>,
): Promise<void> => {
    if (items.length === 0) return;
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
    if (value === null || !Number.isFinite(value)) return '-';
    return new Intl.NumberFormat('en-US', {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
    }).format(value);
};

const formatPercent = (value: number | null, digits = 1): string => {
    if (value === null || !Number.isFinite(value)) return '-';
    return `${value.toFixed(digits)}%`;
};

const formatCompact = (value: number | null): string => {
    if (value === null || !Number.isFinite(value)) return '-';
    return new Intl.NumberFormat('en-US', {
        notation: 'compact',
        maximumFractionDigits: 2,
    }).format(value);
};

const mean = (values: Array<number | null>): number | null => {
    const valid = values.filter((value): value is number => value !== null && Number.isFinite(value));
    if (valid.length === 0) return null;
    return valid.reduce((acc, value) => acc + value, 0) / valid.length;
};

const median = (values: Array<number | null>): number | null => {
    const valid = values
        .filter((value): value is number => value !== null && Number.isFinite(value))
        .sort((a, b) => a - b);
    if (valid.length === 0) return null;
    const middle = Math.floor(valid.length / 2);
    if (valid.length % 2 === 0) {
        return (valid[middle - 1] + valid[middle]) / 2;
    }
    return valid[middle];
};

const betaToneClass = (value: number | null): string => {
    if (value === null || !Number.isFinite(value)) return 'text-base-content/50';
    if (value < 1.0) return 'text-success';
    if (value <= 1.3) return 'text-warning';
    return 'text-error';
};

const volToneClass = (value: number | null): string => {
    if (value === null || !Number.isFinite(value)) return 'text-base-content/50';
    if (value < 30) return 'text-success';
    if (value <= 40) return 'text-warning';
    return 'text-error';
};

const drawdownToneClass = (value: number | null): string => {
    if (value === null || !Number.isFinite(value)) return 'text-base-content/50';
    if (value > -25) return 'text-success';
    if (value >= -40) return 'text-warning';
    return 'text-error';
};

const liquidityBadgeClass = (tier: LiquidityTier): string => {
    switch (tier) {
        case 'Very High':
            return 'badge-info';
        case 'High':
            return 'badge-success';
        case 'Medium':
            return 'badge-warning';
        case 'Low':
        case 'Very Low':
            return 'badge-error';
        default:
            return 'badge-ghost';
    }
};

const riskBadgeClass = (tier: RiskTier): string => {
    switch (tier) {
        case 'Low':
            return 'badge-success';
        case 'Medium':
            return 'badge-warning';
        case 'High':
            return 'badge-error';
        default:
            return 'badge-ghost';
    }
};

const compareNullableNumber = (a: number | null, b: number | null, direction: SortDirection): number => {
    if (a === null && b === null) return 0;
    if (a === null) return 1;
    if (b === null) return -1;
    if (direction === 'asc') return a - b;
    return b - a;
};

export const LiquidityRiskScreener: React.FC<LiquidityRiskScreenerProps> = ({
    benchmarkStocks,
    displayStocks,
    portfolioTickers = [],
    benchmarkLabel,
    sourceLoading,
    benchmarkLoading,
    sourceError,
    searchQuery,
}) => {
    const [rows, setRows] = useState<LiquidityRiskRow[]>([]);
    const [stats, setStats] = useState<LiquidityRiskUniverseStats | null>(null);
    const [progress, setProgress] = useState<ScreenerProgress>({
        total: 0,
        loaded: 0,
        failed: 0,
        running: false,
    });
    const [sortConfig, setSortConfig] = useState<SortConfig>({
        key: 'risk',
        direction: 'desc',
    });
    const [liquidityFilter, setLiquidityFilter] = useState<LiquidityFilter>('All');
    const [riskFilter, setRiskFilter] = useState<RiskFilter>('All');

    const cacheRef = useRef<Map<string, FetchCacheEntry>>(new Map());
    const inflightRef = useRef<Map<string, Promise<FetchCacheEntry>>>(new Map());
    const runIdRef = useRef(0);

    const portfolioTickerSet = useMemo(() => {
        return new Set(portfolioTickers.map((ticker) => ticker.toUpperCase()));
    }, [portfolioTickers]);

    const fetchTickerBundle = useCallback(async (ticker: string): Promise<FetchCacheEntry> => {
        const normalizedTicker = ticker.toUpperCase();
        const cached = cacheRef.current.get(normalizedTicker);
        if (cached) return cached;

        const inflight = inflightRef.current.get(normalizedTicker);
        if (inflight) return inflight;

        const promise = (async () => {
            const result = await Promise.allSettled([
                stockApi.getPriceHistoryOhlcv(normalizedTicker),
            ]);

            const errors: string[] = [];
            let ohlcv: OhlcvDataPoint[] = [];

            const historyResult = result[0];
            if (historyResult.status === 'fulfilled') {
                ohlcv = historyResult.value.data
                    .filter((item): item is OhlcvDataPoint => Boolean(item) && typeof item === 'object');
            } else {
                errors.push(`ohlcv: ${buildErrorMessage(historyResult.reason, 'request failed')}`);
            }

            const bundle: LiquidityRiskTickerBundle = {
                ohlcv,
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
                if (cancelled || runIdRef.current !== activeRunId) return;
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
            if (cancelled || runIdRef.current !== activeRunId) return;

            const bundlesByTicker = new Map<string, LiquidityRiskTickerBundle>();
            const failedTickers = new Set<string>();

            requiredTickers.forEach((ticker) => {
                const cached = cacheRef.current.get(ticker);
                if (!cached) return;
                bundlesByTicker.set(ticker, cached.bundle);
                if (cached.hasErrors) {
                    failedTickers.add(ticker);
                }
            });

            const computed = buildLiquidityRiskScreen({
                benchmarkStocks,
                displayStocks,
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
            console.error('Failed to compute liquidity risk screener rows:', error);
            if (!cancelled && runIdRef.current === activeRunId) {
                setProgress((prev) => ({ ...prev, running: false }));
            }
        });

        return () => {
            cancelled = true;
        };
    }, [benchmarkStocks, displayStocks, fetchTickerBundle]);

    const filteredRows = useMemo(() => {
        const query = searchQuery.trim().toLowerCase();
        return rows.filter((row) => {
            if (query && !row.ticker.toLowerCase().includes(query)) {
                return false;
            }
            if (liquidityFilter !== 'All' && row.liquidity_tier !== liquidityFilter) {
                return false;
            }
            if (riskFilter !== 'All' && row.risk_tier !== riskFilter) {
                return false;
            }
            return true;
        });
    }, [liquidityFilter, riskFilter, rows, searchQuery]);

    const sortedRows = useMemo(() => {
        const items = [...filteredRows];
        items.sort((a, b) => {
            switch (sortConfig.key) {
                case 'ticker':
                    return sortConfig.direction === 'asc'
                        ? a.ticker.localeCompare(b.ticker)
                        : b.ticker.localeCompare(a.ticker);
                case 'beta':
                    return compareNullableNumber(a.beta_1y, b.beta_1y, sortConfig.direction);
                case 'vol':
                    return compareNullableNumber(a.vol_1y_pct, b.vol_1y_pct, sortConfig.direction);
                case 'max_dd':
                    return compareNullableNumber(a.max_dd_3y_pct, b.max_dd_3y_pct, sortConfig.direction);
                case 'turnover':
                    return compareNullableNumber(a.avg_turnover, b.avg_turnover, sortConfig.direction);
                case 'atr':
                    return compareNullableNumber(a.atr_pct, b.atr_pct, sortConfig.direction);
                case 'risk':
                    return compareNullableNumber(a.risk_score, b.risk_score, sortConfig.direction);
                case 'max_pos':
                    return compareNullableNumber(a.adjusted_max_pct, b.adjusted_max_pct, sortConfig.direction);
                default:
                    return 0;
            }
        });
        return items;
    }, [filteredRows, sortConfig]);

    const kpi = useMemo(() => {
        return {
            count: filteredRows.length,
            avgBeta: mean(filteredRows.map((row) => row.beta_1y)),
            avgVol: mean(filteredRows.map((row) => row.vol_1y_pct)),
            avgMaxDd: mean(filteredRows.map((row) => row.max_dd_3y_pct)),
            avgTurnover: mean(filteredRows.map((row) => row.avg_turnover)),
            medianRisk: median(filteredRows.map((row) => row.risk_score)),
        };
    }, [filteredRows]);

    const toggleSort = (key: SortKey) => {
        setSortConfig((prev) => {
            if (prev.key !== key) {
                return { key, direction: key === 'ticker' ? 'asc' : 'desc' };
            }
            return { key, direction: prev.direction === 'desc' ? 'asc' : 'desc' };
        });
    };

    const headerButtonClass = (key: SortKey): string => {
        if (sortConfig.key !== key) return 'text-base-content/60';
        return 'text-primary';
    };

    const formatPosPct = (value: number | null): string => {
        if (value === null || !Number.isFinite(value)) return '-';
        return `${value.toFixed(1)}%`;
    };

    return (
        <div className="space-y-4">
            <div className="card bg-base-100 shadow-md border border-base-300">
                <div className="card-body p-4 gap-4">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                        <h3 className="text-lg font-semibold">Liquidity & Volatility Risk Screener</h3>
                        <div className="text-xs text-base-content/60 max-w-2xl">
                            Position-sizing focus using volatility, beta, drawdown, and liquidity risk.
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
                        <span className="badge badge-outline">
                            Score range: {formatNumber(stats?.score_min ?? null, 1)} - {formatNumber(stats?.score_max ?? null, 1)}
                        </span>
                    </div>

                    <div className="flex flex-wrap items-center gap-2">
                        <span className={`badge ${progress.failed > 0 ? 'badge-warning' : 'badge-outline'}`}>
                            Failed tickers: {progress.failed}
                        </span>
                        <span className="badge badge-outline">P20: {formatCompact(stats?.liquidity_thresholds.p20 ?? null)}</span>
                        <span className="badge badge-outline">P40: {formatCompact(stats?.liquidity_thresholds.p40 ?? null)}</span>
                        <span className="badge badge-outline">P60: {formatCompact(stats?.liquidity_thresholds.p60 ?? null)}</span>
                        <span className="badge badge-outline">P80: {formatCompact(stats?.liquidity_thresholds.p80 ?? null)}</span>
                    </div>

                    {sourceError ? (
                        <div className="alert alert-error">
                            <span>{sourceError}</span>
                        </div>
                    ) : null}

                    {progress.failed > 0 ? (
                        <div className="alert alert-warning">
                            <span>Some ticker requests failed; rows are still computed with best available data.</span>
                        </div>
                    ) : null}

                    {(sourceLoading || benchmarkLoading) ? (
                        <div className="flex flex-col items-center justify-center h-24 gap-2">
                            <span className="loading loading-spinner loading-md text-primary"></span>
                            <p className="text-sm text-base-content/70">Loading source universe...</p>
                        </div>
                    ) : null}

                    <div className="flex flex-wrap items-center gap-3">
                        <label className="form-control w-48">
                            <div className="label py-1">
                                <span className="label-text text-xs">Liquidity tier</span>
                            </div>
                            <select
                                className="select select-sm select-bordered"
                                value={liquidityFilter}
                                onChange={(event) => setLiquidityFilter(event.target.value as LiquidityFilter)}
                            >
                                {LIQUIDITY_FILTER_OPTIONS.map((option) => (
                                    <option key={option} value={option}>{option}</option>
                                ))}
                            </select>
                        </label>

                        <label className="form-control w-40">
                            <div className="label py-1">
                                <span className="label-text text-xs">Risk tier</span>
                            </div>
                            <select
                                className="select select-sm select-bordered"
                                value={riskFilter}
                                onChange={(event) => setRiskFilter(event.target.value as RiskFilter)}
                            >
                                {RISK_FILTER_OPTIONS.map((option) => (
                                    <option key={option} value={option}>{option}</option>
                                ))}
                            </select>
                        </label>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-6 gap-2">
                        <div className="rounded-lg border border-base-300 p-3 bg-base-100">
                            <div className="text-xs text-base-content/60">Stocks shown</div>
                            <div className="text-lg font-semibold">{kpi.count}</div>
                        </div>
                        <div className="rounded-lg border border-base-300 p-3 bg-base-100">
                            <div className="text-xs text-base-content/60">Avg Beta</div>
                            <div className="text-lg font-semibold">{formatNumber(kpi.avgBeta, 2)}</div>
                        </div>
                        <div className="rounded-lg border border-base-300 p-3 bg-base-100">
                            <div className="text-xs text-base-content/60">Avg Volatility</div>
                            <div className="text-lg font-semibold">{formatPercent(kpi.avgVol, 1)}</div>
                        </div>
                        <div className="rounded-lg border border-base-300 p-3 bg-base-100">
                            <div className="text-xs text-base-content/60">Avg Max Drawdown</div>
                            <div className="text-lg font-semibold text-error">{formatPercent(kpi.avgMaxDd, 1)}</div>
                        </div>
                        <div className="rounded-lg border border-base-300 p-3 bg-base-100">
                            <div className="text-xs text-base-content/60">Avg Daily Turnover</div>
                            <div className="text-lg font-semibold">{formatCompact(kpi.avgTurnover)}</div>
                        </div>
                        <div className="rounded-lg border border-base-300 p-3 bg-base-100">
                            <div className="text-xs text-base-content/60">Median Risk Score</div>
                            <div className="text-lg font-semibold">{formatNumber(kpi.medianRisk, 1)}</div>
                        </div>
                    </div>
                </div>
            </div>

            <div className="card bg-base-100 shadow-md border border-base-300">
                <div className="card-body p-0">
                    <div className="dashboard-adaptive-table-wrap border border-base-300 rounded-lg">
                        <table className="table table-xs table-pin-rows w-max min-w-full">
                            <thead>
                                <tr>
                                    <th className="text-base-content/60 text-xs">#</th>
                                    {SORTABLE_HEADERS.map((header) => (
                                        <th key={header.key} className="align-top">
                                            <button
                                                className={`btn btn-ghost btn-xs h-auto min-h-0 px-1 py-0.5 normal-case ${headerButtonClass(header.key)}`}
                                                onClick={() => toggleSort(header.key)}
                                            >
                                                {header.label}
                                            </button>
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {sortedRows.length === 0 ? (
                                    <tr>
                                        <td colSpan={SORTABLE_HEADERS.length + 1} className="text-center text-base-content/60 py-6">
                                            {progress.running ? 'Computing risk rows...' : 'No stocks match current filters.'}
                                        </td>
                                    </tr>
                                ) : sortedRows.map((row, index) => {
                                    const isInPortfolio = portfolioTickerSet.has(row.ticker.toUpperCase());
                                    return (
                                        <tr key={row.ticker} className="hover">
                                            <td>{index + 1}</td>
                                            <td className="min-w-36">
                                                <div className={`font-semibold ${isInPortfolio ? 'text-accent' : ''}`}>{row.ticker}</div>
                                                <div className="text-[11px] text-base-content/60 truncate max-w-48" title={row.industry}>
                                                    {row.industry}
                                                </div>
                                            </td>
                                            <td className={`whitespace-nowrap ${betaToneClass(row.beta_1y)}`}>
                                                {formatNumber(row.beta_1y, 2)}
                                            </td>
                                            <td className={`whitespace-nowrap ${volToneClass(row.vol_1y_pct)}`}>
                                                {formatPercent(row.vol_1y_pct, 1)}
                                            </td>
                                            <td className={`whitespace-nowrap ${drawdownToneClass(row.max_dd_3y_pct)}`}>
                                                {formatPercent(row.max_dd_3y_pct, 1)}
                                            </td>
                                            <td className="whitespace-nowrap">
                                                <div>{formatCompact(row.avg_turnover)}</div>
                                                <span className={`badge badge-xs ${liquidityBadgeClass(row.liquidity_tier)}`}>
                                                    {row.liquidity_tier}
                                                </span>
                                            </td>
                                            <td className="whitespace-nowrap">{formatPercent(row.atr_pct, 1)}</td>
                                            <td className="whitespace-nowrap">
                                                <div className="flex items-center gap-1">
                                                    <span className="font-semibold">{formatNumber(row.risk_score, 1)}</span>
                                                    <span className={`badge badge-sm ${riskBadgeClass(row.risk_tier)}`}>
                                                        {row.risk_tier === 'Unknown' ? 'N/A' : row.risk_tier.toUpperCase()}
                                                    </span>
                                                </div>
                                            </td>
                                            <td className="whitespace-nowrap">
                                                <div className="text-xs leading-tight">
                                                    <div>Base: {formatPosPct(row.base_max_pct)}</div>
                                                    <div>Adj: {formatPosPct(row.adjusted_max_pct)}</div>
                                                    <div className="text-base-content/60">{row.position_tier}</div>
                                                </div>
                                            </td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default LiquidityRiskScreener;
