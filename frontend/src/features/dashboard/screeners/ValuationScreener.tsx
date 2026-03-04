import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { stockApi } from '../../../api/stockApi';
import type { Stock } from '../../../api/stockApi';
import {
    buildValuationScreen,
    type ValuationTickerBundle,
    type ValuationRow,
    type ValuationUniverseStats,
} from './valuationEngine';

export interface ValuationScreenerProps {
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
    bundle: ValuationTickerBundle;
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
    | 'rank'
    | 'ticker'
    | 'sector'
    | 'market_cap'
    | 'pe'
    | 'pb'
    | 'roe'
    | 'roa'
    | 'growth'
    | 'stability'
    | 'valuation'
    | 'quality'
    | 'overall'
    | 'quadrant'
    | 'verdict'
    | 'm1'
    | 'm6'
    | 'm1y'
    | 'completeness';

interface SortConfig {
    key: SortKey;
    direction: SortDirection;
}

interface SortableHeader {
    key: SortKey;
    line1: string;
    line2?: string;
    tooltip?: string;
}

const CONCURRENCY_LIMIT = 6;

const SORTABLE_HEADERS: SortableHeader[] = [
    { key: 'rank', line1: 'Rank' },
    { key: 'ticker', line1: 'Ticker' },
    { key: 'sector', line1: 'Sector' },
    { key: 'market_cap', line1: 'MCap' },
    { key: 'pe', line1: 'P/E' },
    { key: 'pb', line1: 'P/B' },
    { key: 'roe', line1: 'ROE' },
    { key: 'roa', line1: 'ROA' },
    { key: 'growth', line1: 'Growth', line2: 'Score' },
    { key: 'stability', line1: 'Stability', line2: 'Score' },
    { key: 'valuation', line1: 'Valuation', line2: 'Score' },
    { key: 'quality', line1: 'Quality', line2: 'Score' },
    { key: 'overall', line1: 'Overall', line2: 'Score' },
    {
        key: 'quadrant',
        line1: 'Quadrant',
        tooltip: 'Based on Valuation Score and Quality Score (cutoff: 65).\n'
            + 'VALUE PICK: valuation >= 65 and quality >= 65\n'
            + 'Premium for Quality: valuation < 65 and quality >= 65\n'
            + 'Cheap but Risky: valuation >= 65 and quality < 65\n'
            + 'Expensive & Weak: valuation < 65 and quality < 65',
    },
    {
        key: 'verdict',
        line1: 'Verdict',
        tooltip: 'Based on Overall Score.\n'
            + '>= 80: Strong Buy Signal\n'
            + '>= 70: Attractive\n'
            + '>= 60: Fair Value\n'
            + '>= 50: Hold / Monitor\n'
            + '>= 40: Caution\n'
            + '< 40: Avoid / Overpriced',
    },
    { key: 'm1', line1: '1M' },
    { key: 'm6', line1: '6M' },
    { key: 'm1y', line1: '1Y' },
    { key: 'completeness', line1: 'Data' },
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
    if (value === null || !Number.isFinite(value)) return '-';
    return new Intl.NumberFormat('en-US', {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
    }).format(value);
};

const formatCompact = (value: number | null): string => {
    if (value === null || !Number.isFinite(value)) return '-';
    return new Intl.NumberFormat('en-US', {
        notation: 'compact',
        maximumFractionDigits: 2,
    }).format(value);
};

const formatRatePercent = (value: number | null, digits = 1): string => {
    if (value === null || !Number.isFinite(value)) return '-';
    return `${(value * 100).toFixed(digits)}%`;
};

const formatPercent = (value: number | null, digits = 1): string => {
    if (value === null || !Number.isFinite(value)) return '-';
    return `${value.toFixed(digits)}%`;
};

const momentumTextClass = (value: number | null): string => {
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

const quadrantClass = (value: string): string => {
    switch (value) {
        case 'VALUE PICK':
            return 'badge-success';
        case 'Premium for Quality':
            return 'badge-info';
        case 'Cheap but Risky':
            return 'badge-warning';
        case 'Expensive & Weak':
            return 'badge-error';
        default:
            return 'badge-ghost';
    }
};

const verdictClass = (value: string): string => {
    switch (value) {
        case 'Strong Buy Signal':
            return 'badge-success';
        case 'Attractive':
            return 'badge-primary';
        case 'Fair Value':
            return 'badge-info';
        case 'Hold / Monitor':
            return 'badge-warning';
        case 'Caution':
            return 'badge-warning';
        case 'Avoid / Overpriced':
            return 'badge-error';
        default:
            return 'badge-ghost';
    }
};

const completenessClass = (value: ValuationRow['data_completeness']): string => {
    switch (value) {
        case 'Complete':
            return 'badge-success';
        case 'Partial':
            return 'badge-warning';
        case 'Estimated':
            return 'badge-error';
        default:
            return 'badge-ghost';
    }
};

const headerButtonClass = (sortConfig: SortConfig, key: SortKey) => {
    if (sortConfig.key !== key) {
        return 'text-base-content/60';
    }
    return 'text-primary';
};

const renderHeaderLabel = (line1: string, line2?: string) => (
    <span className="inline-flex flex-col items-start leading-tight whitespace-nowrap normal-case">
        <span>{line1}</span>
        <span className={`text-[10px] min-h-[0.75rem] ${line2 ? 'text-base-content/60' : 'invisible'}`}>
            {line2 ?? '\u00A0'}
        </span>
    </span>
);

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

export const ValuationScreener: React.FC<ValuationScreenerProps> = ({
    benchmarkStocks,
    displayStocks,
    portfolioTickers = [],
    benchmarkLabel,
    sourceLoading,
    benchmarkLoading,
    sourceError,
    searchQuery,
}) => {
    const [rows, setRows] = useState<ValuationRow[]>([]);
    const [stats, setStats] = useState<ValuationUniverseStats | null>(null);
    const [progress, setProgress] = useState<ScreenerProgress>({
        total: 0,
        loaded: 0,
        failed: 0,
        running: false,
    });
    const [sortConfig, setSortConfig] = useState<SortConfig>({
        key: 'overall',
        direction: 'desc',
    });

    const cacheRef = useRef<Map<string, FetchCacheEntry>>(new Map());
    const inflightRef = useRef<Map<string, Promise<FetchCacheEntry>>>(new Map());
    const runIdRef = useRef(0);
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
            const [
                ratiosResult,
                incomeResult,
                cashflowResult,
                volumeHistoryResult,
                priceHistoryResult,
            ] = await Promise.allSettled([
                stockApi.getFinancialRatios(normalizedTicker),
                stockApi.getIncomeStatement(normalizedTicker),
                stockApi.getCashFlow(normalizedTicker),
                stockApi.getVolumeHistory(normalizedTicker, 20, { autoSync: false }),
                stockApi.getPriceHistory(normalizedTicker, 10, { autoSync: false }),
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

            const unwrapHistoryRows = (
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

            const bundle: ValuationTickerBundle = {
                ratios: unwrapDataRows(ratiosResult, 'ratios'),
                income: unwrapDataRows(incomeResult, 'income'),
                cashflow: unwrapDataRows(cashflowResult, 'cashflow'),
                volume_history: unwrapHistoryRows(volumeHistoryResult, 'volume_history'),
                price_history: unwrapHistoryRows(priceHistoryResult, 'price_history'),
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

            const bundlesByTicker = new Map<string, ValuationTickerBundle>();
            const failedTickers = new Set<string>();

            requiredTickers.forEach((ticker) => {
                const cached = cacheRef.current.get(ticker);
                if (!cached) return;
                bundlesByTicker.set(ticker, cached.bundle);
                if (cached.hasErrors) {
                    failedTickers.add(ticker);
                }
            });

            const computed = buildValuationScreen({
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
            console.error('Failed to compute valuation screener rows:', error);
            if (!cancelled && runIdRef.current === activeRunId) {
                setProgress((prev) => ({ ...prev, running: false }));
            }
        });

        return () => {
            cancelled = true;
        };
    }, [benchmarkStocks, displayStocks, fetchTickerBundle]);

    const filteredRows = useMemo(() => {
        if (!searchQuery.trim()) {
            return rows;
        }
        const query = searchQuery.trim().toLowerCase();
        return rows.filter((row) => row.ticker.toLowerCase().includes(query));
    }, [rows, searchQuery]);

    const sortedRows = useMemo(() => {
        const direction = sortConfig.direction === 'asc' ? 1 : -1;
        const valueFor = (row: ValuationRow): string | number => {
            switch (sortConfig.key) {
                case 'rank':
                    return row.scores.rank;
                case 'ticker':
                    return row.ticker;
                case 'sector':
                    return row.sector;
                case 'market_cap':
                    return row.metrics.market_cap_bn ?? -Infinity;
                case 'pe':
                    return row.metrics.pe ?? -Infinity;
                case 'pb':
                    return row.metrics.pb ?? -Infinity;
                case 'roe':
                    return row.metrics.roe ?? -Infinity;
                case 'roa':
                    return row.metrics.roa ?? -Infinity;
                case 'growth':
                    return row.scores.growth_score;
                case 'stability':
                    return row.scores.stability_score;
                case 'valuation':
                    return row.scores.valuation_score;
                case 'quality':
                    return row.scores.quality_score;
                case 'overall':
                    return row.scores.overall_score;
                case 'quadrant':
                    return row.scores.quadrant;
                case 'verdict':
                    return row.scores.verdict;
                case 'm1':
                    return row.metrics.momentum_1m ?? -Infinity;
                case 'm6':
                    return row.metrics.momentum_6m ?? -Infinity;
                case 'm1y':
                    return row.metrics.momentum_1y ?? -Infinity;
                case 'completeness':
                    return row.data_completeness;
                default:
                    return row.scores.overall_score;
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
                return { key, direction: key === 'rank' ? 'asc' : 'desc' };
            }
            return { key, direction: prev.direction === 'desc' ? 'asc' : 'desc' };
        });
    };

    const renderSortableHeader = (key: SortKey) => {
        const header = SORTABLE_HEADER_BY_KEY[key];
        return (
            <button
                className={`btn btn-ghost btn-xs h-auto min-h-0 px-1 py-0.5 normal-case ${headerButtonClass(sortConfig, key)}`}
                onClick={() => toggleSort(key)}
                title={header.tooltip}
            >
                {renderHeaderLabel(header.line1, header.line2)}
            </button>
        );
    };

    return (
        <div className="space-y-4">
            <div className="card bg-base-100 shadow-md border border-base-300">
                <div className="card-body p-4 gap-4">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                        <h3 className="text-lg font-semibold">Valuation Screener</h3>
                        <div className="text-xs text-base-content/60 max-w-2xl">
                            Sector-adjusted valuation v3 scoring with price_board-first pricing and local DB fallback when needed.
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
                    </div>

                    <div className="flex flex-wrap items-center gap-2">
                        <span className="badge badge-warning">Failed tickers: {progress.failed}</span>
                        <span className="badge badge-outline">DB price fallback: {stats?.fallback_price_count ?? 0}</span>
                        <span className="badge badge-outline">MCap fallback: {stats?.fallback_market_cap_count ?? 0}</span>
                        <span className="badge badge-outline">P/E fallback: {stats?.fallback_pe_count ?? 0}</span>
                        <span className="badge badge-outline">Partial rows: {stats?.partial_count ?? 0}</span>
                        <span className="badge badge-outline">Estimated rows: {stats?.estimated_count ?? 0}</span>
                    </div>

                    {sourceError ? (
                        <div className="alert alert-error">
                            <span>{sourceError}</span>
                        </div>
                    ) : null}

                    {progress.failed > 0 ? (
                        <div className="alert alert-warning">
                            <span>
                                Some ticker endpoints partially failed. Scores are computed with best available data.
                            </span>
                        </div>
                    ) : null}

                    {(sourceLoading || benchmarkLoading) && (
                        <div className="flex flex-col items-center justify-center h-24 gap-2">
                            <span className="loading loading-spinner loading-md text-primary"></span>
                            <p className="text-sm text-base-content/70">
                                Loading source universe...
                            </p>
                        </div>
                    )}
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
                                </tr>
                            </thead>
                            <tbody>
                                {sortedRows.length === 0 ? (
                                    <tr>
                                        <td colSpan={SORTABLE_HEADERS.length} className="text-center text-base-content/60 py-6">
                                            {progress.running ? 'Computing valuation rows...' : 'No valuation rows for current scope.'}
                                        </td>
                                    </tr>
                                ) : sortedRows.map((row) => {
                                    const isInPortfolio = portfolioTickerSet.has(row.ticker.toUpperCase());
                                    const companyNameTooltip = row.company_name?.trim() || row.ticker;
                                    return (
                                        <tr key={row.ticker} className="hover">
                                            <td className="whitespace-nowrap">{row.scores.rank || '-'}</td>
                                            <td
                                                className={`font-semibold whitespace-nowrap w-20 ${isInPortfolio ? 'text-accent' : ''}`}
                                                title={companyNameTooltip}
                                            >
                                                {row.ticker}
                                            </td>
                                            <td className="max-w-48 truncate" title={row.sector}>{row.sector}</td>
                                            <td className="whitespace-nowrap">{formatCompact(row.metrics.market_cap_bn)}</td>
                                            <td className="whitespace-nowrap">{formatNumber(row.metrics.pe, 2)}</td>
                                            <td className="whitespace-nowrap">{formatNumber(row.metrics.pb, 2)}</td>
                                            <td className="whitespace-nowrap">{formatRatePercent(row.metrics.roe, 1)}</td>
                                            <td className="whitespace-nowrap">{formatRatePercent(row.metrics.roa, 1)}</td>
                                            <td className="whitespace-nowrap">{formatNumber(row.scores.growth_score, 1)}</td>
                                            <td className="whitespace-nowrap">{formatNumber(row.scores.stability_score, 1)}</td>
                                            <td className="whitespace-nowrap">{formatNumber(row.scores.valuation_score, 1)}</td>
                                            <td className="whitespace-nowrap">{formatNumber(row.scores.quality_score, 1)}</td>
                                            <td className="font-semibold whitespace-nowrap">{formatNumber(row.scores.overall_score, 1)}</td>
                                            <td className="whitespace-nowrap">
                                                <span className={`badge badge-sm ${quadrantClass(row.scores.quadrant)}`}>
                                                    {row.scores.quadrant}
                                                </span>
                                            </td>
                                            <td className="whitespace-nowrap">
                                                <span className={`badge badge-sm ${verdictClass(row.scores.verdict)}`}>
                                                    {row.scores.verdict}
                                                </span>
                                            </td>
                                            <td className={`whitespace-nowrap ${momentumTextClass(row.metrics.momentum_1m)}`}>
                                                {formatPercent(row.metrics.momentum_1m, 1)}
                                            </td>
                                            <td className={`whitespace-nowrap ${momentumTextClass(row.metrics.momentum_6m)}`}>
                                                {formatPercent(row.metrics.momentum_6m, 1)}
                                            </td>
                                            <td className={`whitespace-nowrap ${momentumTextClass(row.metrics.momentum_1y)}`}>
                                                {formatPercent(row.metrics.momentum_1y, 1)}
                                            </td>
                                            <td className="whitespace-nowrap">
                                                <span className={`badge badge-sm ${completenessClass(row.data_completeness)}`}>
                                                    {row.data_completeness}
                                                </span>
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

export default ValuationScreener;
