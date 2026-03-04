import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { stockApi } from '../../../api/stockApi';
import type { Stock, IndustryInfo } from '../../../api/stockApi';
import {
    buildFinancialHealthScreen,
    formatPercentile,
    type TickerDatasetBundle,
    type TickerScreeningRow,
    type ScreenerUniverseStats,
} from './financialHealthEngine';

export interface FinancialHealthScreenerProps {
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
    bundle: TickerDatasetBundle;
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
    | 'sector'
    | 'market_cap'
    | 'z_score'
    | 'z_zone_base'
    | 'vf_score'
    | 'vf_pct'
    | 'z_pct'
    | 'de'
    | 'leverage'
    | 'rating'
    | 'quality';

interface SortConfig {
    key: SortKey;
    direction: SortDirection;
}

interface SortableHeaderConfig {
    key: SortKey;
    line1: string;
    line2?: string;
}

const CONCURRENCY_LIMIT = 6;
const SORTABLE_HEADERS: SortableHeaderConfig[] = [
    { key: 'ticker', line1: 'Ticker' },
    { key: 'sector', line1: 'Sector' },
    { key: 'market_cap', line1: 'MCap' },
    { key: 'z_score', line1: 'Z', line2: '(Model)' },
    { key: 'z_zone_base', line1: 'Z Base' },
    { key: 'z_pct', line1: 'Z %ile' },
    { key: 'vf_score', line1: 'Quality', line2: '(VF)' },
    { key: 'vf_pct', line1: 'Quality %ile', line2: '(Peers)' },
    { key: 'de', line1: 'D/E' },
    { key: 'leverage', line1: 'Leverage' },
    { key: 'rating', line1: 'Health' },
    { key: 'quality', line1: 'Data' },
];

const SORTABLE_HEADER_BY_KEY: Record<SortKey, SortableHeaderConfig> = SORTABLE_HEADERS.reduce((acc, header) => {
    acc[header.key] = header;
    return acc;
}, {} as Record<SortKey, SortableHeaderConfig>);

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

const formatCompact = (value: number | null): string => {
    if (value === null || !Number.isFinite(value)) return '-';
    return new Intl.NumberFormat('en-US', {
        notation: 'compact',
        maximumFractionDigits: 2,
    }).format(value);
};

const normalizeIndustryLabel = (value: string): string => {
    return value
        .normalize('NFD')
        .replace(/[\u0300-\u036f]/g, '')
        .toLowerCase()
        .trim();
};

const getVnHealthToneClasses = (rating: string): string => {
    switch (rating) {
        case 'Excellent':
            return 'bg-green-300 text-green-950 border-green-400';
        case 'Strong':
            return 'bg-green-200 text-green-900 border-green-300';
        case 'Good':
            return 'bg-lime-200 text-lime-900 border-lime-300';
        case 'Moderate':
            return 'bg-yellow-200 text-yellow-900 border-yellow-300';
        case 'Mixed':
            return 'bg-orange-200 text-orange-900 border-orange-300';
        case 'Weak':
            return 'bg-red-200 text-red-900 border-red-300';
        case 'Concern':
            return 'bg-red-300 text-red-950 border-red-400';
        default:
            return 'bg-base-300 text-base-content border-base-content/20';
    }
};

const getVnHealthSortValue = (rating: string): number => {
    switch (rating) {
        case 'Excellent':
            return 7;
        case 'Strong':
            return 6;
        case 'Good':
            return 5;
        case 'Moderate':
            return 4;
        case 'Mixed':
            return 3;
        case 'Weak':
            return 2;
        case 'Concern':
            return 1;
        default:
            return 0;
    }
};

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

export const FinancialHealthScreener: React.FC<FinancialHealthScreenerProps> = ({
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
    const [screenerRows, setScreenerRows] = useState<TickerScreeningRow[]>([]);
    const [screenerStats, setScreenerStats] = useState<ScreenerUniverseStats | null>(null);
    const [screenerProgress, setScreenerProgress] = useState<ScreenerProgress>({
        total: 0,
        loaded: 0,
        failed: 0,
        running: false,
    });
    const [sortConfig, setSortConfig] = useState<SortConfig>({
        key: 'vf_score',
        direction: 'desc',
    });
    const [benchmarkQualityOpen, setBenchmarkQualityOpen] = useState(false);

    const cacheRef = useRef<Map<string, FetchCacheEntry>>(new Map());
    const inflightRef = useRef<Map<string, Promise<FetchCacheEntry>>>(new Map());
    const runIdRef = useRef(0);
    const benchmarkQualityRef = useRef<HTMLDivElement>(null);
    const portfolioTickerSet = useMemo(() => {
        return new Set(portfolioTickers.map((ticker) => ticker.toUpperCase()));
    }, [portfolioTickers]);
    const industryFamilyByLevel2Name = useMemo(() => {
        const map = new Map<string, {
            family_code: string | null;
            family_name: string | null;
            family_en_name: string | null;
        }>();
        industries.forEach((industry) => {
            const normalizedName = normalizeIndustryLabel(industry.name ?? '');
            if (!normalizedName) return;
            map.set(normalizedName, {
                family_code: industry.family_code ?? null,
                family_name: industry.family_name ?? null,
                family_en_name: industry.family_en_name ?? null,
            });
        });
        return map;
    }, [industries]);

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
                balanceResult,
                incomeResult,
                cashflowResult,
                ratiosResult,
                priceHistoryResult,
                overviewResult,
                shareholdersResult,
            ] = await Promise.allSettled([
                stockApi.getBalanceSheet(normalizedTicker),
                stockApi.getIncomeStatement(normalizedTicker),
                stockApi.getCashFlow(normalizedTicker),
                stockApi.getFinancialRatios(normalizedTicker),
                stockApi.getPriceHistory(normalizedTicker, 1200, { autoSync: false }),
                stockApi.getCompanyOverview(normalizedTicker),
                stockApi.getShareholders(normalizedTicker),
            ]);

            const errors: string[] = [];
            const unwrapData = (result: PromiseSettledResult<{ data: unknown[] }>, label: string): Record<string, unknown>[] => {
                if (result.status === 'fulfilled') {
                    return result.value.data
                        .filter((row): row is object => Boolean(row) && typeof row === 'object')
                        .map((row) => ({ ...(row as Record<string, unknown>) }));
                }
                errors.push(`${label}: ${buildErrorMessage(result.reason, 'request failed')}`);
                return [];
            };

            const bundle: TickerDatasetBundle = {
                balance: unwrapData(balanceResult, 'balance'),
                income: unwrapData(incomeResult, 'income'),
                cashflow: unwrapData(cashflowResult, 'cashflow'),
                ratios: unwrapData(ratiosResult, 'ratios'),
                price_history: unwrapData(priceHistoryResult, 'price_history'),
                overview: unwrapData(overviewResult, 'overview'),
                shareholders: unwrapData(shareholdersResult, 'shareholders'),
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
                setScreenerRows([]);
                setScreenerStats(null);
                setScreenerProgress({ total: 0, loaded: 0, failed: 0, running: false });
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
            const bundlesByTicker = new Map<string, TickerDatasetBundle>();
            const failedTickers = new Set<string>();

            requiredTickers.forEach((ticker) => {
                const cache = cacheRef.current.get(ticker);
                if (!cache) return;
                bundlesByTicker.set(ticker, cache.bundle);
                if (cache.hasErrors) {
                    failedTickers.add(ticker);
                }
            });

            const computed = buildFinancialHealthScreen({
                benchmarkStocks,
                displayStocks,
                industryFamiliesByLevel2Name: industryFamilyByLevel2Name,
                bundlesByTicker,
                failedTickers,
            });

            setScreenerRows(computed.rows);
            setScreenerStats(computed.stats);
            setScreenerProgress((prev) => ({
                total: requiredTickers.length,
                loaded: bundlesByTicker.size,
                failed: failedTickers.size,
                running: prev.running,
            }));
        };

        const run = async () => {
            setScreenerProgress((prev) => ({
                ...prev,
                total: requiredTickers.length,
                loaded: requiredTickers.filter((ticker) => cacheRef.current.has(ticker)).length,
                failed: requiredTickers.filter((ticker) => cacheRef.current.get(ticker)?.hasErrors).length,
                running: true,
            }));

            recompute();

            const missingTickers = requiredTickers.filter((ticker) => !cacheRef.current.has(ticker));
            await runWithConcurrency(missingTickers, CONCURRENCY_LIMIT, async (ticker) => {
                await fetchTickerBundle(ticker);
                recompute();
            });

            if (!cancelled && runIdRef.current === activeRunId) {
                setScreenerProgress((prev) => ({
                    ...prev,
                    running: false,
                }));
            }
        };

        run().catch((error) => {
            console.error('Failed to compute screener rows:', error);
            if (!cancelled && runIdRef.current === activeRunId) {
                setScreenerProgress((prev) => ({
                    ...prev,
                    running: false,
                }));
            }
        });

        return () => {
            cancelled = true;
        };
    }, [benchmarkStocks, displayStocks, fetchTickerBundle, industryFamilyByLevel2Name]);

    useEffect(() => {
        if (!benchmarkQualityOpen) {
            return;
        }
        const handlePointerDown = (event: PointerEvent) => {
            if (!benchmarkQualityRef.current?.contains(event.target as Node)) {
                setBenchmarkQualityOpen(false);
            }
        };
        const handleEscape = (event: KeyboardEvent) => {
            if (event.key === 'Escape') {
                setBenchmarkQualityOpen(false);
            }
        };
        document.addEventListener('pointerdown', handlePointerDown);
        document.addEventListener('keydown', handleEscape);
        return () => {
            document.removeEventListener('pointerdown', handlePointerDown);
            document.removeEventListener('keydown', handleEscape);
        };
    }, [benchmarkQualityOpen]);

    const filteredRows = useMemo(() => {
        if (!searchQuery.trim()) {
            return screenerRows;
        }
        const query = searchQuery.trim().toLowerCase();
        return screenerRows.filter((row) => row.ticker.toLowerCase().includes(query));
    }, [searchQuery, screenerRows]);

    const sortedRows = useMemo(() => {
        const rows = [...filteredRows];
        const directionMultiplier = sortConfig.direction === 'asc' ? 1 : -1;

        const valueFor = (row: TickerScreeningRow): string | number => {
            switch (sortConfig.key) {
                case 'ticker':
                    return row.ticker;
                case 'sector':
                    return row.sector;
                case 'market_cap':
                    return row.snapshot.market_cap_vnd ?? -Infinity;
                case 'z_score':
                    return row.scores.z_score ?? -Infinity;
                case 'z_zone_base':
                    return row.scores.z_zone_base ?? '';
                case 'vf_score':
                    return row.scores.vf_score;
                case 'vf_pct':
                    return row.scores.vf_sector_pctile ?? -Infinity;
                case 'z_pct':
                    return row.scores.z_sector_pctile ?? -Infinity;
                case 'de':
                    return row.snapshot.debt_to_equity ?? -Infinity;
                case 'leverage':
                    return row.scores.leverage_flag;
                case 'rating':
                    return getVnHealthSortValue(row.scores.vn_health_rating_base);
                case 'quality':
                    return row.snapshot.data_quality;
                default:
                    return row.ticker;
            }
        };

        rows.sort((a, b) => {
            const va = valueFor(a);
            const vb = valueFor(b);
            if (typeof va === 'number' && typeof vb === 'number') {
                return (va - vb) * directionMultiplier;
            }
            return String(va).localeCompare(String(vb)) * directionMultiplier;
        });
        return rows;
    }, [filteredRows, sortConfig]);

    const toggleSort = (key: SortKey) => {
        setSortConfig((prev) => {
            if (prev.key !== key) {
                return { key, direction: 'desc' };
            }
            return { key, direction: prev.direction === 'desc' ? 'asc' : 'desc' };
        });
    };

    const headerButtonClass = (key: SortKey) => {
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

    const renderSortableHeader = (key: SortKey) => {
        const header = SORTABLE_HEADER_BY_KEY[key];
        return (
            <button
                className={`btn btn-ghost btn-xs h-auto min-h-0 px-1 py-0.5 normal-case ${headerButtonClass(key)}`}
                onClick={() => toggleSort(key)}
            >
                {renderHeaderLabel(header.line1, header.line2)}
            </button>
        );
    };

    const renderStaticHeader = (label: string) => (
        <span className="btn btn-ghost btn-xs h-auto min-h-0 px-1 py-0.5 normal-case pointer-events-none text-base-content/60">
            {renderHeaderLabel(label)}
        </span>
    );

    const benchmarkQuality = screenerStats?.benchmark_quality;
    const benchmarkQualityInsufficient = benchmarkQuality?.is_insufficient
        ?? screenerStats?.insufficient_benchmark
        ?? false;
    const benchmarkFallbackUsed = benchmarkQuality?.fallback_used ?? false;
    const benchmarkFallbackRelatedCount = benchmarkQuality?.fallback_counts.related ?? 0;
    const benchmarkFallbackMarketCount = benchmarkQuality?.fallback_counts.market ?? 0;
    const benchmarkFallbackTotal = benchmarkFallbackRelatedCount + benchmarkFallbackMarketCount;
    const benchmarkQualityBadgeClass = benchmarkQualityInsufficient
        ? 'badge-warning'
        : (benchmarkFallbackUsed ? 'badge-info' : 'badge-outline');
    const benchmarkQualityLabel = benchmarkQualityInsufficient
        ? 'Insufficient'
        : (benchmarkFallbackUsed ? 'OK (Fallback-used)' : 'OK');
    const lowPeerSectors = benchmarkQuality?.low_peer_sectors ?? [];
    const lowPeerRequired = lowPeerSectors[0]?.min_required ?? 8;
    return (
        <div className="card bg-base-100 shadow-md border border-base-300">
            <div className="card-body p-4 gap-4">
                <div className="flex flex-wrap items-center justify-between gap-2">
                    <h3 className="text-lg font-semibold">Financial Health Screener</h3>
                    <div
                        className="text-xs text-base-content/60 max-w-2xl"
                        title="Quality (VF) is peer-relative (sector if N>=8, else related family fallback, then market fallback). VN Health uses a fixed Quality x Base-Z matrix. SOE can apply a bounded one-notch modifier."
                    >
                        Benchmark uses selected index base. Search/filter only narrows displayed rows.
                    </div>
                </div>

                {sourceError ? (
                    <div className="alert alert-error">
                        <span>{sourceError}</span>
                    </div>
                ) : null}

                {screenerProgress.failed > 0 ? (
                    <div className="alert alert-warning">
                        <span>
                            {screenerProgress.failed} ticker(s) had partial endpoint failures. Results are still shown with best available data.
                        </span>
                    </div>
                ) : null}

                <div className="flex flex-wrap items-center gap-2">
                    <span className={`badge ${screenerProgress.running ? 'badge-info' : 'badge-ghost'}`}>
                        {screenerProgress.running ? 'Computing...' : 'Ready'}
                    </span>
                    <span className="badge badge-outline">
                        Benchmark: {benchmarkLabel}
                    </span>
                    <span className="badge badge-outline">
                        Universe: {screenerStats?.benchmark_size ?? benchmarkStocks.length}
                    </span>
                    <span className="badge badge-outline">
                        Loaded: {screenerProgress.loaded}/{screenerProgress.total}
                    </span>
                    <div
                        className="relative inline-flex items-center"
                        ref={benchmarkQualityRef}
                        onMouseEnter={() => setBenchmarkQualityOpen(true)}
                        onMouseLeave={() => setBenchmarkQualityOpen(false)}
                    >
                        <span
                            className={`badge ${benchmarkQualityBadgeClass} cursor-pointer`}
                            title={benchmarkFallbackUsed
                                ? `Fallback used for ${benchmarkFallbackTotal} ticker(s): related ${benchmarkFallbackRelatedCount}, market ${benchmarkFallbackMarketCount}`
                                : undefined}
                            onClick={() => setBenchmarkQualityOpen((prev) => !prev)}
                        >
                            Benchmark quality: {benchmarkQualityLabel}
                        </span>
                        {benchmarkQualityOpen ? (
                            <div className="absolute right-0 top-full mt-2 z-20 w-80 max-w-[90vw] rounded-lg border border-base-300 bg-base-100 shadow-xl p-3 text-xs">
                                <div className="font-semibold mb-2">Benchmark quality details</div>
                                <div>
                                    Original threshold:{' '}
                                    <span className={benchmarkQuality?.thresholds.original_ok ? 'text-success' : 'text-warning'}>
                                        {benchmarkQuality?.thresholds.original_ok ? 'OK' : 'Insufficient'}
                                    </span>{' '}
                                    (n={benchmarkQuality?.thresholds.original_n ?? 0})
                                </div>
                                <div>
                                    EMS threshold:{' '}
                                    <span className={benchmarkQuality?.thresholds.ems_ok ? 'text-success' : 'text-warning'}>
                                        {benchmarkQuality?.thresholds.ems_ok ? 'OK' : 'Insufficient'}
                                    </span>{' '}
                                    (n={benchmarkQuality?.thresholds.ems_n ?? 0})
                                </div>
                                <div>
                                    Peer fallback used:{' '}
                                    <span className={benchmarkFallbackUsed ? 'text-info' : 'text-base-content/70'}>
                                        {benchmarkFallbackUsed ? `Yes (related ${benchmarkFallbackRelatedCount}, market ${benchmarkFallbackMarketCount})` : 'No'}
                                    </span>
                                    {benchmarkFallbackUsed ? `, benchmark tickers=${benchmarkFallbackTotal}` : ''}
                                </div>
                                {lowPeerSectors.length > 0 ? (
                                    <div className="mt-2">
                                        <div>Sectors still below peer minimum after fallback (&lt;{lowPeerRequired}):</div>
                                        <ul className="list-disc pl-4 mt-1 space-y-0.5">
                                            {lowPeerSectors.map((sector) => (
                                                <li key={`${sector.sector}-${sector.peer_count}`}>
                                                    {sector.sector}: {sector.peer_count} peer(s), shortfall {sector.shortfall}
                                                </li>
                                            ))}
                                        </ul>
                                    </div>
                                ) : (
                                    <div className="mt-2">Peer fallback coverage: OK</div>
                                )}
                            </div>
                        ) : null}
                    </div>
                    <span className="badge badge-outline">
                        Majority cutoff: {screenerStats?.majority_cutoff ?? 'N/A'}
                    </span>
                    <span className={`badge ${((screenerStats?.off_majority_cutoff_count ?? 0) > 0) ? 'badge-warning' : 'badge-outline'}`}>
                        Off-majority cutoff: {screenerStats?.off_majority_cutoff_count ?? 0}
                    </span>
                    <span className="badge badge-outline">
                        Original Z: D&lt;{formatNumber(screenerStats?.thresholds.original.distress ?? null, 2)} / S&gt;{formatNumber(screenerStats?.thresholds.original.safe ?? null, 2)}
                    </span>
                    <span className="badge badge-outline">
                        EMS Z: D&lt;{formatNumber(screenerStats?.thresholds.ems.distress ?? null, 2)} / S&gt;{formatNumber(screenerStats?.thresholds.ems.safe ?? null, 2)}
                    </span>
                </div>

                {(sourceLoading || benchmarkLoading) && (
                    <div className="flex flex-col items-center justify-center h-24 gap-2">
                        <span className="loading loading-spinner loading-md text-primary"></span>
                        <p className="text-sm text-base-content/70">
                            Loading source universe...
                        </p>
                    </div>
                )}

                <div className="dashboard-adaptive-table-wrap border border-base-300 rounded-lg">
                    <table className="table table-xs table-pin-rows w-max min-w-full">
                        <thead>
                            <tr>
                                <th className="align-top">{renderSortableHeader('ticker')}</th>
                                <th className="align-top">{renderSortableHeader('sector')}</th>
                                <th className="align-top">{renderStaticHeader('Flags')}</th>
                                <th className="align-top">{renderSortableHeader('market_cap')}</th>
                                <th className="align-top">{renderSortableHeader('z_score')}</th>
                                <th className="align-top">{renderSortableHeader('z_zone_base')}</th>
                                <th className="align-top">{renderStaticHeader('Z Adj')}</th>
                                <th className="align-top">{renderSortableHeader('z_pct')}</th>
                                <th className="align-top">{renderSortableHeader('vf_score')}</th>
                                <th className="align-top">{renderSortableHeader('vf_pct')}</th>
                                <th className="align-top">{renderSortableHeader('de')}</th>
                                <th className="align-top">{renderSortableHeader('leverage')}</th>
                                <th className="align-top">{renderSortableHeader('rating')}</th>
                                <th className="align-top">{renderSortableHeader('quality')}</th>
                            </tr>
                        </thead>
                        <tbody>
                            {sortedRows.map((row) => {
                                const isInPortfolio = portfolioTickerSet.has(row.ticker.toUpperCase());
                                const tickerTooltip = row.company_name || row.ticker;
                                return (
                                    <tr key={row.ticker} className="hover">
                                        <td className="w-20">
                                            <div className="tooltip tooltip-right" data-tip={tickerTooltip}>
                                                <span className={`font-semibold ${isInPortfolio ? 'text-accent' : ''}`}>
                                                    {row.ticker}
                                                </span>
                                            </div>
                                        </td>
                                        <td
                                            className="max-w-48 truncate"
                                            title={row.classification.sector_family
                                                ? `${row.sector} | Family: ${row.classification.sector_family}`
                                                : row.sector}
                                        >
                                            {row.sector}
                                        </td>
                                        <td>
                                            <div className="flex flex-wrap gap-1">
                                                {row.classification.is_financial ? <span className="badge badge-warning badge-xs">Financial</span> : null}
                                                {row.classification.is_manufacturing ? <span className="badge badge-info badge-xs">Manufacturing</span> : null}
                                                {row.classification.is_soe ? <span className="badge badge-success badge-xs">SOE</span> : null}
                                                {!row.classification.is_soe && row.classification.partial_soe ? (
                                                    <span className="badge badge-accent badge-xs">SOE 10-30%</span>
                                                ) : null}
                                                {row.scores.cutoff_mismatch_majority ? (
                                                    <span
                                                        className="badge badge-error badge-xs"
                                                        title={`Ticker cutoff ${row.scores.cutoff_label ?? 'N/A'} differs from majority ${screenerStats?.majority_cutoff ?? 'N/A'}`}
                                                    >
                                                        Cutoff!=Maj
                                                    </span>
                                                ) : null}
                                            </div>
                                        </td>
                                        <td>{formatCompact(row.snapshot.market_cap_vnd)}</td>
                                        <td>
                                            {row.scores.z_model === 'N/A'
                                                ? 'N/A'
                                                : `${formatNumber(row.scores.z_score, 2)} (${row.scores.z_model})`}
                                        </td>
                                        <td>{row.scores.z_zone_base || '-'}</td>
                                        <td>{row.scores.z_zone_adjusted || '-'}</td>
                                        <td>{formatPercentile(row.scores.z_sector_pctile)}</td>
                                        <td className="font-semibold">{row.scores.vf_score}</td>
                                        <td title={`Peers: ${row.scores.vf_peer_group} (${row.scores.vf_peer_size})`}>
                                            {formatPercentile(row.scores.vf_sector_pctile)}
                                        </td>
                                        <td>{formatNumber(row.snapshot.debt_to_equity, 2)}</td>
                                        <td>{row.scores.leverage_flag}</td>
                                        <td>
                                            <div className="flex items-center gap-1">
                                                <span className={`badge badge-sm ${getVnHealthToneClasses(row.scores.vn_health_rating_base)}`}>
                                                    {row.scores.vn_health_rating_base}
                                                </span>
                                                {row.scores.vn_health_rating !== row.scores.vn_health_rating_base ? (
                                                    <span className="badge badge-xs badge-outline" title={`Final with SOE modifier: ${row.scores.vn_health_rating}`}>
                                                        SOE↑
                                                    </span>
                                                ) : null}
                                            </div>
                                        </td>
                                        <td>
                                            <span className={`badge badge-sm ${
                                                row.snapshot.data_quality === 'Complete'
                                                    ? 'badge-success'
                                                    : row.snapshot.data_quality === 'Partial'
                                                        ? 'badge-warning'
                                                        : 'badge-error'
                                            }`}
                                            >
                                                {row.snapshot.data_quality}
                                            </span>
                                        </td>
                                    </tr>
                                );
                            })}
                            {sortedRows.length === 0 ? (
                                <tr>
                                    <td colSpan={14} className="text-center text-base-content/60 py-6">
                                        {screenerProgress.running
                                            ? 'Computing screener rows...'
                                            : 'No stocks match current filters.'}
                                    </td>
                                </tr>
                            ) : null}
                        </tbody>
                    </table>
                </div>

                <div className="text-xs text-base-content/60 flex flex-wrap gap-3">
                    <span>Rows: {sortedRows.length}</span>
                    <span>Displayed Market Cap Sum: {formatCompact(sortedRows.reduce((sum, row) => {
                        const value = row.snapshot.market_cap_vnd;
                        if (value === null) return sum;
                        return sum + value;
                    }, 0))}</span>
                    <span>Failed Tickers: {screenerProgress.failed}</span>
                </div>
            </div>
        </div>
    );
};

export default FinancialHealthScreener;
