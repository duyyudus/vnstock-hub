import type { IndustryInfo, Stock } from '../../../api/stockApi';

type RawRow = Record<string, unknown>;

export interface MarginTrendTickerBundle {
    income: RawRow[];
    ratios: RawRow[];
    errors?: string[];
}

export type TrajectorySignal = 'Strong' | 'Improving' | 'Neutral' | 'Compressing' | 'Weak';
export type BenchmarkSource = 'industry' | 'family' | 'market' | 'none';

export interface MarginTrendSeriesPoint {
    year: number;
    quarter: number;
    periodKey: string;
    periodLabel: string;
    revenueVnd: number | null;
    revenueBn: number | null;
    revenueYoy: number | null;
    revenueQoq: number | null;
    netMargin: number | null;
    roe: number | null;
}

export interface MarginTrendBenchmarkSeriesPoint {
    year: number;
    quarter: number;
    periodKey: string;
    periodLabel: string;
    netMarginMedian: number | null;
    roeMedian: number | null;
}

export interface MarginTrendBenchmarkContext {
    source: BenchmarkSource;
    peer_count: number;
    series: MarginTrendBenchmarkSeriesPoint[];
}

export interface MarginTrendRow {
    ticker: string;
    company_name: string;
    industry: string;
    family_name: string | null;
    latest_quarter: string | null;
    latest_revenue_bn: number | null;
    latest_net_margin: number | null;
    latest_roe: number | null;
    latest_revenue_yoy: number | null;
    margin_change_yoy: number | null;
    signal: TrajectorySignal;
    signal_strength: number;
    series: MarginTrendSeriesPoint[];
    benchmark: MarginTrendBenchmarkContext;
    has_partial_errors: boolean;
    errors: string[];
}

export interface MarginTrendUniverseStats {
    benchmark_size: number;
    benchmark_loaded: number;
    display_size: number;
    display_loaded: number;
    failed_count: number;
    signal_counts: {
        strong: number;
        improving: number;
        neutral: number;
        compressing: number;
        weak: number;
    };
}

interface IndustryFamilyMeta {
    family_code: string | null;
    family_name: string | null;
    family_en_name: string | null;
}

interface BuildInput {
    benchmarkStocks: Stock[];
    displayStocks: Stock[];
    industries: IndustryInfo[];
    bundlesByTicker: Map<string, MarginTrendTickerBundle>;
    failedTickers: Set<string>;
}

interface BuildResult {
    rows: MarginTrendRow[];
    stats: MarginTrendUniverseStats;
}

interface PeriodBucket {
    year: number;
    quarter: number;
    incomeRows: RawRow[];
    ratioRows: RawRow[];
}

interface InternalRow {
    row: MarginTrendRow;
    industryKey: string;
    familyKey: string | null;
}

const UNKNOWN_INDUSTRY = 'Unclassified';
const MIN_PEERS = 5;

const FINANCIAL_KEYWORDS = [
    'ngan hang',
    'bao hiem',
    'dich vu tai chinh',
    'chung khoan',
    'bank',
    'insurance',
    'financial service',
    'securities',
    'finance',
];

const normalizeText = (value: unknown): string => {
    return String(value ?? '')
        .normalize('NFD')
        .replace(/[\u0300-\u036f]/g, '')
        .toLowerCase()
        .trim();
};

const normalizeMetricKey = (value: string): string => {
    return normalizeText(value).replace(/[^a-z0-9]/g, '');
};

const parseNumber = (value: unknown): number | null => {
    if (typeof value === 'number') {
        return Number.isFinite(value) ? value : null;
    }
    if (typeof value !== 'string') {
        return null;
    }

    const trimmed = value.trim();
    if (!trimmed) {
        return null;
    }

    const negativeWrapped = /^\(.*\)$/.test(trimmed);
    const raw = negativeWrapped ? trimmed.slice(1, -1) : trimmed;
    const cleaned = raw.replace(/[^0-9,.-]/g, '');
    if (!cleaned) {
        return null;
    }

    const commaCount = (cleaned.match(/,/g) || []).length;
    const dotCount = (cleaned.match(/\./g) || []).length;
    let normalized = cleaned;

    if (commaCount > 0 && dotCount > 0) {
        normalized = cleaned.replace(/,/g, '');
    } else if (commaCount > 0 && dotCount === 0) {
        const lastCommaIndex = cleaned.lastIndexOf(',');
        const fractionLen = lastCommaIndex >= 0 ? cleaned.length - lastCommaIndex - 1 : 0;
        const useThousandsSeparator = commaCount > 1 || fractionLen === 3 || fractionLen === 0;
        normalized = useThousandsSeparator ? cleaned.replace(/,/g, '') : cleaned.replace(',', '.');
    } else if (dotCount > 1 && commaCount === 0) {
        normalized = cleaned.replace(/\./g, '');
    }

    const parsed = Number(normalized);
    if (!Number.isFinite(parsed)) {
        return null;
    }
    return negativeWrapped ? -Math.abs(parsed) : parsed;
};

const safeDiv = (numerator: number | null, denominator: number | null): number | null => {
    if (numerator === null || denominator === null || denominator === 0) {
        return null;
    }
    const value = numerator / denominator;
    return Number.isFinite(value) ? value : null;
};

const asRate = (value: number | null): number | null => {
    if (value === null || !Number.isFinite(value)) {
        return null;
    }
    if (Math.abs(value) > 2) {
        return value / 100;
    }
    return value;
};

const readYear = (row: RawRow): number | null => {
    const year = parseNumber(row.yearReport ?? row.Meta_yearReport ?? row.year ?? row.reporting_year);
    if (!year) return null;
    return Math.round(year);
};

const readQuarter = (row: RawRow): number | null => {
    const quarter = parseNumber(row.lengthReport ?? row.Meta_lengthReport ?? row.quarter ?? row.period_quarter);
    if (!quarter) return null;
    const q = Math.round(quarter);
    if (q < 1 || q > 4) return null;
    return q;
};

const normalizeDataRows = (rows: RawRow[]): Array<Record<string, { key: string; value: number }>> => {
    return rows.map((row) => {
        const result: Record<string, { key: string; value: number }> = {};
        Object.entries(row).forEach(([key, raw]) => {
            if (key === 'ticker' || key.startsWith('Meta_')) {
                return;
            }
            const normalizedKey = normalizeMetricKey(key);
            if (!normalizedKey) {
                return;
            }
            const value = parseNumber(raw);
            if (value === null) {
                return;
            }
            result[normalizedKey] = { key, value };
        });
        return result;
    });
};

const pickMetric = (
    rows: Array<Record<string, { key: string; value: number }>>,
    aliases: string[],
    excludes: string[] = [],
): number | null => {
    let bestValue: number | null = null;
    let bestScore = -1;

    rows.forEach((row) => {
        Object.entries(row).forEach(([key, valueObj]) => {
            if (excludes.some((exclude) => key.includes(exclude))) {
                return;
            }
            aliases.forEach((alias) => {
                if (!key.includes(alias)) {
                    return;
                }
                let score = alias.length * 1000;
                if (key === alias) score += 600;
                if (key.startsWith(alias)) score += 300;
                if (key.endsWith(alias)) score += 120;
                score -= Math.max(0, key.length - alias.length);
                if (score > bestScore) {
                    bestScore = score;
                    bestValue = valueObj.value;
                }
            });
        });
    });

    return bestValue;
};

const comparePeriodAsc = (
    left: { year: number; quarter: number },
    right: { year: number; quarter: number },
): number => {
    if (left.year !== right.year) {
        return left.year - right.year;
    }
    return left.quarter - right.quarter;
};

const toPeriodKey = (year: number, quarter: number): string => {
    return `${year}-Q${quarter}`;
};

const toPeriodLabel = (year: number, quarter: number): string => {
    return `${year} Q${quarter}`;
};

const isFinancialIndustry = (industry: string): boolean => {
    const normalized = normalizeText(industry);
    if (!normalized) {
        return false;
    }
    return FINANCIAL_KEYWORDS.some((keyword) => normalized.includes(keyword));
};

const toVndAmount = (value: number | null): number | null => {
    if (value === null || !Number.isFinite(value)) {
        return null;
    }
    return Math.abs(value) < 1_000_000_000 ? value * 1_000_000_000 : value;
};

const toBillionsVnd = (value: number | null): number | null => {
    if (value === null || !Number.isFinite(value)) {
        return null;
    }
    return value / 1_000_000_000;
};

const median = (values: number[]): number | null => {
    if (values.length === 0) {
        return null;
    }
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    if (sorted.length % 2 === 1) {
        return sorted[mid];
    }
    return (sorted[mid - 1] + sorted[mid]) / 2;
};

const toSignalStrength = (signal: TrajectorySignal): number => {
    switch (signal) {
        case 'Strong':
            return 5;
        case 'Improving':
            return 4;
        case 'Neutral':
            return 3;
        case 'Compressing':
            return 2;
        case 'Weak':
            return 1;
        default:
            return 3;
    }
};

const computeTrajectorySignal = (marginChangeYoy: number | null, revenueYoy: number | null): TrajectorySignal => {
    if (marginChangeYoy === null || revenueYoy === null) {
        return 'Neutral';
    }
    if (marginChangeYoy > 0 && revenueYoy > 0.1) {
        return 'Strong';
    }
    if (marginChangeYoy > 0 && revenueYoy >= 0) {
        return 'Improving';
    }
    if (marginChangeYoy < 0 && revenueYoy < 0) {
        return 'Weak';
    }
    if (marginChangeYoy < 0 && revenueYoy >= 0) {
        return 'Compressing';
    }
    return 'Neutral';
};

const buildIndustryFamilyMap = (industries: IndustryInfo[]): Map<string, IndustryFamilyMeta> => {
    const map = new Map<string, IndustryFamilyMeta>();
    industries.forEach((industry) => {
        const key = normalizeText(industry.name ?? '');
        if (!key) {
            return;
        }
        map.set(key, {
            family_code: industry.family_code ?? null,
            family_name: industry.family_name ?? null,
            family_en_name: industry.family_en_name ?? null,
        });
    });
    return map;
};

const toFamilyKey = (meta: IndustryFamilyMeta | null): string | null => {
    if (!meta) {
        return null;
    }
    const codeKey = normalizeText(meta.family_code ?? '');
    if (codeKey) {
        return codeKey;
    }
    const nameKey = normalizeText(meta.family_name ?? meta.family_en_name ?? '');
    return nameKey || null;
};

const buildTickerSeries = (industry: string, bundle: MarginTrendTickerBundle): MarginTrendSeriesPoint[] => {
    const periods = new Map<string, PeriodBucket>();

    const upsertPeriod = (year: number, quarter: number): PeriodBucket => {
        const key = toPeriodKey(year, quarter);
        const existing = periods.get(key);
        if (existing) {
            return existing;
        }
        const created: PeriodBucket = {
            year,
            quarter,
            incomeRows: [],
            ratioRows: [],
        };
        periods.set(key, created);
        return created;
    };

    bundle.income.forEach((row) => {
        const year = readYear(row);
        const quarter = readQuarter(row);
        if (!year || !quarter) {
            return;
        }
        const bucket = upsertPeriod(year, quarter);
        bucket.incomeRows.push(row);
    });

    bundle.ratios.forEach((row) => {
        const year = readYear(row);
        const quarter = readQuarter(row);
        if (!year || !quarter) {
            return;
        }
        const bucket = upsertPeriod(year, quarter);
        bucket.ratioRows.push(row);
    });

    const isFinancial = isFinancialIndustry(industry);

    const points = Array.from(periods.values())
        .sort(comparePeriodAsc)
        .filter((bucket) => bucket.incomeRows.length > 0)
        .map((bucket) => {
            const incomeRows = normalizeDataRows(bucket.incomeRows);
            const ratioRows = normalizeDataRows(bucket.ratioRows);

            const revenueRaw = isFinancial
                ? (
                    pickMetric(incomeRows, ['totaloperatingincome'], ['yoy'])
                    ?? pickMetric(incomeRows, ['totaloperatingrevenue'], ['yoy'])
                    ?? pickMetric(incomeRows, ['revenue'], ['yoy'])
                )
                : (
                    pickMetric(incomeRows, ['netsales'], ['yoy'])
                    ?? pickMetric(incomeRows, ['totaloperatingrevenue'], ['yoy'])
                    ?? pickMetric(incomeRows, ['revenue'], ['yoy'])
                );

            const netIncome = pickMetric(
                incomeRows,
                ['netincome', 'profitaftertax', 'loinhuansauthue', 'earningsaftertax', 'netprofit'],
                ['associated', 'minority'],
            );

            const ratioNetMargin = asRate(
                pickMetric(
                    ratioRows,
                    ['aftertaxprofitmargin', 'netprofitmargin', 'netmargin', 'npm', 'profitmargin'],
                    ['gross', 'operating', 'ebit', 'beforetax', 'pretax', 'yoy'],
                ),
            );

            const derivedMargin = safeDiv(netIncome, revenueRaw);
            const netMargin = ratioNetMargin ?? derivedMargin;

            const roe = asRate(
                pickMetric(
                    ratioRows,
                    ['returnonaverageequity', 'returnonequity', 'roe'],
                    ['yoy'],
                ),
            );

            const revenueVnd = toVndAmount(revenueRaw);

            return {
                year: bucket.year,
                quarter: bucket.quarter,
                periodKey: toPeriodKey(bucket.year, bucket.quarter),
                periodLabel: toPeriodLabel(bucket.year, bucket.quarter),
                revenueVnd,
                revenueBn: toBillionsVnd(revenueVnd),
                revenueYoy: null,
                revenueQoq: null,
                netMargin,
                roe,
            } as MarginTrendSeriesPoint;
        });

    const pointByPeriod = new Map<string, MarginTrendSeriesPoint>(
        points.map((point) => [point.periodKey, point]),
    );

    points.forEach((point, index) => {
        const prev = index > 0 ? points[index - 1] : null;
        if (prev) {
            point.revenueQoq = safeDiv(
                point.revenueVnd !== null && prev.revenueVnd !== null
                    ? point.revenueVnd - prev.revenueVnd
                    : null,
                prev.revenueVnd,
            );
        }
        const yoyBase = pointByPeriod.get(toPeriodKey(point.year - 1, point.quarter)) ?? null;
        const yoyBaseRevenue = yoyBase !== null && yoyBase.revenueVnd !== null ? yoyBase.revenueVnd : null;
        point.revenueYoy = safeDiv(
            point.revenueVnd !== null && yoyBaseRevenue !== null
                ? point.revenueVnd - yoyBaseRevenue
                : null,
            yoyBaseRevenue,
        );
    });

    return points;
};

const computeRow = (
    stock: Stock,
    bundle: MarginTrendTickerBundle,
    hasPartialErrors: boolean,
    industryFamilyMap: Map<string, IndustryFamilyMeta>,
): InternalRow => {
    const ticker = stock.ticker.toUpperCase();
    const industry = stock.industry || UNKNOWN_INDUSTRY;
    const industryKey = normalizeText(industry) || normalizeText(UNKNOWN_INDUSTRY);
    const familyMeta = industryFamilyMap.get(industryKey) ?? null;
    const familyKey = toFamilyKey(familyMeta);
    const familyName = familyMeta?.family_name ?? familyMeta?.family_en_name ?? null;

    const series = buildTickerSeries(industry, bundle);
    const latest = series.length > 0 ? series[series.length - 1] : null;
    const latestQuarterLabel = latest?.periodLabel ?? null;

    const marginYoyBase = latest
        ? series.find((point) => point.year === latest.year - 1 && point.quarter === latest.quarter)
        : null;

    const marginChangeYoy = latest && marginYoyBase
        ? (
            latest.netMargin !== null && marginYoyBase.netMargin !== null
                ? latest.netMargin - marginYoyBase.netMargin
                : null
        )
        : null;

    const latestRevenueYoy = latest?.revenueYoy ?? null;
    const signal = computeTrajectorySignal(marginChangeYoy, latestRevenueYoy);

    const row: MarginTrendRow = {
        ticker,
        company_name: stock.company_name,
        industry,
        family_name: familyName,
        latest_quarter: latestQuarterLabel,
        latest_revenue_bn: latest?.revenueBn ?? null,
        latest_net_margin: latest?.netMargin ?? null,
        latest_roe: latest?.roe ?? null,
        latest_revenue_yoy: latestRevenueYoy,
        margin_change_yoy: marginChangeYoy,
        signal,
        signal_strength: toSignalStrength(signal),
        series,
        benchmark: {
            source: 'none',
            peer_count: 0,
            series: [],
        },
        has_partial_errors: hasPartialErrors,
        errors: [...(bundle.errors ?? []), ...(hasPartialErrors ? ['partial endpoint failures'] : [])],
    };

    return {
        row,
        industryKey,
        familyKey,
    };
};

const buildMedianSeries = (peers: InternalRow[]): MarginTrendBenchmarkSeriesPoint[] => {
    const periodMap = new Map<string, {
        year: number;
        quarter: number;
        periodLabel: string;
        netMarginValues: number[];
        roeValues: number[];
    }>();

    peers.forEach((peer) => {
        peer.row.series.forEach((point) => {
            const existing = periodMap.get(point.periodKey) ?? {
                year: point.year,
                quarter: point.quarter,
                periodLabel: point.periodLabel,
                netMarginValues: [],
                roeValues: [],
            };
            if (point.netMargin !== null) {
                existing.netMarginValues.push(point.netMargin);
            }
            if (point.roe !== null) {
                existing.roeValues.push(point.roe);
            }
            periodMap.set(point.periodKey, existing);
        });
    });

    return Array.from(periodMap.entries())
        .map(([periodKey, values]) => ({
            year: values.year,
            quarter: values.quarter,
            periodKey,
            periodLabel: values.periodLabel,
            netMarginMedian: median(values.netMarginValues),
            roeMedian: median(values.roeValues),
        }))
        .filter((point) => point.netMarginMedian !== null || point.roeMedian !== null)
        .sort(comparePeriodAsc);
};

export const buildMarginTrendScreen = (input: BuildInput): BuildResult => {
    const industryFamilyMap = buildIndustryFamilyMap(input.industries);

    const benchmarkTickerSet = new Set(input.benchmarkStocks.map((stock) => stock.ticker.toUpperCase()));
    const displayTickerSet = new Set(input.displayStocks.map((stock) => stock.ticker.toUpperCase()));
    const allStocksByTicker = new Map<string, Stock>([
        ...input.benchmarkStocks.map((stock) => [stock.ticker.toUpperCase(), stock] as const),
        ...input.displayStocks.map((stock) => [stock.ticker.toUpperCase(), stock] as const),
    ]);

    const computedByTicker = new Map<string, InternalRow>();
    allStocksByTicker.forEach((stock, ticker) => {
        const bundle = input.bundlesByTicker.get(ticker);
        if (!bundle) {
            const empty: MarginTrendTickerBundle = { income: [], ratios: [] };
            computedByTicker.set(ticker, computeRow(stock, empty, input.failedTickers.has(ticker), industryFamilyMap));
            return;
        }
        computedByTicker.set(ticker, computeRow(stock, bundle, input.failedTickers.has(ticker), industryFamilyMap));
    });

    const benchmarkRows = Array.from(benchmarkTickerSet)
        .map((ticker) => computedByTicker.get(ticker))
        .filter((row): row is InternalRow => Boolean(row));

    const industryPeers = new Map<string, InternalRow[]>();
    const familyPeers = new Map<string, InternalRow[]>();
    const marketPeers = benchmarkRows.filter((item) => item.row.series.length > 0);

    benchmarkRows.forEach((item) => {
        if (item.row.series.length === 0) {
            return;
        }
        const industryList = industryPeers.get(item.industryKey) ?? [];
        industryList.push(item);
        industryPeers.set(item.industryKey, industryList);

        if (item.familyKey) {
            const familyList = familyPeers.get(item.familyKey) ?? [];
            familyList.push(item);
            familyPeers.set(item.familyKey, familyList);
        }
    });

    const industryMedianCache = new Map<string, MarginTrendBenchmarkContext>();
    industryPeers.forEach((peers, key) => {
        industryMedianCache.set(key, {
            source: 'industry',
            peer_count: peers.length,
            series: buildMedianSeries(peers),
        });
    });

    const familyMedianCache = new Map<string, MarginTrendBenchmarkContext>();
    familyPeers.forEach((peers, key) => {
        familyMedianCache.set(key, {
            source: 'family',
            peer_count: peers.length,
            series: buildMedianSeries(peers),
        });
    });

    const marketMedianContext: MarginTrendBenchmarkContext = {
        source: 'market',
        peer_count: marketPeers.length,
        series: buildMedianSeries(marketPeers),
    };

    const assignBenchmarkContext = (item: InternalRow): MarginTrendBenchmarkContext => {
        const industryContext = industryMedianCache.get(item.industryKey);
        if (industryContext && industryContext.peer_count >= MIN_PEERS) {
            return industryContext;
        }
        if (item.familyKey) {
            const familyContext = familyMedianCache.get(item.familyKey);
            if (familyContext && familyContext.peer_count >= MIN_PEERS) {
                return familyContext;
            }
        }
        if (marketMedianContext.peer_count > 0) {
            return marketMedianContext;
        }
        return {
            source: 'none',
            peer_count: 0,
            series: [],
        };
    };

    const rows = Array.from(displayTickerSet)
        .map((ticker) => computedByTicker.get(ticker))
        .filter((item): item is InternalRow => Boolean(item))
        .map((item) => ({
            ...item.row,
            benchmark: assignBenchmarkContext(item),
        }));

    const stats: MarginTrendUniverseStats = {
        benchmark_size: input.benchmarkStocks.length,
        benchmark_loaded: benchmarkRows.filter((item) => item.row.series.length > 0).length,
        display_size: input.displayStocks.length,
        display_loaded: rows.filter((row) => row.series.length > 0).length,
        failed_count: input.failedTickers.size,
        signal_counts: {
            strong: rows.filter((row) => row.signal === 'Strong').length,
            improving: rows.filter((row) => row.signal === 'Improving').length,
            neutral: rows.filter((row) => row.signal === 'Neutral').length,
            compressing: rows.filter((row) => row.signal === 'Compressing').length,
            weak: rows.filter((row) => row.signal === 'Weak').length,
        },
    };

    return {
        rows,
        stats,
    };
};
