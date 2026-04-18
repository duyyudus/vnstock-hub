import type { Stock } from '../../../api/stockApi';
import {
    DEFAULT_SECTOR_BENCHMARK,
    VALUATION_SECTOR_BENCHMARKS,
    type SectorBenchmark,
} from './valuationBenchmarks';

type RawRow = Record<string, unknown>;

export interface ValuationTickerBundle {
    ratios: RawRow[];
    income: RawRow[];
    cashflow: RawRow[];
    volume_history: RawRow[];
    price_history: RawRow[];
    errors?: string[];
}

export type ValuationDataCompleteness = 'Complete' | 'Partial' | 'Estimated';

export interface ValuationRawMetrics {
    price_vnd: number | null;
    market_cap_bn: number | null;
    avg_vol_20d_m: number | null;
    pe: number | null;
    pb: number | null;
    ps: number | null;
    eps_vnd: number | null;
    roe: number | null;
    roa: number | null;
    npm: number | null;
    bvps_vnd: number | null;
    rev_cagr_3y: number | null;
    profit_cagr_3y: number | null;
    rev_yoy: number | null;
    profit_yoy: number | null;
    de: number | null;
    current_ratio: number | null;
    cfo_np: number | null;
    momentum_1m: number | null;
    momentum_6m: number | null;
    momentum_1y: number | null;
}

export interface ValuationScores {
    pe_score: number;
    pb_score: number;
    roe_score: number;
    roa_score: number;
    growth_score: number;
    stability_score: number;
    valuation_score: number;
    quality_score: number;
    overall_score: number;
    rank: number;
    quadrant: string;
    verdict: string;
}

export interface ValuationDerivedFlags {
    used_price_fallback: boolean;
    used_market_cap_fallback: boolean;
    used_pe_fallback: boolean;
}

export interface ValuationRow {
    ticker: string;
    company_name: string;
    sector: string;
    metrics: ValuationRawMetrics;
    scores: ValuationScores;
    flags: ValuationDerivedFlags;
    data_completeness: ValuationDataCompleteness;
    errors: string[];
}

export interface ValuationUniverseStats {
    benchmark_size: number;
    benchmark_loaded: number;
    display_size: number;
    display_loaded: number;
    failed_count: number;
    fallback_price_count: number;
    fallback_market_cap_count: number;
    fallback_pe_count: number;
    partial_count: number;
    estimated_count: number;
}

export interface BuildValuationInput {
    benchmarkStocks: Stock[];
    displayStocks: Stock[];
    bundlesByTicker: Map<string, ValuationTickerBundle>;
    failedTickers: Set<string>;
}

export interface BuildValuationResult {
    rows: ValuationRow[];
    stats: ValuationUniverseStats;
}

const UNKNOWN_SECTOR = 'Unclassified';

const normalizeText = (value: unknown): string => {
    return String(value ?? '')
        .normalize('NFD')
        .replace(/[\u0300-\u036f]/g, '')
        .toLowerCase()
        .trim();
};

const SECTOR_BENCHMARKS = new Map<string, SectorBenchmark>(
    VALUATION_SECTOR_BENCHMARKS.map(([name, benchmark]) => [normalizeText(name), benchmark]),
);

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

const findLatestPeriod = (rows: RawRow[]): { year: number; quarter: number } | null => {
    let best: { year: number; quarter: number } | null = null;
    rows.forEach((row) => {
        const year = readYear(row);
        const quarter = readQuarter(row);
        if (!year || !quarter) return;
        if (!best || year > best.year || (year === best.year && quarter > best.quarter)) {
            best = { year, quarter };
        }
    });
    return best;
};

const aggregateRowsToYearCutoff = (rows: RawRow[], year: number, quarterCutoff: number): Record<string, number> => {
    const totals: Record<string, number> = {};
    rows.forEach((row) => {
        const rowYear = readYear(row);
        const rowQuarter = readQuarter(row);
        if (rowYear !== year || rowQuarter === null || rowQuarter > quarterCutoff) {
            return;
        }
        Object.entries(row).forEach(([key, raw]) => {
            if (key === 'ticker' || key.startsWith('Meta_')) return;
            const normalizedKey = normalizeMetricKey(key);
            if (!normalizedKey) return;
            const value = parseNumber(raw);
            if (value === null) return;
            totals[normalizedKey] = (totals[normalizedKey] ?? 0) + value;
        });
    });
    return totals;
};

const pickRowsAtOrBeforeQuarter = (rows: RawRow[], year: number, quarterTarget: number): RawRow[] => {
    let bestQuarter = -1;
    const candidates: RawRow[] = [];

    rows.forEach((row) => {
        const rowYear = readYear(row);
        const rowQuarter = readQuarter(row);
        if (rowYear !== year || rowQuarter === null || rowQuarter > quarterTarget) {
            return;
        }

        if (rowQuarter > bestQuarter) {
            bestQuarter = rowQuarter;
            candidates.length = 0;
            candidates.push(row);
            return;
        }

        if (rowQuarter === bestQuarter) {
            candidates.push(row);
        }
    });

    return candidates;
};

const normalizeDataRows = (rows: RawRow[]): Array<Record<string, { key: string; value: number }>> => {
    return rows.map((row) => {
        const result: Record<string, { key: string; value: number }> = {};
        Object.entries(row).forEach(([key, raw]) => {
            if (key === 'ticker' || key.startsWith('Meta_')) return;
            const normalizedKey = normalizeMetricKey(key);
            if (!normalizedKey) return;
            const value = parseNumber(raw);
            if (value === null) return;
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
            if (excludes.some((ex) => key.includes(ex))) {
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

const pickAggMetric = (agg: Record<string, number>, aliases: string[], excludes: string[] = []): number | null => {
    return pickMetric(
        [Object.fromEntries(Object.entries(agg).map(([key, value]) => [key, { key, value }]))],
        aliases,
        excludes,
    );
};

const toLatestDbCloseVnd = (rows: RawRow[]): number | null => {
    let latestDate = '';
    let latestClose: number | null = null;
    rows.forEach((row) => {
        const date = String(row.date ?? '');
        const close = parseNumber(row.close);
        if (!date || close === null) return;
        if (date >= latestDate) {
            latestDate = date;
            latestClose = close;
        }
    });
    if (latestClose === null) return null;
    return latestClose >= 1000 ? latestClose : latestClose * 1000;
};

const normalizeSharesOutstanding = (value: number | null): number | null => {
    if (value === null || !Number.isFinite(value) || value <= 0) {
        return null;
    }
    return value >= 1_000_000 ? value : value * 1_000_000;
};

const computeAverageVolume20M = (rows: RawRow[]): number | null => {
    const volumes = rows
        .map((row) => parseNumber(row.volume))
        .filter((value): value is number => value !== null && value > 0);
    if (volumes.length === 0) {
        return null;
    }
    const tail = volumes.slice(-20);
    const avg = tail.reduce((sum, value) => sum + value, 0) / tail.length;
    return avg / 1_000_000;
};

const cagr = (current: number | null, base: number | null, years: number): number | null => {
    if (current === null || base === null || years <= 0) return null;
    if (base <= 0 || current <= 0) return null;
    const growth = Math.pow(current / base, 1 / years) - 1;
    return Number.isFinite(growth) ? growth : null;
};

const getSectorBenchmark = (sector: string): SectorBenchmark => {
    return SECTOR_BENCHMARKS.get(normalizeText(sector)) ?? DEFAULT_SECTOR_BENCHMARK;
};

const scorePeOrPb = (value: number | null, good: number, acceptable: number): number => {
    if (value === null || !Number.isFinite(value) || value <= 0 || good <= 0 || acceptable <= 0 || acceptable <= good) {
        return 0;
    }
    if (value <= good) {
        return Math.max(95, 100 - (5 * (good - value)) / good);
    }
    if (value <= acceptable) {
        return 100 - (40 * (value - good)) / (acceptable - good);
    }
    return Math.max(0, 60 - 60 * Math.min((value - acceptable) / acceptable, 1));
};

const scoreRoeOrRoa = (value: number | null, good: number, acceptable: number): number => {
    if (value === null || !Number.isFinite(value) || good <= 0 || acceptable <= 0 || acceptable > good) {
        return 0;
    }
    if (value >= good) {
        return Math.min(105, 100 + (5 * (value - good)) / good);
    }
    if (value >= acceptable) {
        return 60 + (40 * (value - acceptable)) / (good - acceptable);
    }
    return Math.max(0, (60 * value) / Math.max(acceptable, 0.001));
};

const scoreGrowth = (revCagr: number | null, profitCagr: number | null): number => {
    const compute = (value: number | null, clampHigh: number | null = null): number | null => {
        if (value === null || !Number.isFinite(value)) return null;
        const bounded = Math.max(clampHigh === null ? value : Math.min(value, clampHigh), -0.95);
        const score = 50 + Math.log(1 + bounded) * 100;
        return Math.min(100, Math.max(0, score));
    };

    const revScore = compute(revCagr);
    const profitScore = compute(profitCagr, 1.5);
    if (revScore === null || profitScore === null) {
        return 50;
    }
    return 0.5 * revScore + 0.5 * profitScore;
};

const scoreStability = (sector: string, de: number | null, cfoNp: number | null, currentRatio: number | null): number => {
    const normalizedSector = normalizeText(sector);
    const isBank = normalizedSector.includes('ngan hang') || normalizedSector.includes('bank');

    const cfoScore = (() => {
        if (cfoNp === null || !Number.isFinite(cfoNp)) return null;
        if (cfoNp >= 1) return Math.min(105, 100 + 5 * (cfoNp - 1));
        if (cfoNp >= 0.5) return 60 + 80 * (cfoNp - 0.5);
        return Math.max(0, 120 * cfoNp);
    })();

    const deScore = (() => {
        if (de === null || !Number.isFinite(de)) return null;
        if (isBank) {
            if (de <= 10) return 100;
            if (de <= 15) return 80;
            if (de <= 20) return 60;
            return 30;
        }
        if (de <= 0.5) return 100;
        if (de <= 1) return 80;
        if (de <= 2) return 60;
        return Math.max(0, 60 - 30 * (de - 2));
    })();

    const currentRatioScore = (() => {
        if (isBank) {
            return 70;
        }
        if (currentRatio === null || !Number.isFinite(currentRatio)) return null;
        if (currentRatio >= 2) return 100;
        if (currentRatio >= 1) return 60 + 40 * (currentRatio - 1);
        return Math.max(0, 60 * currentRatio);
    })();

    if (cfoScore === null || deScore === null || currentRatioScore === null) {
        return 50;
    }

    return 0.4 * cfoScore + 0.3 * deScore + 0.3 * currentRatioScore;
};

const toQuadrant = (valuationScore: number, qualityScore: number): string => {
    if (valuationScore >= 65 && qualityScore >= 65) return 'VALUE PICK';
    if (valuationScore < 65 && qualityScore >= 65) return 'Premium for Quality';
    if (valuationScore >= 65 && qualityScore < 65) return 'Cheap but Risky';
    return 'Expensive & Weak';
};

const toVerdict = (overallScore: number): string => {
    if (overallScore >= 80) return 'Strong Buy Signal';
    if (overallScore >= 70) return 'Attractive';
    if (overallScore >= 60) return 'Fair Value';
    if (overallScore >= 50) return 'Hold / Monitor';
    if (overallScore >= 40) return 'Caution';
    return 'Avoid / Overpriced';
};

const getRankMap = (rows: ValuationRow[]): Map<string, number> => {
    const sorted = [...rows].sort((a, b) => b.scores.overall_score - a.scores.overall_score);
    const rankByTicker = new Map<string, number>();
    let prevScore: number | null = null;
    let prevRank = 0;

    sorted.forEach((row, index) => {
        let rank = index + 1;
        if (prevScore !== null && row.scores.overall_score === prevScore) {
            rank = prevRank;
        }
        rankByTicker.set(row.ticker, rank);
        prevScore = row.scores.overall_score;
        prevRank = rank;
    });

    return rankByTicker;
};

const buildRow = (stock: Stock, bundle: ValuationTickerBundle | undefined, hasErrors: boolean): ValuationRow => {
    const safeBundle: ValuationTickerBundle = bundle ?? {
        ratios: [],
        income: [],
        cashflow: [],
        volume_history: [],
        price_history: [],
        errors: ['missing bundle'],
    };

    const period = findLatestPeriod(safeBundle.income)
        ?? findLatestPeriod(safeBundle.ratios)
        ?? findLatestPeriod(safeBundle.cashflow);

    const ratiosSlice = period
        ? pickRowsAtOrBeforeQuarter(safeBundle.ratios, period.year, period.quarter)
        : safeBundle.ratios;
    const ratioRows = normalizeDataRows(ratiosSlice.length > 0 ? ratiosSlice : safeBundle.ratios);

    const currentIncome = period
        ? aggregateRowsToYearCutoff(safeBundle.income, period.year, period.quarter)
        : {};
    const priorIncome1 = period
        ? aggregateRowsToYearCutoff(safeBundle.income, period.year - 1, period.quarter)
        : {};
    const priorIncome3 = period
        ? aggregateRowsToYearCutoff(safeBundle.income, period.year - 3, period.quarter)
        : {};

    const currentCashflow = period
        ? aggregateRowsToYearCutoff(safeBundle.cashflow, period.year, period.quarter)
        : {};

    const ratioPe = pickMetric(ratioRows, ['chitieudinhgiape', 'pricetoearnings', 'priceearning', 'pe'], ['peg']);
    const ratioPb = pickMetric(ratioRows, ['chitieudinhgiapb', 'pricetobook', 'pb']);
    const ratioPs = pickMetric(ratioRows, ['chitieudinhgiaps', 'pricetosales', 'ps']);
    const ratioEps = pickMetric(ratioRows, ['earningpershare', 'eps']);
    const ratioRoe = asRate(pickMetric(ratioRows, ['roe']));
    const ratioRoa = asRate(pickMetric(ratioRows, ['roa']));
    const ratioNpm = asRate(
        pickMetric(
            ratioRows,
            ['aftertaxprofitmargin', 'netprofitmargin', 'netmargin', 'npm', 'ros', 'profitmargin'],
            ['gross', 'operating', 'ebit', 'beforetax', 'pretax', 'yoy'],
        ),
    );
    const ratioBvps = pickMetric(ratioRows, ['bookvaluepershare', 'bvps']);
    const ratioDe = pickMetric(ratioRows, ['debtequity', 'debttoequity', 'detoequity']);
    const ratioCurrent = pickMetric(ratioRows, ['currentratio']);
    const ratioShares = normalizeSharesOutstanding(
        pickMetric(
            ratioRows,
            [
                'chitieudinhgiaoutstandingsharemilshares',
                'outstandingsharemilshares',
                'sharesoutstanding',
                'outstandingshare',
                'outstandingshares',
                'issueshare',
                'shareissued',
                'sharesissued',
                'numberofshares',
                'totalshares',
                'listedshares',
                'listingvolume',
            ],
        ),
    );

    const revenueCurrent = pickAggMetric(
        currentIncome,
        ['totaloperatingincome', 'totaloperatingrevenue', 'netsales', 'revenue'],
        ['yoy'],
    );
    const revenuePrior1 = pickAggMetric(
        priorIncome1,
        ['totaloperatingincome', 'totaloperatingrevenue', 'netsales', 'revenue'],
        ['yoy'],
    );
    const revenuePrior3 = pickAggMetric(
        priorIncome3,
        ['totaloperatingincome', 'totaloperatingrevenue', 'netsales', 'revenue'],
        ['yoy'],
    );

    const profitCurrent = pickAggMetric(currentIncome, ['netincome', 'profitaftertax', 'loinhuansauthue', 'netprofit']);
    const profitPrior1 = pickAggMetric(priorIncome1, ['netincome', 'profitaftertax', 'loinhuansauthue', 'netprofit']);
    const profitPrior3 = pickAggMetric(priorIncome3, ['netincome', 'profitaftertax', 'loinhuansauthue', 'netprofit']);

    const revCagr3y = cagr(revenueCurrent, revenuePrior3, 3);
    const profitCagr3y = cagr(profitCurrent, profitPrior3, 3);
    const revYoy = safeDiv(revenueCurrent !== null && revenuePrior1 !== null ? revenueCurrent - revenuePrior1 : null, revenuePrior1);
    const profitYoy = safeDiv(profitCurrent !== null && profitPrior1 !== null ? profitCurrent - profitPrior1 : null, profitPrior1);

    const cfoCurrent = pickAggMetric(
        currentCashflow,
        [
            'operatingcashflow',
            'cashflowfromoperatingactivities',
            'netcashinflowsoutflowsfromoperatingactivities',
            'netcashflowsfromoperatingactivities',
            'luuchuyentientuhdkinhdoanh',
            'netcashfromoperatingactivities',
            'cfo',
        ],
        ['increase', 'decrease'],
    );

    const cfoNp = safeDiv(cfoCurrent, profitCurrent);

    const latestDbClose = toLatestDbCloseVnd(safeBundle.price_history);
    const primaryPrice = stock.price > 0 ? stock.price : null;
    const priceVnd = primaryPrice ?? latestDbClose;

    const epsFromIncome = ratioShares !== null && profitCurrent !== null ? (profitCurrent * 1_000_000_000) / ratioShares : null;
    const epsVnd = ratioEps ?? epsFromIncome;

    const primaryMarketCap = stock.market_cap > 0 ? stock.market_cap : null;
    const fallbackMarketCap = priceVnd !== null && ratioShares !== null
        ? (priceVnd * ratioShares) / 1_000_000_000
        : null;
    const marketCapBn = primaryMarketCap ?? fallbackMarketCap;

    const primaryPe = stock.pe_ratio && stock.pe_ratio > 0 ? stock.pe_ratio : null;
    const fallbackPe = latestDbClose !== null && epsVnd !== null && epsVnd > 0
        ? latestDbClose / epsVnd
        : null;
    const pe = ratioPe ?? primaryPe ?? fallbackPe;

    const benchmark = getSectorBenchmark(stock.industry || UNKNOWN_SECTOR);

    const peScore = scorePeOrPb(pe, benchmark.pe_good, benchmark.pe_accept);
    const pbScore = scorePeOrPb(ratioPb, benchmark.pb_good, benchmark.pb_accept);
    const roeScore = scoreRoeOrRoa(ratioRoe, benchmark.roe_good, benchmark.roe_accept);
    const roaScore = scoreRoeOrRoa(ratioRoa, benchmark.roa_good, benchmark.roa_accept);
    const growthScore = scoreGrowth(revCagr3y, profitCagr3y);
    const stabilityScore = scoreStability(stock.industry || UNKNOWN_SECTOR, ratioDe, cfoNp, ratioCurrent);

    const valuationScore = peScore * benchmark.w_pe + pbScore * benchmark.w_pb;
    const qualityScore = roeScore * benchmark.w_roe
        + roaScore * benchmark.w_roa
        + growthScore * benchmark.w_growth
        + stabilityScore * benchmark.w_stability;
    const overallScore = valuationScore * benchmark.w_val + qualityScore * benchmark.w_qual;

    const completenessKeys = [
        pe,
        ratioPb,
        ratioRoe,
        ratioRoa,
        revCagr3y,
        profitCagr3y,
        ratioDe,
        cfoNp,
        ratioCurrent,
    ];
    const present = completenessKeys.filter((value) => value !== null && Number.isFinite(value)).length;
    let completeness: ValuationDataCompleteness = 'Estimated';
    if (present >= 8 && !hasErrors) {
        completeness = 'Complete';
    } else if (present >= 5) {
        completeness = 'Partial';
    }

    return {
        ticker: stock.ticker.toUpperCase(),
        company_name: stock.company_name,
        sector: stock.industry || UNKNOWN_SECTOR,
        metrics: {
            price_vnd: priceVnd,
            market_cap_bn: marketCapBn,
            avg_vol_20d_m: computeAverageVolume20M(safeBundle.volume_history),
            pe,
            pb: ratioPb,
            ps: ratioPs,
            eps_vnd: epsVnd,
            roe: ratioRoe,
            roa: ratioRoa,
            npm: ratioNpm,
            bvps_vnd: ratioBvps,
            rev_cagr_3y: revCagr3y,
            profit_cagr_3y: profitCagr3y,
            rev_yoy: revYoy,
            profit_yoy: profitYoy,
            de: ratioDe,
            current_ratio: ratioCurrent,
            cfo_np: cfoNp,
            momentum_1m: stock.price_change_1m,
            momentum_6m: stock.price_change_6m ?? null,
            momentum_1y: stock.price_change_1y,
        },
        scores: {
            pe_score: peScore,
            pb_score: pbScore,
            roe_score: roeScore,
            roa_score: roaScore,
            growth_score: growthScore,
            stability_score: stabilityScore,
            valuation_score: valuationScore,
            quality_score: qualityScore,
            overall_score: overallScore,
            rank: 0,
            quadrant: toQuadrant(valuationScore, qualityScore),
            verdict: toVerdict(overallScore),
        },
        flags: {
            used_price_fallback: primaryPrice === null && latestDbClose !== null,
            used_market_cap_fallback: primaryMarketCap === null && fallbackMarketCap !== null,
            used_pe_fallback: primaryPe === null && fallbackPe !== null,
        },
        data_completeness: completeness,
        errors: [...(safeBundle.errors ?? []), ...(hasErrors ? ['partial endpoint failures'] : [])],
    };
};

export const buildValuationScreen = ({
    benchmarkStocks,
    displayStocks,
    bundlesByTicker,
    failedTickers,
}: BuildValuationInput): BuildValuationResult => {
    const displayRows = displayStocks.map((stock) => {
        const ticker = stock.ticker.toUpperCase();
        return buildRow(stock, bundlesByTicker.get(ticker), failedTickers.has(ticker));
    });

    const rankByTicker = getRankMap(displayRows);
    const rows = displayRows.map((row) => ({
        ...row,
        scores: {
            ...row.scores,
            rank: rankByTicker.get(row.ticker) ?? 0,
        },
    }));

    const benchmarkTickerSet = new Set(benchmarkStocks.map((stock) => stock.ticker.toUpperCase()));
    const benchmarkLoaded = Array.from(benchmarkTickerSet).filter((ticker) => bundlesByTicker.has(ticker)).length;

    const stats: ValuationUniverseStats = {
        benchmark_size: benchmarkStocks.length,
        benchmark_loaded: benchmarkLoaded,
        display_size: displayStocks.length,
        display_loaded: displayStocks.filter((stock) => bundlesByTicker.has(stock.ticker.toUpperCase())).length,
        failed_count: failedTickers.size,
        fallback_price_count: rows.filter((row) => row.flags.used_price_fallback).length,
        fallback_market_cap_count: rows.filter((row) => row.flags.used_market_cap_fallback).length,
        fallback_pe_count: rows.filter((row) => row.flags.used_pe_fallback).length,
        partial_count: rows.filter((row) => row.data_completeness === 'Partial').length,
        estimated_count: rows.filter((row) => row.data_completeness === 'Estimated').length,
    };

    return { rows, stats };
};
