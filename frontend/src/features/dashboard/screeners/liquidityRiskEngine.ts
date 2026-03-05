import type { OhlcvDataPoint, Stock } from '../../../api/stockApi';

export interface LiquidityRiskTickerBundle {
    ohlcv: OhlcvDataPoint[];
    errors?: string[];
}

export type LiquidityTier = 'Very High' | 'High' | 'Medium' | 'Low' | 'Very Low' | 'Unknown';
export type RiskTier = 'High' | 'Medium' | 'Low' | 'Unknown';
export type PositionTier = 'T1' | 'T2' | 'T3' | 'T4' | 'T5' | 'N/A';

export interface LiquidityRiskRow {
    ticker: string;
    company_name: string;
    industry: string;
    last_price: number | null;
    beta_1y: number | null;
    vol_1y_pct: number | null;
    max_dd_3y_pct: number | null;
    avg_turnover: number | null;
    atr_pct: number | null;
    risk_score: number | null;
    liquidity_tier: LiquidityTier;
    risk_tier: RiskTier;
    position_tier: PositionTier;
    base_max_pct: number | null;
    adjusted_max_pct: number | null;
    has_errors: boolean;
}

export interface LiquidityRiskUniverseStats {
    benchmark_size: number;
    benchmark_loaded: number;
    display_size: number;
    display_loaded: number;
    failed_count: number;
    score_min: number | null;
    score_max: number | null;
    liquidity_thresholds: {
        p20: number | null;
        p40: number | null;
        p60: number | null;
        p80: number | null;
    };
}

export interface BuildLiquidityRiskInput {
    benchmarkStocks: Stock[];
    displayStocks: Stock[];
    bundlesByTicker: Map<string, LiquidityRiskTickerBundle>;
    failedTickers: Set<string>;
}

export interface BuildLiquidityRiskResult {
    rows: LiquidityRiskRow[];
    stats: LiquidityRiskUniverseStats;
}

interface NormalizedPoint {
    date: string;
    close: number;
    high: number;
    low: number;
    volume: number | null;
}

interface DailyReturnPoint {
    date: string;
    value: number;
}

interface InternalMetrics {
    ticker: string;
    company_name: string;
    industry: string;
    last_price: number | null;
    vol_1y_pct: number | null;
    max_dd_3y_pct: number | null;
    avg_turnover: number | null;
    atr_pct: number | null;
    beta_1y: number | null;
    daily_returns: DailyReturnPoint[];
    has_errors: boolean;
}

interface MinMaxRange {
    min: number;
    max: number;
}

const UNKNOWN_INDUSTRY = 'Unclassified';
const ONE_YEAR_DAYS = 252;
const THREE_YEAR_DAYS = 756;
const ATR_PERIOD = 14;

const clamp = (value: number, min: number, max: number): number => {
    if (value < min) return min;
    if (value > max) return max;
    return value;
};

const roundTo = (value: number, decimals: number): number => {
    const p = 10 ** decimals;
    return Math.round(value * p) / p;
};

const parseNumber = (value: unknown): number | null => {
    if (typeof value === 'number') {
        return Number.isFinite(value) ? value : null;
    }
    if (typeof value === 'string') {
        const parsed = Number(value);
        return Number.isFinite(parsed) ? parsed : null;
    }
    return null;
};

const normalizePriceVnd = (price: number): number => {
    return Math.abs(price) < 1000 ? price * 1000 : price;
};

const normalizeSeries = (points: OhlcvDataPoint[]): NormalizedPoint[] => {
    const deduped = new Map<string, NormalizedPoint>();
    points.forEach((point) => {
        if (!point?.date) return;
        const closeRaw = parseNumber(point.close);
        if (closeRaw === null || closeRaw <= 0) return;
        const close = normalizePriceVnd(closeRaw);
        const highRaw = parseNumber(point.high);
        const lowRaw = parseNumber(point.low);
        const volumeRaw = parseNumber(point.volume);

        const high = normalizePriceVnd(highRaw !== null && highRaw > 0 ? highRaw : closeRaw);
        const low = normalizePriceVnd(lowRaw !== null && lowRaw > 0 ? lowRaw : closeRaw);
        const volume = volumeRaw !== null && volumeRaw >= 0 ? volumeRaw : null;

        deduped.set(point.date, {
            date: point.date,
            close,
            high,
            low,
            volume,
        });
    });
    return Array.from(deduped.values()).sort((a, b) => a.date.localeCompare(b.date));
};

const sampleStdDev = (values: number[]): number | null => {
    if (values.length < 2) return null;
    const mean = values.reduce((acc, value) => acc + value, 0) / values.length;
    const variance = values.reduce((acc, value) => acc + ((value - mean) ** 2), 0) / (values.length - 1);
    if (!Number.isFinite(variance) || variance < 0) return null;
    return Math.sqrt(variance);
};

const mean = (values: number[]): number | null => {
    if (values.length === 0) return null;
    const value = values.reduce((acc, x) => acc + x, 0) / values.length;
    return Number.isFinite(value) ? value : null;
};

const percentile = (values: number[], p: number): number | null => {
    if (values.length === 0) return null;
    const sorted = [...values].sort((a, b) => a - b);
    const pos = clamp(p, 0, 1) * (sorted.length - 1);
    const lower = Math.floor(pos);
    const upper = Math.ceil(pos);
    if (lower === upper) {
        return sorted[lower];
    }
    const weight = pos - lower;
    return sorted[lower] + (sorted[upper] - sorted[lower]) * weight;
};

const computeDailyReturns = (series: NormalizedPoint[]): DailyReturnPoint[] => {
    const returns: DailyReturnPoint[] = [];
    for (let i = 1; i < series.length; i += 1) {
        const prev = series[i - 1].close;
        const curr = series[i].close;
        if (prev <= 0 || curr <= 0) continue;
        const value = (curr / prev) - 1;
        if (!Number.isFinite(value)) continue;
        returns.push({
            date: series[i].date,
            value,
        });
    }
    return returns;
};

const computeVolatility1yPct = (dailyReturns: DailyReturnPoint[]): number | null => {
    const values = dailyReturns.slice(-ONE_YEAR_DAYS).map((item) => item.value);
    const stdev = sampleStdDev(values);
    if (stdev === null) return null;
    return stdev * Math.sqrt(252) * 100;
};

const computeMaxDrawdown3yPct = (series: NormalizedPoint[]): number | null => {
    const closes = series.slice(-THREE_YEAR_DAYS).map((point) => point.close);
    if (closes.length < 2) return null;
    let peak = closes[0];
    let minDrawdown = 0;
    closes.forEach((price) => {
        if (price > peak) {
            peak = price;
            return;
        }
        if (peak <= 0) return;
        const drawdown = (price - peak) / peak;
        if (drawdown < minDrawdown) {
            minDrawdown = drawdown;
        }
    });
    return minDrawdown * 100;
};

const computeAvgTurnover = (series: NormalizedPoint[]): number | null => {
    const turnovers = series
        .slice(-ONE_YEAR_DAYS)
        .map((point) => {
            if (point.volume === null) return null;
            return point.close * point.volume;
        })
        .filter((value): value is number => value !== null && Number.isFinite(value) && value >= 0);
    return mean(turnovers);
};

const computeAtrPct = (series: NormalizedPoint[]): number | null => {
    if (series.length < 2) return null;
    const trueRanges: number[] = [];
    for (let i = 1; i < series.length; i += 1) {
        const curr = series[i];
        const prevClose = series[i - 1].close;
        const tr1 = curr.high - curr.low;
        const tr2 = Math.abs(curr.high - prevClose);
        const tr3 = Math.abs(curr.low - prevClose);
        const tr = Math.max(tr1, tr2, tr3);
        if (!Number.isFinite(tr)) continue;
        trueRanges.push(tr);
    }
    if (trueRanges.length === 0) return null;
    const atrWindow = trueRanges.slice(-ATR_PERIOD);
    const atr = mean(atrWindow);
    const lastClose = series[series.length - 1].close;
    if (atr === null || lastClose <= 0) return null;
    return (atr / lastClose) * 100;
};

const computeBeta1y = (
    dailyReturns: DailyReturnPoint[],
    marketReturnsByDate: Map<string, number>,
): number | null => {
    const pairs = dailyReturns
        .slice(-ONE_YEAR_DAYS)
        .map((item) => {
            const market = marketReturnsByDate.get(item.date);
            if (market === undefined || !Number.isFinite(market)) {
                return null;
            }
            return { stock: item.value, market };
        })
        .filter((pair): pair is { stock: number; market: number } => pair !== null);

    if (pairs.length < 2) return null;
    const stockMean = mean(pairs.map((pair) => pair.stock));
    const marketMean = mean(pairs.map((pair) => pair.market));
    if (stockMean === null || marketMean === null) return null;

    let covariance = 0;
    let marketVariance = 0;
    pairs.forEach((pair) => {
        const marketDelta = pair.market - marketMean;
        covariance += (pair.stock - stockMean) * marketDelta;
        marketVariance += marketDelta * marketDelta;
    });

    covariance /= (pairs.length - 1);
    marketVariance /= (pairs.length - 1);
    if (!Number.isFinite(covariance) || !Number.isFinite(marketVariance) || marketVariance === 0) {
        return null;
    }
    return covariance / marketVariance;
};

const computeMinMax = (values: Array<number | null>): MinMaxRange | null => {
    const valid = values.filter((value): value is number => value !== null && Number.isFinite(value));
    if (valid.length === 0) return null;
    return {
        min: Math.min(...valid),
        max: Math.max(...valid),
    };
};

const normalizeMinMax = (value: number | null, range: MinMaxRange | null): number | null => {
    if (value === null || !range) return null;
    if (range.max === range.min) return 0.5;
    return clamp((value - range.min) / (range.max - range.min), 0, 1);
};

const toLiquidityTier = (
    avgTurnover: number | null,
    thresholds: LiquidityRiskUniverseStats['liquidity_thresholds'],
): LiquidityTier => {
    if (avgTurnover === null) return 'Unknown';
    if (thresholds.p80 !== null && avgTurnover >= thresholds.p80) return 'Very High';
    if (thresholds.p60 !== null && avgTurnover >= thresholds.p60) return 'High';
    if (thresholds.p40 !== null && avgTurnover >= thresholds.p40) return 'Medium';
    if (thresholds.p20 !== null && avgTurnover >= thresholds.p20) return 'Low';
    return 'Very Low';
};

const toRiskTier = (riskScore: number | null): RiskTier => {
    if (riskScore === null) return 'Unknown';
    if (riskScore > 65) return 'High';
    if (riskScore >= 45) return 'Medium';
    return 'Low';
};

const toPositionTier = (riskScore: number | null): PositionTier => {
    if (riskScore === null) return 'N/A';
    if (riskScore < 40) return 'T1';
    if (riskScore < 50) return 'T2';
    if (riskScore < 60) return 'T3';
    if (riskScore <= 70) return 'T4';
    return 'T5';
};

const computeSizing = (
    riskScore: number | null,
    scoreMin: number | null,
    scoreMax: number | null,
    positionTier: PositionTier,
): { base: number | null; adjusted: number | null } => {
    if (riskScore === null || scoreMin === null || scoreMax === null) {
        return { base: null, adjusted: null };
    }
    const norm = scoreMax === scoreMin
        ? 0.5
        : clamp((riskScore - scoreMin) / (scoreMax - scoreMin), 0, 1);
    const base = Math.max(0.5, 10 - (norm * 9));
    const adjusted = (positionTier === 'T4' || positionTier === 'T5') ? base * 0.5 : base;
    return {
        base: roundTo(base, 1),
        adjusted: roundTo(adjusted, 1),
    };
};

const computeInternalMetrics = (
    stock: Stock,
    bundle: LiquidityRiskTickerBundle | undefined,
    hasErrors: boolean,
): InternalMetrics => {
    const series = normalizeSeries(bundle?.ohlcv ?? []);
    const dailyReturns = computeDailyReturns(series);
    const lastPrice = series.length > 0
        ? series[series.length - 1].close
        : (Number.isFinite(stock.price) && stock.price > 0 ? stock.price : null);
    return {
        ticker: stock.ticker.toUpperCase(),
        company_name: stock.company_name || stock.ticker,
        industry: stock.industry || UNKNOWN_INDUSTRY,
        last_price: lastPrice,
        beta_1y: null,
        vol_1y_pct: computeVolatility1yPct(dailyReturns),
        max_dd_3y_pct: computeMaxDrawdown3yPct(series),
        avg_turnover: computeAvgTurnover(series),
        atr_pct: computeAtrPct(series),
        daily_returns: dailyReturns,
        has_errors: hasErrors || ((bundle?.errors?.length ?? 0) > 0),
    };
};

export const buildLiquidityRiskScreen = ({
    benchmarkStocks,
    displayStocks,
    bundlesByTicker,
    failedTickers,
}: BuildLiquidityRiskInput): BuildLiquidityRiskResult => {
    const benchmarkTickerSet = new Set(benchmarkStocks.map((stock) => stock.ticker.toUpperCase()));

    const allStocksByTicker = new Map<string, Stock>([
        ...benchmarkStocks.map((stock) => [stock.ticker.toUpperCase(), stock] as const),
        ...displayStocks.map((stock) => [stock.ticker.toUpperCase(), stock] as const),
    ]);

    const metricsByTicker = new Map<string, InternalMetrics>();
    allStocksByTicker.forEach((stock, ticker) => {
        const bundle = bundlesByTicker.get(ticker);
        metricsByTicker.set(ticker, computeInternalMetrics(stock, bundle, failedTickers.has(ticker)));
    });

    const benchmarkRows = Array.from(benchmarkTickerSet)
        .map((ticker) => metricsByTicker.get(ticker))
        .filter((row): row is InternalMetrics => Boolean(row));

    const marketReturnsAccumulator = new Map<string, number[]>();
    benchmarkRows.forEach((row) => {
        row.daily_returns.forEach((daily) => {
            const arr = marketReturnsAccumulator.get(daily.date) ?? [];
            arr.push(daily.value);
            marketReturnsAccumulator.set(daily.date, arr);
        });
    });

    const marketReturnsByDate = new Map<string, number>();
    marketReturnsAccumulator.forEach((values, date) => {
        const avg = mean(values);
        if (avg !== null) {
            marketReturnsByDate.set(date, avg);
        }
    });

    metricsByTicker.forEach((row) => {
        row.beta_1y = computeBeta1y(row.daily_returns, marketReturnsByDate);
    });

    const volRange = computeMinMax(benchmarkRows.map((row) => row.vol_1y_pct));
    const betaRange = computeMinMax(benchmarkRows.map((row) => row.beta_1y));
    const drawdownAbsRange = computeMinMax(benchmarkRows.map((row) => {
        if (row.max_dd_3y_pct === null) return null;
        return Math.abs(row.max_dd_3y_pct);
    }));
    const turnoverRange = computeMinMax(benchmarkRows.map((row) => row.avg_turnover));

    const liquidityValues = benchmarkRows
        .map((row) => row.avg_turnover)
        .filter((value): value is number => value !== null && Number.isFinite(value));
    const liquidityThresholds = {
        p20: percentile(liquidityValues, 0.2),
        p40: percentile(liquidityValues, 0.4),
        p60: percentile(liquidityValues, 0.6),
        p80: percentile(liquidityValues, 0.8),
    };

    const riskScoreByTicker = new Map<string, number | null>();
    metricsByTicker.forEach((row, ticker) => {
        const volNorm = normalizeMinMax(row.vol_1y_pct, volRange);
        const betaNorm = normalizeMinMax(row.beta_1y, betaRange);
        const drawdownNorm = normalizeMinMax(
            row.max_dd_3y_pct === null ? null : Math.abs(row.max_dd_3y_pct),
            drawdownAbsRange,
        );
        const turnoverNorm = normalizeMinMax(row.avg_turnover, turnoverRange);
        if (volNorm === null || betaNorm === null || drawdownNorm === null || turnoverNorm === null) {
            riskScoreByTicker.set(ticker, null);
            return;
        }
        const score = (
            (volNorm * 0.35)
            + (betaNorm * 0.25)
            + (drawdownNorm * 0.20)
            + ((1 - turnoverNorm) * 0.20)
        ) * 100;
        riskScoreByTicker.set(ticker, roundTo(score, 1));
    });

    const benchmarkScores = Array.from(benchmarkTickerSet)
        .map((ticker) => riskScoreByTicker.get(ticker) ?? null)
        .filter((value): value is number => value !== null && Number.isFinite(value));
    const scoreMin = benchmarkScores.length > 0 ? Math.min(...benchmarkScores) : null;
    const scoreMax = benchmarkScores.length > 0 ? Math.max(...benchmarkScores) : null;

    const rows = displayStocks.map((stock) => {
        const ticker = stock.ticker.toUpperCase();
        const metrics = metricsByTicker.get(ticker) ?? computeInternalMetrics(stock, undefined, failedTickers.has(ticker));
        const riskScore = riskScoreByTicker.get(ticker) ?? null;
        const liquidityTier = toLiquidityTier(metrics.avg_turnover, liquidityThresholds);
        const riskTier = toRiskTier(riskScore);
        const positionTier = toPositionTier(riskScore);
        const sizing = computeSizing(riskScore, scoreMin, scoreMax, positionTier);

        return {
            ticker,
            company_name: metrics.company_name,
            industry: metrics.industry,
            last_price: metrics.last_price,
            beta_1y: metrics.beta_1y,
            vol_1y_pct: metrics.vol_1y_pct !== null ? roundTo(metrics.vol_1y_pct, 1) : null,
            max_dd_3y_pct: metrics.max_dd_3y_pct !== null ? roundTo(metrics.max_dd_3y_pct, 1) : null,
            avg_turnover: metrics.avg_turnover !== null ? roundTo(metrics.avg_turnover, 0) : null,
            atr_pct: metrics.atr_pct !== null ? roundTo(metrics.atr_pct, 1) : null,
            risk_score: riskScore,
            liquidity_tier: liquidityTier,
            risk_tier: riskTier,
            position_tier: positionTier,
            base_max_pct: sizing.base,
            adjusted_max_pct: sizing.adjusted,
            has_errors: metrics.has_errors,
        } satisfies LiquidityRiskRow;
    });

    const benchmarkLoaded = Array.from(benchmarkTickerSet).filter((ticker) => bundlesByTicker.has(ticker)).length;
    const displayLoaded = displayStocks.filter((stock) => bundlesByTicker.has(stock.ticker.toUpperCase())).length;

    return {
        rows,
        stats: {
            benchmark_size: benchmarkStocks.length,
            benchmark_loaded: benchmarkLoaded,
            display_size: displayStocks.length,
            display_loaded: displayLoaded,
            failed_count: failedTickers.size,
            score_min: scoreMin !== null ? roundTo(scoreMin, 1) : null,
            score_max: scoreMax !== null ? roundTo(scoreMax, 1) : null,
            liquidity_thresholds: {
                p20: liquidityThresholds.p20 !== null ? roundTo(liquidityThresholds.p20, 0) : null,
                p40: liquidityThresholds.p40 !== null ? roundTo(liquidityThresholds.p40, 0) : null,
                p60: liquidityThresholds.p60 !== null ? roundTo(liquidityThresholds.p60, 0) : null,
                p80: liquidityThresholds.p80 !== null ? roundTo(liquidityThresholds.p80, 0) : null,
            },
        },
    };
};
