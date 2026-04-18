import type { Stock } from '../../../api/stockApi';

const FINANCIAL_KEYWORDS = [
    'ngan hang',
    'bao hiem',
    'dich vu tai chinh',
    'chung khoan',
    'bank',
    'insurance',
    'financial service',
    'securities',
];

const MANUFACTURING_SECTORS = new Set([
    'tainguyencoban',
    'hoachat',
    'xaydungvatlieu',
    'hangdichvucongnghiep',
    'otolinhkienphutung',
]);

const SOE_NAME_KEYWORDS = ['nha nuoc', 'scic', 'bo ', 'tong cong ty', 'state'];

const UNKNOWN_SECTOR = 'Unclassified';
const MIN_SECTOR_PEERS = 8;

const CORE_METRIC_KEYS = [
    'currentAssets',
    'currentLiabilities',
    'totalAssets',
    'totalLiabilities',
    'equity',
    'retainedEarnings',
    'longTermDebt',
    'revenue',
    'grossProfit',
    'ebit',
    'netIncome',
    'operatingCashFlow',
    'sharesOutstanding',
] as const;

type CoreMetricKey = (typeof CORE_METRIC_KEYS)[number];

type RawRow = Record<string, unknown>;

type ZoneBase = 'Safe' | 'Grey' | 'Distress';
type SolvencyBucket = ZoneBase | 'Unknown';
type QualityBucket = 'Q1' | 'Q2' | 'Q3' | 'Q4';
type HealthRating = 'Excellent' | 'Good' | 'Moderate' | 'Mixed' | 'Concern' | 'Strong' | 'Weak';

export type DataQuality = 'Complete' | 'Partial' | 'Insufficient';

export interface TickerDatasetBundle {
    balance: RawRow[];
    income: RawRow[];
    cashflow: RawRow[];
    ratios: RawRow[];
    price_history: RawRow[];
    overview: RawRow[];
    shareholders: RawRow[];
    errors?: string[];
}

export interface TickerClassification {
    sector: string;
    sector_family_code: string | null;
    sector_family: string | null;
    is_financial: boolean;
    is_bank: boolean;
    is_manufacturing: boolean;
    state_ownership_pct: number;
    is_soe: boolean;
    partial_soe: boolean;
}

export interface TickerFinancialSnapshot {
    ticker: string;
    quarter_cutoff: number | null;
    current_year: number | null;
    prior_year: number | null;
    market_cap_vnd: number | null;
    core_metrics_current: Record<CoreMetricKey, number | null>;
    core_metrics_prior: Record<CoreMetricKey, number | null>;
    current_ratio_current: number | null;
    current_ratio_prior: number | null;
    leverage_ratio_current: number | null;
    leverage_ratio_prior: number | null;
    liabilities_asset_current: number | null;
    liabilities_asset_prior: number | null;
    equity_asset_current: number | null;
    equity_asset_prior: number | null;
    gross_margin_current: number | null;
    gross_margin_prior: number | null;
    asset_turnover_current: number | null;
    asset_turnover_prior: number | null;
    roa_current: number | null;
    roa_prior: number | null;
    roa_delta: number | null;
    accrual_ratio: number | null;
    debt_to_equity: number | null;
    classification: TickerClassification;
    missing_core_metrics: number;
    data_quality: DataQuality;
}

export interface TickerScores {
    z_model: 'Original' | 'EMS' | 'N/A';
    z_score: number | null;
    z_zone_base: ZoneBase | null;
    z_zone_adjusted: string | null;
    vf_score: number;
    vf_sector_pctile: number | null;
    vf_peer_group: 'sector' | 'related' | 'market';
    vf_peer_size: number;
    z_sector_pctile: number | null;
    leverage_flag: 'HIGH vs Sector' | 'Elevated' | 'Normal' | 'Conservative' | 'N/A';
    cutoff_label: string | null;
    cutoff_mismatch_majority: boolean;
    vn_health_rating_base: HealthRating;
    vn_health_rating: HealthRating;
}

export interface TickerScreeningRow {
    ticker: string;
    company_name: string;
    sector: string;
    classification: TickerClassification;
    snapshot: TickerFinancialSnapshot;
    scores: TickerScores;
}

export interface ScreenerUniverseStats {
    benchmark_size: number;
    benchmark_loaded: number;
    benchmark_non_financial_loaded: number;
    failed_count: number;
    majority_cutoff: string | null;
    majority_cutoff_count: number;
    majority_cutoff_coverage_pct: number | null;
    off_majority_cutoff_count: number;
    thresholds: {
        original: { distress: number | null; safe: number | null; sample_count: number };
        ems: { distress: number | null; safe: number | null; sample_count: number };
    };
    benchmark_quality: {
        is_insufficient: boolean;
        thresholds: {
            original_ok: boolean;
            original_n: number;
            ems_ok: boolean;
            ems_n: number;
        };
        fallback_used: boolean;
        fallback_counts: {
            related: number;
            market: number;
        };
        low_peer_sectors: Array<{
            sector: string;
            peer_count: number;
            min_required: number;
            shortfall: number;
        }>;
    };
    insufficient_benchmark: boolean;
}

interface TickerRuntimeMetrics {
    ticker: string;
    sector: string;
    sectorFamilyCode: string | null;
    sectorFamily: string | null;
    isFinancial: boolean;
    isSOE: boolean;
    isPartialSOE: boolean;
    zModel: 'Original' | 'EMS' | 'N/A';
    zScore: number | null;
    vfBaseMetrics: {
        roa: number | null;
        roaDelta: number | null;
        accrual: number | null;
        deltaLeverage: number | null;
        deltaCurrentRatio: number | null;
        deltaGrossMargin: number | null;
        deltaAssetTurnover: number | null;
        noDilution: boolean;
        cfoPositive: boolean;
    };
    debtToEquity: number | null;
}

interface PercentileThreshold {
    distress: number | null;
    safe: number | null;
    sample_count: number;
}

interface VfPeerStats {
    roa: number | null;
    roaDelta: number | null;
    accrual: number | null;
    deltaLeverage: number | null;
    deltaCurrentRatio: number | null;
    deltaGrossMargin: number | null;
    deltaAssetTurnover: number | null;
    debtToEquity: number | null;
}

interface BuildInput {
    benchmarkStocks: Stock[];
    displayStocks: Stock[];
    industryFamiliesByLevel2Name: Map<string, IndustryFamilyMeta>;
    bundlesByTicker: Map<string, TickerDatasetBundle>;
    failedTickers: Set<string>;
}

interface IndustryFamilyMeta {
    family_code: string | null;
    family_name: string | null;
    family_en_name: string | null;
}

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

const normalizeSectorToken = (value: unknown): string => {
    return normalizeText(value).replace(/[^a-z0-9]/g, '');
};

const toRelatedFamilyKey = (
    sectorFamilyCode: string | null,
    sectorFamilyName: string | null,
    isFinancial: boolean,
): string | null => {
    const familyToken = sectorFamilyCode
        ? normalizeSectorToken(sectorFamilyCode)
        : normalizeSectorToken(sectorFamilyName ?? '');
    if (!familyToken) return null;
    return `${isFinancial ? 'financial' : 'non-financial'}:${familyToken}`;
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
        // Mixed separators: assume comma is thousands and dot is decimal.
        normalized = cleaned.replace(/,/g, '');
    } else if (commaCount > 0 && dotCount === 0) {
        // Comma-only format can be either decimal or thousands.
        const lastCommaIndex = cleaned.lastIndexOf(',');
        const fractionLen = lastCommaIndex >= 0 ? cleaned.length - lastCommaIndex - 1 : 0;
        const useThousandsSeparator = commaCount > 1 || fractionLen === 3 || fractionLen === 0;
        normalized = useThousandsSeparator ? cleaned.replace(/,/g, '') : cleaned.replace(',', '.');
    } else if (dotCount > 1 && commaCount === 0) {
        // Dot-only with repeated dots likely means thousands separators.
        normalized = cleaned.replace(/\./g, '');
    }
    const parsed = Number(normalized);
    if (!Number.isFinite(parsed)) {
        return null;
    }
    return negativeWrapped ? -Math.abs(parsed) : parsed;
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

const aggregateRowsToYearCutoff = (
    rows: RawRow[],
    year: number,
    quarterCutoff: number,
): Record<string, number> => {
    const totals: Record<string, number> = {};
    rows.forEach((row) => {
        const rowYear = readYear(row);
        const rowQuarter = readQuarter(row);
        if (rowYear !== year || rowQuarter === null || rowQuarter > quarterCutoff) {
            return;
        }
        Object.entries(row).forEach(([key, raw]) => {
            if (key === 'ticker' || key.startsWith('Meta_')) {
                return;
            }
            const normalizedKey = normalizeMetricKey(key);
            if (!normalizedKey) {
                return;
            }
            const value = parseNumber(raw);
            if (value === null) return;
            totals[normalizedKey] = (totals[normalizedKey] ?? 0) + value;
        });
    });
    return totals;
};

const pickBalanceRow = (
    rows: RawRow[],
    year: number,
    quarterTarget: number,
): Record<string, number> | null => {
    let candidateQuarter = -1;
    let candidate: RawRow | null = null;

    rows.forEach((row) => {
        const rowYear = readYear(row);
        const rowQuarter = readQuarter(row);
        if (rowYear !== year || rowQuarter === null) return;
        if (rowQuarter > quarterTarget) return;
        if (rowQuarter >= candidateQuarter) {
            candidateQuarter = rowQuarter;
            candidate = row;
        }
    });

    if (!candidate) return null;

    const normalized: Record<string, number> = {};
    Object.entries(candidate).forEach(([key, raw]) => {
        if (key === 'ticker' || key.startsWith('Meta_')) return;
        const nk = normalizeMetricKey(key);
        if (!nk) return;
        const value = parseNumber(raw);
        if (value === null) return;
        normalized[nk] = value;
    });
    return normalized;
};

const pickRowsAtOrBeforeQuarter = (
    rows: RawRow[],
    year: number,
    quarterTarget: number,
): RawRow[] => {
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
            const nk = normalizeMetricKey(key);
            if (!nk) return;
            const value = parseNumber(raw);
            if (value === null) return;
            result[nk] = { key, value };
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
                // Prefer closer semantic matches over long accidental substrings.
                let score = alias.length * 1000;
                if (key === alias) {
                    score += 600;
                }
                if (key.startsWith(alias)) {
                    score += 300;
                }
                if (key.endsWith(alias)) {
                    score += 120;
                }
                const extraLength = Math.max(0, key.length - alias.length);
                score -= extraLength;
                if (score > bestScore) {
                    bestScore = score;
                    bestValue = valueObj.value;
                }
            });
        });
    });

    return bestValue;
};

const median = (values: number[]): number | null => {
    if (values.length === 0) return null;
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    if (sorted.length % 2 === 0) {
        return (sorted[mid - 1] + sorted[mid]) / 2;
    }
    return sorted[mid];
};

const quantile = (values: number[], q: number): number | null => {
    if (values.length === 0) return null;
    const sorted = [...values].sort((a, b) => a - b);
    const position = (sorted.length - 1) * q;
    const lower = Math.floor(position);
    const upper = Math.ceil(position);
    if (lower === upper) return sorted[lower];
    const weight = position - lower;
    return sorted[lower] * (1 - weight) + sorted[upper] * weight;
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
    if (Math.abs(value) > 1 && Math.abs(value) <= 100) {
        return value / 100;
    }
    return value;
};

const average = (a: number | null, b: number | null): number | null => {
    if (a === null || b === null) return null;
    return (a + b) / 2;
};

const toCutoffLabel = (year: number | null, quarter: number | null): string | null => {
    if (year === null || quarter === null) return null;
    return `${year}-Q${quarter}`;
};

const cutoffSortRank = (label: string): number => {
    const match = /^(\d{4})-Q([1-4])$/.exec(label);
    if (!match) return -1;
    return Number(match[1]) * 10 + Number(match[2]);
};

const parseIsoDateUtc = (value: unknown): Date | null => {
    if (typeof value !== 'string') return null;
    const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value.trim());
    if (!match) return null;
    const year = Number(match[1]);
    const month = Number(match[2]);
    const day = Number(match[3]);
    if (!Number.isFinite(year) || !Number.isFinite(month) || !Number.isFinite(day)) {
        return null;
    }
    return new Date(Date.UTC(year, month - 1, day));
};

const toVndClosePrice = (price: number): number => {
    // Historical close is stored in 1,000 VND unit in cache for some sources.
    return price >= 1000 ? price : price * 1000;
};

const normalizeMarketCapToVnd = (value: number | null): number | null => {
    if (value === null || !Number.isFinite(value)) {
        return null;
    }
    // Some sources expose market cap in raw VND, others in billion VND.
    // Treat small values as billion-VND and convert to VND.
    return Math.abs(value) < 1_000_000_000 ? value * 1_000_000_000 : value;
};

const normalizeSharesOutstandingToCount = (value: number | null): number | null => {
    if (value === null || !Number.isFinite(value) || value <= 0) {
        return null;
    }
    // Ratios may provide either share count or "million shares".
    return value >= 1_000_000 ? value : value * 1_000_000;
};

const pickCloseAtOrBeforeQuarterEndVnd = (
    priceRows: RawRow[],
    year: number,
    quarter: number,
): number | null => {
    const quarterEndUtc = new Date(Date.UTC(year, quarter * 3, 0));
    let bestTimeAtOrBefore: number | null = null;
    let bestCloseAtOrBefore: number | null = null;
    let bestTimeAny: number | null = null;
    let bestCloseAny: number | null = null;

    priceRows.forEach((row) => {
        const rowDate = parseIsoDateUtc(row.date ?? row.trading_date ?? row.time);
        if (!rowDate) return;
        const close = parseNumber(row.close);
        if (close === null) return;
        const closeVnd = toVndClosePrice(close);
        const rowTime = rowDate.getTime();

        if (bestTimeAny === null || rowTime > bestTimeAny) {
            bestTimeAny = rowTime;
            bestCloseAny = closeVnd;
        }

        if (rowTime <= quarterEndUtc.getTime() && (bestTimeAtOrBefore === null || rowTime > bestTimeAtOrBefore)) {
            bestTimeAtOrBefore = rowTime;
            bestCloseAtOrBefore = closeVnd;
        }
    });

    return bestCloseAtOrBefore ?? bestCloseAny;
};

const toPercent = (value: number | null): number | null => {
    if (value === null) return null;
    if (Math.abs(value) <= 1) {
        return value * 100;
    }
    return value;
};

const buildClassification = (
    stock: Stock,
    overviewRows: RawRow[],
    shareholdersRows: RawRow[],
    industryFamiliesByLevel2Name: Map<string, IndustryFamilyMeta>,
): TickerClassification => {
    const overview = overviewRows[0] || {};
    const level1 = String(overview.icb_name1 ?? '').trim();
    const level2 = String(overview.icb_name2 ?? stock.industry ?? '').trim();
    const level3 = String(overview.icb_name3 ?? '').trim();
    const sector = level2 || level3 || stock.industry || UNKNOWN_SECTOR;
    const level2Family = industryFamiliesByLevel2Name.get(normalizeText(level2));
    const shouldUseOverviewLevel1 = level1 && normalizeText(level1) !== normalizeText(level2);
    const sectorFamily = shouldUseOverviewLevel1
        ? level1
        : (level2Family?.family_name || level1 || null);
    const sectorFamilyCode = level2Family?.family_code || null;

    const sectorText = normalizeText(`${level2} ${level3}`);
    const level3Token = normalizeSectorToken(level3);
    const isFinancial = FINANCIAL_KEYWORDS.some((keyword) => sectorText.includes(keyword));
    const isBank = level3Token.includes('nganhang') || level3Token.includes('bank');
    const isManufacturing = MANUFACTURING_SECTORS.has(normalizeSectorToken(level2));

    const ownership = extractStateOwnershipPercent(shareholdersRows);
    const isSOE = ownership > 30;
    const partialSOE = ownership >= 10 && ownership <= 30;

    return {
        sector: sector || UNKNOWN_SECTOR,
        sector_family_code: sectorFamilyCode,
        sector_family: sectorFamily,
        is_financial: isFinancial,
        is_bank: isBank,
        is_manufacturing: isManufacturing,
        state_ownership_pct: ownership,
        is_soe: isSOE,
        partial_soe: partialSOE,
    };
};

const extractStateOwnershipPercent = (shareholdersRows: RawRow[]): number => {
    let total = 0;
    shareholdersRows.forEach((row) => {
        let shareholderName = '';
        let ownerPercent: number | null = null;
        Object.entries(row).forEach(([key, raw]) => {
            const nk = normalizeMetricKey(key);
            if (!nk) return;
            if (
                nk.includes('shareholder')
                || nk.includes('holdername')
                || nk.includes('name')
                || nk.includes('investor')
            ) {
                if (!shareholderName && typeof raw === 'string') {
                    shareholderName = raw;
                }
            }
            if (nk.includes('percent') || nk.includes('ownership') || nk.includes('shareown')) {
                const parsed = parseNumber(raw);
                if (parsed !== null) {
                    ownerPercent = toPercent(parsed);
                }
            }
        });

        const normalizedName = normalizeText(shareholderName);
        const isStateOwner = SOE_NAME_KEYWORDS.some((keyword) => normalizedName.includes(keyword));
        if (isStateOwner && ownerPercent !== null) {
            total += ownerPercent;
        }
    });

    if (total < 0) return 0;
    if (total > 100) return 100;
    return total;
};

const mapCoreMetrics = (
    balancePoint: Record<string, number> | null,
    incomeAgg: Record<string, number>,
    cashflowAgg: Record<string, number>,
    ratioRowsNormalized: Array<Record<string, { key: string; value: number }>>,
    isFinancial: boolean,
): Record<CoreMetricKey, number | null> => {
    const balanceRows = balancePoint ? [Object.fromEntries(
        Object.entries(balancePoint).map(([key, value]) => [key, { key, value }]),
    )] : [];
    const incomeRows = [Object.fromEntries(
        Object.entries(incomeAgg).map(([key, value]) => [key, { key, value }]),
    )];
    const cashRows = [Object.fromEntries(
        Object.entries(cashflowAgg).map(([key, value]) => [key, { key, value }]),
    )];

    const currentAssets = pickMetric(balanceRows, [
        'currentassetsbnvnd',
        'currentassets',
        'shorttermassets',
        'totalcurrentassets',
        'taisannganhan',
    ], ['currentratio', 'othercurrentassets']);
    const currentLiabilities = pickMetric(balanceRows, [
        'currentliabilitiesbnvnd',
        'currentliabilities',
        'shorttermliabilities',
        'totalcurrentliabilities',
        'noinganhan',
    ]);
    const totalAssets = pickMetric(balanceRows, [
        'totalassetsbnvnd',
        'totalassets',
        'tongtaisan',
    ], ['currentassets', 'longtermassets', 'othercurrentassets', 'othernoncurrentassets', 'shorttermassets']);
    const totalLiabilities = pickMetric(balanceRows, [
        'liabilities',
        'liabilitiesbnvnd',
        'totalliabilities',
        'tongno',
        'nophaitra',
    ], ['currentliabilities', 'shorttermliabilities', 'longtermliabilities', 'otherliabilities', 'derivative']);
    const equity = pickMetric(balanceRows, [
        'ownersequitybnvnd',
        'totalequity',
        'ownersequity',
        'equityattributabletoowners',
        'vonchusohuu',
    ], ['debtequity', 'minortyinterest', 'minorityinterest']);
    const retainedEarnings = pickMetric(balanceRows, [
        'undistributedearningsbnvnd',
        'retainedearnings',
        'retainedprofit',
        'undistributedearnings',
        'undistributedprofit',
        'accumulatedprofits',
        'loinhuanchuaphanphoi',
    ]);
    const longTermDebt = pickMetric(balanceRows, [
        'longtermborrowingsbnvnd',
        'longtermdebt',
        'longtermborrowings',
        'longtermborrowing',
        'debtnoncurrent',
        'vaydaihan',
    ]);
    const revenue = isFinancial
        ? (
            pickMetric(incomeRows, ['totaloperatingincome'], ['yoy'])
            ?? pickMetric(incomeRows, ['totaloperatingrevenue'], ['yoy'])
            ?? pickMetric(incomeRows, ['revenue'], ['yoy'])
        )
        : (
            pickMetric(incomeRows, ['netsales'], ['yoy'])
            ?? pickMetric(incomeRows, ['revenuebnvnd', 'revenue'], ['yoy'])
        );
    const grossProfit = isFinancial
        ? pickMetric(
            incomeRows,
            ['grossprofit', 'laigop', 'grossincome', 'netinterestincome'],
            ['margin', 'yoy'],
        )
        : pickMetric(incomeRows, ['grossprofit', 'laigop', 'grossincome'], ['margin', 'yoy']);
    const operatingProfit = isFinancial
        ? (
            pickMetric(incomeRows, ['netoperatingprofitbeforeallowanceforcreditloss'])
            ?? pickMetric(incomeRows, ['operatingprofitbeforeprovision'])
            ?? pickMetric(incomeRows, ['operatingprofitloss'])
        )
        : pickMetric(incomeRows, ['operatingprofitloss']);
    const interestExpenses = isFinancial
        ? (
            pickMetric(incomeRows, ['interestandsimilarexpenses'])
            ?? pickMetric(incomeRows, ['interestexpenses'])
            ?? 0
        )
        : (pickMetric(incomeRows, ['interestexpenses']) ?? 0);
    const ebitFromOps = operatingProfit !== null ? operatingProfit + Math.abs(interestExpenses) : null;
    const ebitFallback = pickMetric(incomeRows, [
        'profitbeforetax',
        'pretaxprofit',
        'earningsbeforetax',
        'loinhuantruocthue',
    ]);
    const ebit = ebitFromOps ?? ebitFallback;
    const netIncome = pickMetric(incomeRows, [
        'netincome',
        'profitaftertax',
        'loinhuansauthue',
        'earningsaftertax',
        'netprofit',
    ], ['associated', 'minority']);
    const operatingCashFlow = pickMetric(cashRows, [
        'operatingcashflow',
        'cashflowfromoperatingactivities',
        'netcashinflowsoutflowsfromoperatingactivities',
        'netcashflowsfromoperatingactivities',
        'luuchuyentientuhdkinhdoanh',
        'netcashfromoperatingactivities',
        'cfo',
    ], ['increase', 'decrease']);
    const sharesFromRatios = pickMetric(ratioRowsNormalized, [
        'chitieudinhgiaoutstandingsharemilshares',
        'outstandingsharemilshares',
        'sharesoutstanding',
        'outstandingshare',
        'outstandingshares',
        'issue_share',
        'issueshare',
        'shareissued',
        'sharesissued',
        'totalshares',
        'numberofshares',
        'listedshares',
        'listingshares',
        'listingvolume',
        'ordinaryshares',
    ]);
    const sharesFromBalance = pickMetric(balanceRows, [
        'sharesoutstanding',
        'outstandingshares',
        'issueshare',
        'shareissued',
        'sharesissued',
        'totalshares',
        'numberofshares',
        'listedshares',
        'listingshares',
        'listingvolume',
        'cophieuluuhanh',
        'soluongcophieuluuhanh',
    ]);
    const sharesOutstanding = normalizeSharesOutstandingToCount(sharesFromRatios ?? sharesFromBalance ?? null);

    return {
        currentAssets,
        currentLiabilities,
        totalAssets,
        totalLiabilities,
        equity,
        retainedEarnings,
        longTermDebt,
        revenue,
        grossProfit,
        ebit,
        netIncome,
        operatingCashFlow,
        sharesOutstanding,
    };
};

const computeSnapshot = (
    ticker: string,
    stock: Stock,
    bundle: TickerDatasetBundle,
    industryFamiliesByLevel2Name: Map<string, IndustryFamilyMeta>,
): TickerFinancialSnapshot | null => {
    const latest = findLatestPeriod(bundle.balance);
    if (!latest) {
        return null;
    }
    const currentYear = latest.year;
    const quarterCutoff = latest.quarter;
    const priorYear = currentYear - 1;

    const currentBalance = pickBalanceRow(bundle.balance, currentYear, quarterCutoff);
    const priorBalance = pickBalanceRow(bundle.balance, priorYear, quarterCutoff);
    const currentIncome = aggregateRowsToYearCutoff(bundle.income, currentYear, quarterCutoff);
    const priorIncome = aggregateRowsToYearCutoff(bundle.income, priorYear, quarterCutoff);
    const currentCashflow = aggregateRowsToYearCutoff(bundle.cashflow, currentYear, quarterCutoff);
    const priorCashflow = aggregateRowsToYearCutoff(bundle.cashflow, priorYear, quarterCutoff);

    const classification = buildClassification(stock, bundle.overview, bundle.shareholders, industryFamiliesByLevel2Name);
    const currentRatioRows = normalizeDataRows(pickRowsAtOrBeforeQuarter(bundle.ratios, currentYear, quarterCutoff));
    const priorRatioRows = normalizeDataRows(pickRowsAtOrBeforeQuarter(bundle.ratios, priorYear, quarterCutoff));

    const coreCurrent = mapCoreMetrics(currentBalance, currentIncome, currentCashflow, currentRatioRows, classification.is_financial);
    const corePrior = mapCoreMetrics(priorBalance, priorIncome, priorCashflow, priorRatioRows, classification.is_financial);
    const marketCapFromRatioVnd = normalizeMarketCapToVnd(
        pickMetric(currentRatioRows, ['chitieudinhgiamarketcapitalbnvnd', 'marketcapitalbnvnd', 'marketcap']),
    );
    const closeAtCutoffVnd = pickCloseAtOrBeforeQuarterEndVnd(bundle.price_history, currentYear, quarterCutoff);
    const marketCapFromHistoryVnd = closeAtCutoffVnd !== null
        && coreCurrent.sharesOutstanding !== null
        && coreCurrent.sharesOutstanding > 0
        ? closeAtCutoffVnd * coreCurrent.sharesOutstanding
        : null;

    const avgAssetsCurrent = average(coreCurrent.totalAssets, corePrior.totalAssets);
    const prePriorBalance = pickBalanceRow(bundle.balance, priorYear - 1, quarterCutoff);
    const prePriorAssets = prePriorBalance
        ? pickMetric(
            [Object.fromEntries(Object.entries(prePriorBalance).map(([key, value]) => [key, { key, value }]))],
            ['totalassets', 'tongtaisan'],
            ['currentassets'],
        )
        : null;
    const avgAssetsPriorForRoa = prePriorAssets !== null
        ? average(corePrior.totalAssets, prePriorAssets)
        : corePrior.totalAssets;
    const avgAssetsPrior = average(corePrior.totalAssets, prePriorAssets);

    const roaCurrent = safeDiv(coreCurrent.netIncome, avgAssetsCurrent);
    const roaPrior = safeDiv(corePrior.netIncome, avgAssetsPriorForRoa);
    const roaDelta = roaCurrent !== null && roaPrior !== null ? roaCurrent - roaPrior : null;
    const accrualRatio = safeDiv(
        coreCurrent.operatingCashFlow !== null && coreCurrent.netIncome !== null
            ? coreCurrent.operatingCashFlow - coreCurrent.netIncome
            : null,
        avgAssetsCurrent,
    );

    const currentRatioCurrent = safeDiv(coreCurrent.currentAssets, coreCurrent.currentLiabilities);
    const currentRatioPrior = safeDiv(corePrior.currentAssets, corePrior.currentLiabilities);

    const leverageCurrent = safeDiv(coreCurrent.longTermDebt, coreCurrent.totalAssets);
    const leveragePrior = safeDiv(corePrior.longTermDebt, corePrior.totalAssets);
    const liabilitiesAssetCurrent = safeDiv(coreCurrent.totalLiabilities, coreCurrent.totalAssets);
    const liabilitiesAssetPrior = safeDiv(corePrior.totalLiabilities, corePrior.totalAssets);
    const equityAssetCurrent = safeDiv(coreCurrent.equity, coreCurrent.totalAssets);
    const equityAssetPrior = safeDiv(corePrior.equity, corePrior.totalAssets);

    const grossMarginFromRatioCurrent = asRate(pickMetric(currentRatioRows, ['grossmargin'], ['yoy']));
    const grossMarginFromRatioPrior = asRate(pickMetric(priorRatioRows, ['grossmargin'], ['yoy']));
    const grossMarginCurrent = safeDiv(coreCurrent.grossProfit, coreCurrent.revenue) ?? grossMarginFromRatioCurrent;
    const grossMarginPrior = safeDiv(corePrior.grossProfit, corePrior.revenue) ?? grossMarginFromRatioPrior;
    const assetTurnoverCurrent = safeDiv(coreCurrent.revenue, avgAssetsCurrent);
    const assetTurnoverPrior = safeDiv(corePrior.revenue, avgAssetsPrior);

    const latestRatioRows = currentRatioRows;
    const debtToEquityFromRatio = pickMetric(latestRatioRows, ['debtequity', 'detoequity', 'debttoequity']);
    const debtToEquityDerived = safeDiv(coreCurrent.totalLiabilities, coreCurrent.equity);
    const debtToEquity = debtToEquityFromRatio ?? debtToEquityDerived;

    const essentialCurrent: CoreMetricKey[] = [
        'totalAssets',
        'totalLiabilities',
        'equity',
        'revenue',
        'grossProfit',
        'netIncome',
        'operatingCashFlow',
    ];
    const essentialPrior: CoreMetricKey[] = [
        'totalAssets',
        'equity',
        'revenue',
        'grossProfit',
        'netIncome',
    ];
    const currentMissing = essentialCurrent.filter((key) => coreCurrent[key] === null).length;
    const priorMissing = essentialPrior.filter((key) => corePrior[key] === null).length;
    const hardInsufficient = currentMissing >= 5;
    const hasStrongCoverage = currentMissing <= 1 && priorMissing <= 2;
    const hasModerateCoverage = currentMissing <= 3 && priorMissing <= 4;
    let dataQuality: DataQuality = 'Insufficient';
    if (!hardInsufficient && hasStrongCoverage) {
        dataQuality = 'Complete';
    } else if (!hardInsufficient && hasModerateCoverage) {
        dataQuality = 'Partial';
    }

    const missingCoreMetrics = CORE_METRIC_KEYS.reduce((acc, key) => {
        const cur = coreCurrent[key];
        const prev = corePrior[key];
        return acc + (cur === null ? 1 : 0) + (prev === null ? 1 : 0);
    }, 0);

    return {
        ticker,
        quarter_cutoff: quarterCutoff,
        current_year: currentYear,
        prior_year: priorYear,
        market_cap_vnd: marketCapFromRatioVnd
            ?? marketCapFromHistoryVnd
            ?? normalizeMarketCapToVnd(Number.isFinite(stock.market_cap) ? stock.market_cap : null),
        core_metrics_current: coreCurrent,
        core_metrics_prior: corePrior,
        current_ratio_current: currentRatioCurrent,
        current_ratio_prior: currentRatioPrior,
        leverage_ratio_current: leverageCurrent,
        leverage_ratio_prior: leveragePrior,
        liabilities_asset_current: liabilitiesAssetCurrent,
        liabilities_asset_prior: liabilitiesAssetPrior,
        equity_asset_current: equityAssetCurrent,
        equity_asset_prior: equityAssetPrior,
        gross_margin_current: grossMarginCurrent,
        gross_margin_prior: grossMarginPrior,
        asset_turnover_current: assetTurnoverCurrent,
        asset_turnover_prior: assetTurnoverPrior,
        roa_current: roaCurrent,
        roa_prior: roaPrior,
        roa_delta: roaDelta,
        accrual_ratio: accrualRatio,
        debt_to_equity: debtToEquity,
        classification,
        missing_core_metrics: missingCoreMetrics,
        data_quality: dataQuality,
    };
};

const computeZScore = (snapshot: TickerFinancialSnapshot): { model: 'Original' | 'EMS' | 'N/A'; score: number | null } => {
    const classification = snapshot.classification;
    if (classification.is_financial) {
        return { model: 'N/A', score: null };
    }

    const current = snapshot.core_metrics_current;
    const wc = current.currentAssets !== null && current.currentLiabilities !== null
        ? current.currentAssets - current.currentLiabilities
        : null;
    const x1 = safeDiv(wc, current.totalAssets);
    const x2 = safeDiv(current.retainedEarnings, current.totalAssets);
    const x3 = safeDiv(current.ebit, current.totalAssets);
    const x4 = safeDiv(snapshot.market_cap_vnd, current.totalLiabilities);
    const x5 = safeDiv(current.revenue, current.totalAssets);

    if (classification.is_manufacturing) {
        if ([x1, x2, x3, x4, x5].some((x) => x === null)) {
            return { model: 'Original', score: null };
        }
        const score = 1.2 * (x1 as number)
            + 1.4 * (x2 as number)
            + 3.3 * (x3 as number)
            + 0.6 * (x4 as number)
            + 1.0 * (x5 as number);
        return { model: 'Original', score };
    }

    if ([x1, x2, x3, x4].some((x) => x === null)) {
        return { model: 'EMS', score: null };
    }
    const score = 6.56 * (x1 as number)
        + 3.26 * (x2 as number)
        + 6.72 * (x3 as number)
        + 1.05 * (x4 as number)
        + 3.25;
    return { model: 'EMS', score };
};

const toThreshold = (values: number[]): PercentileThreshold => {
    if (values.length < 5) {
        return { distress: null, safe: null, sample_count: values.length };
    }
    return {
        distress: quantile(values, 0.15),
        safe: quantile(values, 0.5),
        sample_count: values.length,
    };
};

const signalGtMedian = (value: number | null, medianValue: number | null): boolean => {
    if (value === null) return false;
    if (medianValue === null) return value > 0;
    return value > medianValue;
};

const signalLtMedian = (value: number | null, medianValue: number | null): boolean => {
    if (value === null) return false;
    if (medianValue === null) return value < 0;
    return value < medianValue;
};


const computePercentile = (value: number | null, values: number[]): number | null => {
    if (value === null || values.length === 0) {
        return null;
    }
    const greaterCount = values.filter((v) => v > value).length;
    const rank = greaterCount + 1;
    return 1 - (rank - 1) / values.length;
};

const zoneFromScore = (score: number, threshold: PercentileThreshold): ZoneBase | null => {
    if (threshold.distress === null || threshold.safe === null) {
        return null;
    }
    if (score < threshold.distress) return 'Distress';
    if (score > threshold.safe) return 'Safe';
    return 'Grey';
};

const applySOEAdjustment = (
    zone: ZoneBase | null,
    classification: TickerClassification,
): string | null => {
    if (!zone) return null;
    if (classification.is_soe) {
        if (zone === 'Grey') return 'Safe (SOE↑)';
        if (zone === 'Distress') return 'Grey (SOE↑)';
        return 'Safe';
    }
    if (classification.partial_soe && zone === 'Distress') {
        return 'Grey (SOE↑)';
    }
    return zone;
};

const NON_FINANCIAL_RATING_ORDER: Array<'Concern' | 'Mixed' | 'Moderate' | 'Good' | 'Excellent'> = [
    'Concern',
    'Mixed',
    'Moderate',
    'Good',
    'Excellent',
];

const NON_FINANCIAL_MATRIX: Record<QualityBucket, Record<SolvencyBucket, HealthRating>> = {
    Q4: { Safe: 'Excellent', Grey: 'Good', Distress: 'Mixed', Unknown: 'Good' },
    Q3: { Safe: 'Good', Grey: 'Moderate', Distress: 'Concern', Unknown: 'Moderate' },
    Q2: { Safe: 'Moderate', Grey: 'Mixed', Distress: 'Concern', Unknown: 'Mixed' },
    Q1: { Safe: 'Mixed', Grey: 'Concern', Distress: 'Concern', Unknown: 'Concern' },
};

const toSolvencyBucket = (zone: ZoneBase | null): SolvencyBucket => {
    return zone ?? 'Unknown';
};

const toQualityBucket = (
    vfPercentile: number | null,
    vfScore: number,
): QualityBucket => {
    if (vfPercentile !== null) {
        if (vfPercentile >= 0.75) return 'Q4';
        if (vfPercentile >= 0.5) return 'Q3';
        if (vfPercentile >= 0.25) return 'Q2';
        return 'Q1';
    }
    if (vfScore >= 7) return 'Q4';
    if (vfScore >= 5) return 'Q3';
    if (vfScore >= 3) return 'Q2';
    return 'Q1';
};

const cappedDistressRating = (rating: HealthRating): HealthRating => {
    const mixedRank = NON_FINANCIAL_RATING_ORDER.indexOf('Mixed');
    const ratingRank = NON_FINANCIAL_RATING_ORDER.indexOf(rating as typeof NON_FINANCIAL_RATING_ORDER[number]);
    if (ratingRank > mixedRank) {
        return 'Mixed';
    }
    return rating;
};

const matrixHealthRating = (
    snapshot: TickerFinancialSnapshot,
    zZoneBase: ZoneBase | null,
    vfScore: number,
    vfPercentile: number | null,
): HealthRating => {
    const qualityBucket = toQualityBucket(vfPercentile, vfScore);
    if (snapshot.classification.is_financial) {
        if (qualityBucket === 'Q4' || qualityBucket === 'Q3') {
            return 'Strong';
        }
        if (qualityBucket === 'Q2') {
            return 'Moderate';
        }
        return 'Weak';
    }

    const solvencyBucket = toSolvencyBucket(zZoneBase);
    const baseRating = NON_FINANCIAL_MATRIX[qualityBucket][solvencyBucket];
    if (zZoneBase === 'Distress') {
        return cappedDistressRating(baseRating);
    }
    return baseRating;
};

const upgradeNonFinancialRatingOneNotch = (rating: HealthRating): HealthRating => {
    const rank = NON_FINANCIAL_RATING_ORDER.indexOf(rating as typeof NON_FINANCIAL_RATING_ORDER[number]);
    if (rank < 0 || rank === NON_FINANCIAL_RATING_ORDER.length - 1) {
        return rating;
    }
    return NON_FINANCIAL_RATING_ORDER[rank + 1];
};

const applySoeLabelModifier = (
    baseRating: HealthRating,
    snapshot: TickerFinancialSnapshot,
    zZoneBase: ZoneBase | null,
): HealthRating => {
    if (snapshot.classification.is_financial || zZoneBase === null) {
        return baseRating;
    }

    const shouldUpgrade = snapshot.classification.is_soe
        ? (zZoneBase === 'Grey' || zZoneBase === 'Distress')
        : (snapshot.classification.partial_soe && zZoneBase === 'Grey');

    if (!shouldUpgrade) {
        return baseRating;
    }

    const upgraded = upgradeNonFinancialRatingOneNotch(baseRating);
    if (zZoneBase === 'Distress') {
        return cappedDistressRating(upgraded);
    }
    return upgraded;
};

const toLeverageFlag = (de: number | null, sectorMedian: number | null): TickerScores['leverage_flag'] => {
    if (de === null || sectorMedian === null || sectorMedian <= 0) {
        return 'N/A';
    }
    if (de > sectorMedian * 2) return 'HIGH vs Sector';
    if (de > sectorMedian * 1.5) return 'Elevated';
    if (de < sectorMedian * 0.5) return 'Conservative';
    return 'Normal';
};

const buildVfDerivedMetrics = (snapshot: TickerFinancialSnapshot) => {
    const isFinancial = snapshot.classification.is_financial;
    const isBank = snapshot.classification.is_bank;
    const leverageCurrent = isFinancial
        ? snapshot.liabilities_asset_current
        : (snapshot.leverage_ratio_current ?? snapshot.liabilities_asset_current);
    const leveragePrior = isFinancial
        ? snapshot.liabilities_asset_prior
        : (snapshot.leverage_ratio_prior ?? snapshot.liabilities_asset_prior);
    const deltaLeverage = isFinancial
        ? null
        : (leverageCurrent !== null && leveragePrior !== null
            ? leverageCurrent - leveragePrior
            : null);

    const wcRatioCurrent = snapshot.core_metrics_current.currentAssets !== null
        && snapshot.core_metrics_current.currentLiabilities !== null
        && snapshot.core_metrics_current.totalAssets !== null
        ? safeDiv(
            snapshot.core_metrics_current.currentAssets - snapshot.core_metrics_current.currentLiabilities,
            snapshot.core_metrics_current.totalAssets,
        )
        : null;
    const wcRatioPrior = snapshot.core_metrics_prior.currentAssets !== null
        && snapshot.core_metrics_prior.currentLiabilities !== null
        && snapshot.core_metrics_prior.totalAssets !== null
        ? safeDiv(
            snapshot.core_metrics_prior.currentAssets - snapshot.core_metrics_prior.currentLiabilities,
            snapshot.core_metrics_prior.totalAssets,
        )
        : null;
    const liquidityCurrent = isFinancial
        ? (isBank ? null : (snapshot.current_ratio_current ?? wcRatioCurrent))
        : (snapshot.current_ratio_current ?? wcRatioCurrent);
    const liquidityPrior = isFinancial
        ? (isBank ? null : (snapshot.current_ratio_prior ?? wcRatioPrior))
        : (snapshot.current_ratio_prior ?? wcRatioPrior);
    const deltaCurrentRatio = liquidityCurrent !== null && liquidityPrior !== null
        ? liquidityCurrent - liquidityPrior
        : null;

    const noDilution = snapshot.core_metrics_current.sharesOutstanding !== null
        && snapshot.core_metrics_prior.sharesOutstanding !== null
        ? snapshot.core_metrics_current.sharesOutstanding <= snapshot.core_metrics_prior.sharesOutstanding
        : true;

    return {
        deltaLeverage,
        deltaCurrentRatio,
        noDilution,
    };
};

const buildRuntimeMetric = (
    ticker: string,
    snapshot: TickerFinancialSnapshot,
): TickerRuntimeMetrics => {
    const z = computeZScore(snapshot);
    const derived = buildVfDerivedMetrics(snapshot);
    return {
        ticker,
        sector: snapshot.classification.sector || UNKNOWN_SECTOR,
        sectorFamilyCode: snapshot.classification.sector_family_code,
        sectorFamily: snapshot.classification.sector_family,
        isFinancial: snapshot.classification.is_financial,
        isSOE: snapshot.classification.is_soe,
        isPartialSOE: snapshot.classification.partial_soe,
        zModel: z.model,
        zScore: z.score,
        vfBaseMetrics: {
            roa: snapshot.roa_current,
            roaDelta: snapshot.roa_delta,
            accrual: snapshot.accrual_ratio,
            deltaLeverage: derived.deltaLeverage,
            deltaCurrentRatio: derived.deltaCurrentRatio,
            deltaGrossMargin: snapshot.gross_margin_current !== null && snapshot.gross_margin_prior !== null
                ? snapshot.gross_margin_current - snapshot.gross_margin_prior
                : null,
            deltaAssetTurnover: snapshot.asset_turnover_current !== null && snapshot.asset_turnover_prior !== null
                ? snapshot.asset_turnover_current - snapshot.asset_turnover_prior
                : null,
            noDilution: derived.noDilution,
            cfoPositive: snapshot.core_metrics_current.operatingCashFlow !== null
                && snapshot.core_metrics_current.operatingCashFlow > 0,
        },
        debtToEquity: snapshot.debt_to_equity,
    };
};

const buildVfPeerStats = (peers: TickerRuntimeMetrics[]): VfPeerStats => {
    return {
        roa: median(peers.map((p) => p.vfBaseMetrics.roa).filter((v): v is number => v !== null)),
        roaDelta: median(peers.map((p) => p.vfBaseMetrics.roaDelta).filter((v): v is number => v !== null)),
        accrual: median(peers.map((p) => p.vfBaseMetrics.accrual).filter((v): v is number => v !== null)),
        deltaLeverage: median(peers.map((p) => p.vfBaseMetrics.deltaLeverage).filter((v): v is number => v !== null)),
        deltaCurrentRatio: median(peers.map((p) => p.vfBaseMetrics.deltaCurrentRatio).filter((v): v is number => v !== null)),
        deltaGrossMargin: median(peers.map((p) => p.vfBaseMetrics.deltaGrossMargin).filter((v): v is number => v !== null)),
        deltaAssetTurnover: median(peers.map((p) => p.vfBaseMetrics.deltaAssetTurnover).filter((v): v is number => v !== null)),
        debtToEquity: median(peers.map((p) => p.debtToEquity).filter((v): v is number => v !== null)),
    };
};

const computeVfScore = (metric: TickerRuntimeMetrics, peerStats: VfPeerStats): number => {
    const vf1 = signalGtMedian(metric.vfBaseMetrics.roa, peerStats.roa);
    const vf2 = metric.vfBaseMetrics.cfoPositive;
    const vf3 = signalGtMedian(metric.vfBaseMetrics.roaDelta, peerStats.roaDelta);
    const vf4 = signalGtMedian(metric.vfBaseMetrics.accrual, peerStats.accrual);
    const vf5 = signalLtMedian(metric.vfBaseMetrics.deltaLeverage, peerStats.deltaLeverage);
    const vf6 = signalGtMedian(metric.vfBaseMetrics.deltaCurrentRatio, peerStats.deltaCurrentRatio);
    const vf7 = metric.vfBaseMetrics.noDilution;
    const vf8 = signalGtMedian(metric.vfBaseMetrics.deltaGrossMargin, peerStats.deltaGrossMargin);
    const vf9 = signalGtMedian(metric.vfBaseMetrics.deltaAssetTurnover, peerStats.deltaAssetTurnover);
    return [vf1, vf2, vf3, vf4, vf5, vf6, vf7, vf8, vf9].reduce((sum, signal) => sum + (signal ? 1 : 0), 0);
};

const resolvePeerContext = (
    metric: TickerRuntimeMetrics,
    sectorPeerMap: Map<string, TickerRuntimeMetrics[]>,
    relatedFamilyPeerMap: Map<string, TickerRuntimeMetrics[]>,
    marketNonFinancialPeers: TickerRuntimeMetrics[],
    marketFinancialPeers: TickerRuntimeMetrics[],
): { peerGroup: 'sector' | 'related' | 'market'; peers: TickerRuntimeMetrics[] } => {
    const sectorPeers = sectorPeerMap.get(metric.sector) ?? [];
    if (sectorPeers.length >= MIN_SECTOR_PEERS) {
        return { peerGroup: 'sector', peers: sectorPeers };
    }

    const relatedFamilyKey = toRelatedFamilyKey(metric.sectorFamilyCode, metric.sectorFamily, metric.isFinancial);
    if (relatedFamilyKey) {
        const relatedPeers = relatedFamilyPeerMap.get(relatedFamilyKey) ?? [];
        if (relatedPeers.length >= MIN_SECTOR_PEERS) {
            return { peerGroup: 'related', peers: relatedPeers };
        }
    }

    const marketPeers = metric.isFinancial ? marketFinancialPeers : marketNonFinancialPeers;
    if (marketPeers.length > 0) {
        return { peerGroup: 'market', peers: marketPeers };
    }
    return { peerGroup: 'sector', peers: sectorPeers };
};

export const buildFinancialHealthScreen = (input: BuildInput): {
    rows: TickerScreeningRow[];
    stats: ScreenerUniverseStats;
} => {
    const benchmarkStockMap = new Map<string, Stock>(
        input.benchmarkStocks.map((stock) => [stock.ticker.toUpperCase(), stock]),
    );
    const displayStockMap = new Map<string, Stock>(
        input.displayStocks.map((stock) => [stock.ticker.toUpperCase(), stock]),
    );

    const allStockMap = new Map<string, Stock>([...benchmarkStockMap.entries(), ...displayStockMap.entries()]);

    const snapshotMap = new Map<string, TickerFinancialSnapshot>();
    allStockMap.forEach((stock, ticker) => {
        const bundle = input.bundlesByTicker.get(ticker);
        if (!bundle) return;
        const snapshot = computeSnapshot(ticker, stock, bundle, input.industryFamiliesByLevel2Name);
        if (snapshot) {
            snapshotMap.set(ticker, snapshot);
        }
    });

    const cutoffCounts = new Map<string, number>();
    benchmarkStockMap.forEach((_stock, ticker) => {
        const snapshot = snapshotMap.get(ticker);
        if (!snapshot) return;
        const cutoffLabel = toCutoffLabel(snapshot.current_year, snapshot.quarter_cutoff);
        if (!cutoffLabel) return;
        cutoffCounts.set(cutoffLabel, (cutoffCounts.get(cutoffLabel) ?? 0) + 1);
    });

    let majorityCutoff: string | null = null;
    let majorityCutoffCount = 0;
    cutoffCounts.forEach((count, label) => {
        if (count > majorityCutoffCount) {
            majorityCutoff = label;
            majorityCutoffCount = count;
            return;
        }
        if (count === majorityCutoffCount && majorityCutoff !== null) {
            if (cutoffSortRank(label) > cutoffSortRank(majorityCutoff)) {
                majorityCutoff = label;
            }
        }
    });

    const benchmarkMetrics: TickerRuntimeMetrics[] = [];
    const benchmarkMetricMap = new Map<string, TickerRuntimeMetrics>();
    benchmarkStockMap.forEach((_stock, ticker) => {
        const snapshot = snapshotMap.get(ticker);
        if (!snapshot) return;
        const metric = buildRuntimeMetric(ticker, snapshot);
        benchmarkMetrics.push(metric);
        benchmarkMetricMap.set(ticker, metric);
    });

    const originalZ = benchmarkMetrics
        .filter((m) => !m.isFinancial && m.zModel === 'Original' && m.zScore !== null)
        .map((m) => m.zScore as number);
    const emsZ = benchmarkMetrics
        .filter((m) => !m.isFinancial && m.zModel === 'EMS' && m.zScore !== null)
        .map((m) => m.zScore as number);
    const originalThreshold = toThreshold(originalZ);
    const emsThreshold = toThreshold(emsZ);

    const sectorPeerMap = new Map<string, TickerRuntimeMetrics[]>();
    benchmarkMetrics.forEach((metric) => {
        const arr = sectorPeerMap.get(metric.sector) ?? [];
        arr.push(metric);
        sectorPeerMap.set(metric.sector, arr);
    });

    const relatedFamilyPeerMap = new Map<string, TickerRuntimeMetrics[]>();
    benchmarkMetrics.forEach((metric) => {
        const relatedFamilyKey = toRelatedFamilyKey(metric.sectorFamilyCode, metric.sectorFamily, metric.isFinancial);
        if (!relatedFamilyKey) return;
        const arr = relatedFamilyPeerMap.get(relatedFamilyKey) ?? [];
        arr.push(metric);
        relatedFamilyPeerMap.set(relatedFamilyKey, arr);
    });

    const marketNonFinancialPeers = benchmarkMetrics.filter((metric) => !metric.isFinancial);
    const marketFinancialPeers = benchmarkMetrics.filter((metric) => metric.isFinancial);

    const sectorPeerStatsMap = new Map<string, VfPeerStats>();
    const sectorZValuesMap = new Map<string, number[]>();
    sectorPeerMap.forEach((peers, sector) => {
        sectorPeerStatsMap.set(sector, buildVfPeerStats(peers));
        sectorZValuesMap.set(sector, peers.map((p) => p.zScore).filter((v): v is number => v !== null));
    });

    const relatedPeerStatsMap = new Map<string, VfPeerStats>();
    relatedFamilyPeerMap.forEach((peers, relatedFamilyKey) => {
        relatedPeerStatsMap.set(relatedFamilyKey, buildVfPeerStats(peers));
    });

    const marketPeerStatsByClass = {
        nonFinancial: buildVfPeerStats(marketNonFinancialPeers),
        financial: buildVfPeerStats(marketFinancialPeers),
    };

    const lowPeerSectors = Array.from(sectorPeerMap.entries())
        .map(([sector, peers]) => {
            const resolvedPeerCount = peers.reduce((minCount, peerMetric) => {
                const peerContext = resolvePeerContext(
                    peerMetric,
                    sectorPeerMap,
                    relatedFamilyPeerMap,
                    marketNonFinancialPeers,
                    marketFinancialPeers,
                );
                return Math.min(minCount, peerContext.peers.length);
            }, Number.POSITIVE_INFINITY);
            const peerCount = Number.isFinite(resolvedPeerCount) ? resolvedPeerCount : peers.length;
            return {
                sector,
                peer_count: peerCount,
                min_required: MIN_SECTOR_PEERS,
                shortfall: Math.max(0, MIN_SECTOR_PEERS - peerCount),
            };
        })
        .filter((sectorStats) => sectorStats.peer_count < MIN_SECTOR_PEERS)
        .sort((a, b) => {
            if (b.shortfall !== a.shortfall) {
                return b.shortfall - a.shortfall;
            }
            if (a.peer_count !== b.peer_count) {
                return a.peer_count - b.peer_count;
            }
            return a.sector.localeCompare(b.sector);
        });

    const fallbackCounts = benchmarkMetrics.reduce((acc, metric) => {
        const peerContext = resolvePeerContext(
            metric,
            sectorPeerMap,
            relatedFamilyPeerMap,
            marketNonFinancialPeers,
            marketFinancialPeers,
        );
        if (peerContext.peerGroup === 'related') {
            acc.related += 1;
        } else if (peerContext.peerGroup === 'market') {
            acc.market += 1;
        }
        return acc;
    }, { related: 0, market: 0 });
    const fallbackUsed = (fallbackCounts.related + fallbackCounts.market) > 0;

    const originalThresholdOk = originalThreshold.distress !== null && originalThreshold.safe !== null;
    const emsThresholdOk = emsThreshold.distress !== null && emsThreshold.safe !== null;
    const benchmarkQualityInsufficient = !originalThresholdOk || !emsThresholdOk || lowPeerSectors.length > 0;

    const rows: TickerScreeningRow[] = [];
    displayStockMap.forEach((stock, ticker) => {
        const snapshot = snapshotMap.get(ticker);
        if (!snapshot) {
            return;
        }

        const metric = benchmarkMetricMap.get(ticker) ?? buildRuntimeMetric(ticker, snapshot);
        const peerContext = resolvePeerContext(
            metric,
            sectorPeerMap,
            relatedFamilyPeerMap,
            marketNonFinancialPeers,
            marketFinancialPeers,
        );
        let peerStats: VfPeerStats;
        if (peerContext.peerGroup === 'market') {
            peerStats = metric.isFinancial ? marketPeerStatsByClass.financial : marketPeerStatsByClass.nonFinancial;
        } else if (peerContext.peerGroup === 'related') {
            const relatedFamilyKey = toRelatedFamilyKey(metric.sectorFamilyCode, metric.sectorFamily, metric.isFinancial);
            peerStats = (relatedFamilyKey ? relatedPeerStatsMap.get(relatedFamilyKey) : undefined)
                ?? buildVfPeerStats(peerContext.peers);
        } else {
            peerStats = sectorPeerStatsMap.get(metric.sector) ?? buildVfPeerStats(peerContext.peers);
        }

        const zThreshold = metric.zModel === 'Original' ? originalThreshold : emsThreshold;
        const zZoneBase = metric.zScore !== null ? zoneFromScore(metric.zScore, zThreshold) : null;
        const zZoneAdjusted = applySOEAdjustment(zZoneBase, snapshot.classification);

        const vfScore = computeVfScore(metric, peerStats);
        const vfPeerScores = peerContext.peers.map((peerMetric) => computeVfScore(peerMetric, peerStats));
        const vfPercentile = computePercentile(vfScore, vfPeerScores);
        const zPercentile = computePercentile(metric.zScore, sectorZValuesMap.get(metric.sector) ?? []);
        const leverageFlag = toLeverageFlag(metric.debtToEquity, peerStats.debtToEquity);
        const baseHealthRating = matrixHealthRating(snapshot, zZoneBase, vfScore, vfPercentile);
        const healthRating = applySoeLabelModifier(baseHealthRating, snapshot, zZoneBase);
        const cutoffLabel = toCutoffLabel(snapshot.current_year, snapshot.quarter_cutoff);
        const cutoffMismatchMajority = majorityCutoff !== null
            && cutoffLabel !== null
            && cutoffLabel !== majorityCutoff;

        rows.push({
            ticker,
            company_name: stock.company_name,
            sector: metric.sector,
            classification: snapshot.classification,
            snapshot,
            scores: {
                z_model: metric.zModel,
                z_score: metric.zScore,
                z_zone_base: zZoneBase,
                z_zone_adjusted: zZoneAdjusted,
                vf_score: vfScore,
                vf_sector_pctile: vfPercentile,
                vf_peer_group: peerContext.peerGroup,
                vf_peer_size: peerContext.peers.length,
                z_sector_pctile: zPercentile,
                leverage_flag: leverageFlag,
                cutoff_label: cutoffLabel,
                cutoff_mismatch_majority: cutoffMismatchMajority,
                vn_health_rating_base: baseHealthRating,
                vn_health_rating: healthRating,
            },
        });
    });

    const nonFinancialLoaded = benchmarkMetrics.filter((m) => !m.isFinancial).length;
    const offMajorityCutoffCount = majorityCutoff !== null
        ? benchmarkMetrics.length - majorityCutoffCount
        : 0;

    return {
        rows,
        stats: {
            benchmark_size: input.benchmarkStocks.length,
            benchmark_loaded: benchmarkMetrics.length,
            benchmark_non_financial_loaded: nonFinancialLoaded,
            failed_count: input.failedTickers.size,
            majority_cutoff: majorityCutoff,
            majority_cutoff_count: majorityCutoffCount,
            majority_cutoff_coverage_pct: benchmarkMetrics.length > 0
                ? majorityCutoffCount / benchmarkMetrics.length
                : null,
            off_majority_cutoff_count: offMajorityCutoffCount,
            thresholds: {
                original: originalThreshold,
                ems: emsThreshold,
            },
            benchmark_quality: {
                is_insufficient: benchmarkQualityInsufficient,
                thresholds: {
                    original_ok: originalThresholdOk,
                    original_n: originalThreshold.sample_count,
                    ems_ok: emsThresholdOk,
                    ems_n: emsThreshold.sample_count,
                },
                fallback_used: fallbackUsed,
                fallback_counts: fallbackCounts,
                low_peer_sectors: lowPeerSectors,
            },
            insufficient_benchmark: benchmarkQualityInsufficient,
        },
    };
};

export const formatPercentile = (value: number | null): string => {
    if (value === null) return '-';
    return `${(value * 100).toFixed(0)}%`;
};
