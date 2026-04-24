import React, { useEffect, useMemo, useState } from 'react';
import {
    Bar,
    CartesianGrid,
    Cell,
    ComposedChart,
    Line,
    ReferenceLine,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from 'recharts';
import { stockApi } from '../../../api/stockApi';
import type { IndexContributionResponse, IndexContributionRow } from '../../../api/stockApi';
import type { IndexConfig } from './indexConfig';

interface IndexContributionProps {
    selectedIndex: IndexConfig | null;
}

type ContributionMode = 'points' | 'percent';

interface WaterfallDatum {
    ticker: string;
    companyName: string;
    base: number;
    size: number;
    start: number;
    end: number;
    connector: number | null;
    contribution: number;
    sessionReturn: number;
    effectiveWeight: number;
    freeFloatRatio: number;
    cappingFactor: number;
    isTotal: boolean;
    hasFallback: boolean;
}

interface TooltipPayloadEntry {
    payload?: WaterfallDatum;
}

interface ContributionTooltipProps {
    active?: boolean;
    payload?: TooltipPayloadEntry[];
    mode: ContributionMode;
}

const POSITIVE_COLOR = '#16a34a';
const NEGATIVE_COLOR = '#e11d48';
const TOTAL_COLOR = '#1d4ed8';
const MAX_VISIBLE_ROWS = 30;

const formatSigned = (value: number, fractionDigits = 2): string => {
    const sign = value > 0 ? '+' : value < 0 ? '-' : '';
    return `${sign}${Math.abs(value).toFixed(fractionDigits)}`;
};

const formatContribution = (value: number | null, mode: ContributionMode): string => {
    if (value === null || !Number.isFinite(value)) {
        return 'N/A';
    }
    return mode === 'points'
        ? `${formatSigned(value, 2)} pts`
        : `${formatSigned(value, 3)}%`;
};

const formatPercent = (value: number): string => `${formatSigned(value, 2)}%`;

const getDatumColor = (datum: WaterfallDatum): string => {
    if (datum.isTotal) return TOTAL_COLOR;
    return datum.contribution >= 0 ? POSITIVE_COLOR : NEGATIVE_COLOR;
};

const getModeValue = (row: IndexContributionRow, mode: ContributionMode): number | null => {
    return mode === 'points' ? row.point_contribution : row.percent_contribution;
};

const buildVisibleRows = (rows: IndexContributionRow[], mode: ContributionMode): IndexContributionRow[] => {
    const rowsWithValues = rows.filter((row) => getModeValue(row, mode) !== null);
    if (rowsWithValues.length <= MAX_VISIBLE_ROWS) {
        return rowsWithValues;
    }

    const visible = rowsWithValues.slice(0, MAX_VISIBLE_ROWS);
    const remaining = rowsWithValues.slice(MAX_VISIBLE_ROWS);
    const otherPointContribution = remaining.some((row) => row.point_contribution !== null)
        ? remaining.reduce((total, row) => total + (row.point_contribution ?? 0), 0)
        : null;

    visible.push({
        ticker: 'Other',
        company_name: `${remaining.length} remaining stocks`,
        price: 0,
        prior_price: 0,
        session_return: 0,
        outstanding_shares: 0,
        free_float_ratio: 1,
        capping_factor: 1,
        effective_weight: remaining.reduce((total, row) => total + row.effective_weight, 0),
        percent_contribution: remaining.reduce((total, row) => total + row.percent_contribution, 0),
        point_contribution: otherPointContribution,
        missing_outstanding_shares: remaining.some((row) => row.missing_outstanding_shares),
        missing_free_float: remaining.some((row) => row.missing_free_float),
        used_market_cap_shares_fallback: remaining.some((row) => row.used_market_cap_shares_fallback),
    });

    return visible;
};

const buildWaterfallData = (rows: IndexContributionRow[], mode: ContributionMode): WaterfallDatum[] => {
    let cumulative = 0;
    const visibleRows = buildVisibleRows(rows, mode);
    const data: WaterfallDatum[] = visibleRows.map((row) => {
        const contribution = getModeValue(row, mode) ?? 0;
        const start = cumulative;
        const end = cumulative + contribution;
        cumulative = end;

        return {
            ticker: row.ticker,
            companyName: row.company_name,
            base: Math.min(start, end),
            size: Math.abs(contribution),
            start,
            end,
            connector: end,
            contribution,
            sessionReturn: row.session_return,
            effectiveWeight: row.effective_weight,
            freeFloatRatio: row.free_float_ratio,
            cappingFactor: row.capping_factor,
            isTotal: false,
            hasFallback: row.missing_free_float || row.missing_outstanding_shares || row.used_market_cap_shares_fallback,
        };
    });

    data.push({
        ticker: 'Total',
        companyName: 'Basket total',
        base: Math.min(0, cumulative),
        size: Math.abs(cumulative),
        start: 0,
        end: cumulative,
        connector: null,
        contribution: cumulative,
        sessionReturn: 0,
        effectiveWeight: 1,
        freeFloatRatio: 1,
        cappingFactor: 1,
        isTotal: true,
        hasFallback: false,
    });

    return data;
};

const ContributionTooltip: React.FC<ContributionTooltipProps> = ({ active, payload, mode }) => {
    const datum = payload?.[0]?.payload;
    if (!active || !datum) {
        return null;
    }

    return (
        <div className="rounded-lg border border-base-300 bg-base-100 p-3 shadow-lg">
            <p className="text-sm font-semibold">{datum.ticker}</p>
            <p className="mt-1 max-w-64 whitespace-normal text-xs text-base-content/70">{datum.companyName}</p>
            <div className="mt-3 space-y-1 text-xs">
                <p>
                    Contribution:{' '}
                    <span className={datum.contribution >= 0 ? 'font-medium text-success' : 'font-medium text-error'}>
                        {formatContribution(datum.contribution, mode)}
                    </span>
                </p>
                {!datum.isTotal ? (
                    <>
                        <p>Session return: <span className="font-medium">{formatPercent(datum.sessionReturn)}</span></p>
                        <p>Effective weight: <span className="font-medium">{(datum.effectiveWeight * 100).toFixed(2)}%</span></p>
                        <p>Free float used: <span className="font-medium">{(datum.freeFloatRatio * 100).toFixed(0)}%</span></p>
                        <p>Capping factor: <span className="font-medium">{datum.cappingFactor.toFixed(3)}</span></p>
                    </>
                ) : null}
            </div>
        </div>
    );
};

export const IndexContribution: React.FC<IndexContributionProps> = ({ selectedIndex }) => {
    const [response, setResponse] = useState<IndexContributionResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [mode, setMode] = useState<ContributionMode>('points');

    useEffect(() => {
        if (!selectedIndex) {
            setResponse(null);
            return;
        }

        let isMounted = true;
        const fetchContribution = async () => {
            try {
                setLoading(true);
                setError(null);
                const data = await stockApi.getIndexContribution(selectedIndex.apiEndpoint);
                if (!isMounted) return;
                setResponse(data);
                setMode(data.rows.some((row) => row.point_contribution !== null) ? 'points' : 'percent');
            } catch (err) {
                console.error('Failed to fetch index contribution:', err);
                if (!isMounted) return;
                setError('Failed to load index contribution data.');
            } finally {
                if (isMounted) {
                    setLoading(false);
                }
            }
        };

        fetchContribution();
        return () => {
            isMounted = false;
        };
    }, [selectedIndex]);

    const canShowPoints = response?.rows.some((row) => row.point_contribution !== null) ?? false;
    const effectiveMode = mode === 'points' && !canShowPoints ? 'percent' : mode;
    const chartData = useMemo(
        () => buildWaterfallData(response?.rows ?? [], effectiveMode),
        [effectiveMode, response?.rows],
    );
    const totalValue = chartData[chartData.length - 1]?.contribution ?? 0;
    const positiveValue = response
        ? effectiveMode === 'points'
            ? response.totals.positive_points
            : response.totals.positive_percent
        : null;
    const negativeValue = response
        ? effectiveMode === 'points'
            ? response.totals.negative_points
            : response.totals.negative_percent
        : null;
    const fallbackCount = response
        ? Math.max(response.totals.missing_free_float_count, response.totals.missing_outstanding_shares_count)
        : 0;

    if (!selectedIndex) {
        return (
            <div className="flex h-64 items-center justify-center text-base-content/50">
                Select an index to view contribution.
            </div>
        );
    }

    if (loading) {
        return (
            <div className="flex h-64 flex-col items-center justify-center gap-3 text-base-content/60">
                <span className="loading loading-spinner loading-md text-primary"></span>
                Loading index contribution...
            </div>
        );
    }

    if (error) {
        return <div className="alert alert-error text-sm">{error}</div>;
    }

    if (!response || response.rows.length === 0) {
        return (
            <div className="flex h-64 items-center justify-center text-base-content/50">
                No current-session contribution data available.
            </div>
        );
    }

    return (
        <section className="w-full space-y-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                    <h3 className="text-lg font-bold">Contribution to {response.symbol} Index Move</h3>
                    <p className="text-sm text-base-content/60">Today</p>
                </div>
                <div className="join">
                    <button
                        type="button"
                        className={`join-item btn btn-sm ${effectiveMode === 'points' ? 'btn-primary' : 'btn-ghost'}`}
                        onClick={() => setMode('points')}
                        disabled={!canShowPoints}
                    >
                        Index points
                    </button>
                    <button
                        type="button"
                        className={`join-item btn btn-sm ${effectiveMode === 'percent' ? 'btn-primary' : 'btn-ghost'}`}
                        onClick={() => setMode('percent')}
                    >
                        Percent (%)
                    </button>
                </div>
            </div>

            {fallbackCount > 0 || response.totals.excluded_count > 0 ? (
                <div className="alert alert-warning py-2 text-xs">
                    <span>
                        Official divisor/capping factors are not exposed by the source. {fallbackCount} row(s) used fallback shares or free-float data; {response.totals.excluded_count} row(s) were excluded for missing price/change data.
                    </span>
                </div>
            ) : null}

            <div className="h-[560px] min-w-0">
                <ResponsiveContainer width="100%" height="100%" debounce={50}>
                    <ComposedChart data={chartData} margin={{ top: 28, right: 18, left: 10, bottom: 36 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.12} />
                        <XAxis
                            dataKey="ticker"
                            tick={{ fontSize: 11, fontWeight: 600 }}
                            stroke="currentColor"
                            opacity={0.7}
                            interval={0}
                            angle={chartData.length > 16 ? -35 : 0}
                            textAnchor={chartData.length > 16 ? 'end' : 'middle'}
                            height={chartData.length > 16 ? 72 : 40}
                        />
                        <YAxis
                            width={60}
                            tick={{ fontSize: 11 }}
                            stroke="currentColor"
                            opacity={0.65}
                            tickFormatter={(value: number) => effectiveMode === 'points' ? value.toFixed(1) : `${value.toFixed(1)}%`}
                        />
                        <Tooltip content={<ContributionTooltip mode={effectiveMode} />} isAnimationActive={false} />
                        <ReferenceLine y={0} stroke="currentColor" strokeOpacity={0.45} />
                        <Bar dataKey="base" stackId="waterfall" fill="transparent" isAnimationActive={false} />
                        <Bar dataKey="size" stackId="waterfall" maxBarSize={48} isAnimationActive={false}>
                            {chartData.map((datum) => (
                                <Cell key={datum.ticker} fill={getDatumColor(datum)} />
                            ))}
                        </Bar>
                        <Line
                            type="stepAfter"
                            dataKey="connector"
                            stroke="currentColor"
                            strokeDasharray="4 3"
                            strokeOpacity={0.45}
                            dot={false}
                            activeDot={false}
                            isAnimationActive={false}
                            connectNulls={false}
                        />
                    </ComposedChart>
                </ResponsiveContainer>
            </div>

            <div className="grid gap-3 md:grid-cols-3">
                <div className="rounded-lg border border-base-300 bg-base-100 p-4 text-center">
                    <p className="text-sm text-base-content/70">Total Contribution</p>
                    <p className={totalValue >= 0 ? 'mt-1 text-2xl font-bold text-success' : 'mt-1 text-2xl font-bold text-error'}>
                        {formatContribution(totalValue, effectiveMode)}
                    </p>
                    {response.change !== null ? (
                        <p className="mt-1 text-xs text-base-content/60">{response.symbol} {formatPercent(response.change)}</p>
                    ) : null}
                </div>
                <div className="rounded-lg border border-success/20 bg-success/5 p-4 text-center">
                    <p className="text-sm text-base-content/70">Positive Contribution</p>
                    <p className="mt-1 text-2xl font-bold text-success">{formatContribution(positiveValue, effectiveMode)}</p>
                </div>
                <div className="rounded-lg border border-error/20 bg-error/5 p-4 text-center">
                    <p className="text-sm text-base-content/70">Negative Contribution</p>
                    <p className="mt-1 text-2xl font-bold text-error">{formatContribution(negativeValue, effectiveMode)}</p>
                </div>
            </div>
        </section>
    );
};

export default IndexContribution;
