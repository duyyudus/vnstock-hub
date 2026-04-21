import React, { useMemo, useState } from 'react';
import {
    CartesianGrid,
    ReferenceLine,
    ResponsiveContainer,
    Scatter,
    ScatterChart,
    Tooltip,
    XAxis,
    YAxis,
    ZAxis,
} from 'recharts';
import type { Stock } from '../../../api/stockApi';

interface PriceChangeTradedValueScatterProps {
    stocks: Stock[];
}

interface ScatterPoint {
    ticker: string;
    name: string;
    x: number;
    y: number;
    z: number;
    marketCap: number;
    foreignNet: number | null;
    color: string;
    fillOpacity: number;
    strokeOpacity: number;
}

interface TooltipPayloadEntry {
    payload: ScatterPoint;
}

interface ScatterTooltipProps {
    active?: boolean;
    payload?: TooltipPayloadEntry[];
}

interface ScatterShapeProps {
    cx?: number;
    cy?: number;
    size?: number;
    payload?: ScatterPoint;
}

const POSITIVE_FLOW_COLOR = '#16a34a';
const NEGATIVE_FLOW_COLOR = '#ef4444';
const NEUTRAL_FLOW_COLOR = '#94a3b8';
const MIN_BUBBLE_MARKET_CAP = 1;
const CHART_MARGIN = { top: 28, right: 34, left: 24, bottom: 46 };
const MIN_FLOW_OPACITY = 0.38;
const MAX_FLOW_OPACITY = 0.95;

const formatPercent = (value: number): string => {
    const prefix = value > 0 ? '+' : '';
    return `${prefix}${value.toFixed(2)}%`;
};

const formatBilVnd = (value: number): string => {
    const absolute = Math.abs(value);
    if (absolute >= 1000) return `${value < 0 ? '-' : ''}${(absolute / 1000).toFixed(1)}K`;
    if (absolute >= 100) return `${value < 0 ? '-' : ''}${absolute.toFixed(0)}`;
    if (absolute >= 10) return `${value < 0 ? '-' : ''}${absolute.toFixed(1)}`;
    return `${value < 0 ? '-' : ''}${absolute.toFixed(2)}`;
};

const formatSignedBilVnd = (value: number | null): string => {
    if (value === null) {
        return 'N/A';
    }
    const formatted = new Intl.NumberFormat('en-US', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    }).format(Math.abs(value));
    return `${value >= 0 ? '+' : '-'}${formatted} Bil VND`;
};

const getForeignNet = (stock: Stock): number | null => {
    if (stock.foreign_buy_value == null || stock.foreign_sell_value == null) {
        return null;
    }

    const value = stock.foreign_buy_value - stock.foreign_sell_value;
    return Number.isFinite(value) ? value : null;
};

const getFlowColor = (foreignNet: number | null): string => {
    if (foreignNet == null || foreignNet === 0) {
        return NEUTRAL_FLOW_COLOR;
    }
    if (foreignNet > 0) {
        return POSITIVE_FLOW_COLOR;
    }
    return NEGATIVE_FLOW_COLOR;
};

const getMedian = (values: number[]): number => {
    if (values.length === 0) {
        return 0;
    }

    const sorted = [...values].sort((a, b) => a - b);
    const midpoint = Math.floor(sorted.length / 2);
    if (sorted.length % 2 === 0) {
        return (sorted[midpoint - 1] + sorted[midpoint]) / 2;
    }
    return sorted[midpoint];
};

const getQuantile = (values: number[], quantile: number): number => {
    if (values.length === 0) {
        return 0;
    }

    const sorted = [...values].sort((a, b) => a - b);
    const index = Math.min(sorted.length - 1, Math.max(0, Math.floor((sorted.length - 1) * quantile)));
    return sorted[index];
};

const getFlowOpacity = (foreignNet: number | null, referenceValue: number): number => {
    if (foreignNet == null || foreignNet === 0 || referenceValue <= 0) {
        return MIN_FLOW_OPACITY;
    }

    const normalized = Math.min(Math.abs(foreignNet) / referenceValue, 1);
    return MIN_FLOW_OPACITY + ((MAX_FLOW_OPACITY - MIN_FLOW_OPACITY) * normalized);
};

const getPercentDomain = (points: ScatterPoint[]): [number, number] => {
    const values = points.map((point) => point.x);
    const min = Math.min(...values, 0);
    const max = Math.max(...values, 0);
    const amplitude = Math.max(Math.abs(min), Math.abs(max), 1);
    const padded = amplitude * 1.15;
    return [-Number(padded.toFixed(2)), Number(padded.toFixed(2))];
};

const getValueDomain = (points: ScatterPoint[]): [number, number] => {
    const values = points.map((point) => point.y).filter((value) => value > 0);
    if (values.length === 0) {
        return [1, 10];
    }

    const min = Math.min(...values);
    const max = Math.max(...values);
    if (min === max) {
        return [Math.max(min * 0.5, 0.01), max * 1.5];
    }
    return [Math.max(min * 0.75, 0.01), max * 1.25];
};

const buildLogTicks = (domain: [number, number], referenceValue: number): number[] => {
    const [min, max] = domain;
    const tickSet = new Set<number>();
    const startPower = Math.floor(Math.log10(Math.max(min, 0.01)));
    const endPower = Math.ceil(Math.log10(Math.max(max, 1)));

    for (let power = startPower; power <= endPower; power += 1) {
        [1, 2, 5].forEach((multiplier) => {
            const value = multiplier * Math.pow(10, power);
            if (value >= min && value <= max) {
                tickSet.add(Number(value.toFixed(2)));
            }
        });
    }

    if (referenceValue >= min && referenceValue <= max) {
        tickSet.add(Number(referenceValue.toFixed(2)));
    }

    return Array.from(tickSet).sort((a, b) => a - b);
};

const ScatterTooltip: React.FC<ScatterTooltipProps> = ({ active, payload }) => {
    const point = payload?.[0]?.payload;
    if (!active || !point) {
        return null;
    }

    return (
        <div className="rounded-lg border border-base-300 bg-base-100 p-3 shadow-lg">
            <div className="mb-2">
                <p className="text-sm font-semibold">{point.ticker}</p>
                <p className="text-xs text-base-content/70">{point.name}</p>
            </div>
            <p className={point.x > 0 ? 'text-xs text-success' : point.x < 0 ? 'text-xs text-error' : 'text-xs'}>
                Daily change: <span className="font-medium">{formatPercent(point.x)}</span>
            </p>
            <p className="mt-1 text-xs">
                Traded value: <span className="font-medium">{formatBilVnd(point.y)} Bil VND</span>
            </p>
            <p className="mt-1 text-xs">
                Market cap: <span className="font-medium">{formatBilVnd(point.marketCap)} Bil VND</span>
            </p>
            <p className="mt-1 text-xs" style={{ color: point.color }}>
                Foreign net: <span className="font-medium">{formatSignedBilVnd(point.foreignNet)}</span>
            </p>
        </div>
    );
};

const BubblePoint = (props: ScatterShapeProps) => {
    const { cx, cy, size, payload } = props;
    if (cx == null || cy == null || !payload) {
        return null;
    }

    const radius = Math.max(5, Math.sqrt(Math.max(size ?? 90, 1) / Math.PI));
    const labelX = cx + radius + 5;

    return (
        <g>
            <circle
                cx={cx}
                cy={cy}
                r={radius}
                fill={payload.color}
                fillOpacity={payload.fillOpacity}
                stroke={payload.color}
                strokeOpacity={payload.strokeOpacity}
            />
            <text
                x={labelX}
                y={cy}
                dy="0.35em"
                fill="currentColor"
                fontSize={11}
                fontWeight={600}
                pointerEvents="none"
            >
                {payload.ticker}
            </text>
        </g>
    );
};

const formatCount = (value: number): string => new Intl.NumberFormat('en-US').format(value);

export const PriceChangeTradedValueScatter: React.FC<PriceChangeTradedValueScatterProps> = ({ stocks }) => {
    const [activePoint, setActivePoint] = useState<ScatterPoint | null>(null);

    const chartData = useMemo<ScatterPoint[]>(() => {
        const foreignFlowScale = getQuantile(
            stocks
                .map((stock) => getForeignNet(stock))
                .filter((value): value is number => value !== null && value !== 0)
                .map((value) => Math.abs(value)),
            0.9,
        );

        return stocks
            .map((stock): ScatterPoint | null => {
                const change = Number(stock.price_change_24h);
                const tradedValue = Number(stock.accumulated_value);
                if (!Number.isFinite(change) || !Number.isFinite(tradedValue) || tradedValue <= 0) {
                    return null;
                }

                const marketCap = Number(stock.market_cap);
                const safeMarketCap = Number.isFinite(marketCap) && marketCap > 0
                    ? marketCap
                    : MIN_BUBBLE_MARKET_CAP;
                const foreignNet = getForeignNet(stock);
                const fillOpacity = getFlowOpacity(foreignNet, foreignFlowScale);

                return {
                    ticker: stock.ticker.toUpperCase(),
                    name: stock.company_name || stock.ticker.toUpperCase(),
                    x: change,
                    y: tradedValue,
                    z: safeMarketCap,
                    marketCap: safeMarketCap,
                    foreignNet,
                    color: getFlowColor(foreignNet),
                    fillOpacity,
                    strokeOpacity: Math.min(fillOpacity + 0.12, 1),
                };
            })
            .filter((entry): entry is ScatterPoint => entry !== null)
            .sort((a, b) => b.marketCap - a.marketCap);
    }, [stocks]);

    const xDomain = useMemo(() => getPercentDomain(chartData), [chartData]);
    const yDomain = useMemo(() => getValueDomain(chartData), [chartData]);
    const highValueReference = useMemo(
        () => getMedian(chartData.map((point) => point.y)),
        [chartData],
    );
    const yTicks = useMemo(() => buildLogTicks(yDomain, highValueReference), [highValueReference, yDomain]);
    const marketCapDomain = useMemo<[number, number]>(() => {
        if (chartData.length === 0) {
            return [MIN_BUBBLE_MARKET_CAP, MIN_BUBBLE_MARKET_CAP];
        }

        const values = chartData.map((point) => point.marketCap);
        return [Math.min(...values), Math.max(...values)];
    }, [chartData]);
    const priceChangeStats = useMemo(() => {
        return chartData.reduce(
            (accumulator, point) => {
                if (point.x < -1) {
                    accumulator.down += 1;
                } else if (point.x > 1) {
                    accumulator.up += 1;
                } else {
                    accumulator.flat += 1;
                }
                return accumulator;
            },
            { down: 0, flat: 0, up: 0 },
        );
    }, [chartData]);

    if (stocks.length === 0) {
        return (
            <div className="flex items-center justify-center rounded-lg border border-base-300 bg-base-100 h-64 text-base-content/50">
                No stocks available for price change analysis.
            </div>
        );
    }

    if (chartData.length === 0) {
        return (
            <div className="flex items-center justify-center rounded-lg border border-base-300 bg-base-100 h-64 text-base-content/50">
                No current-session price and traded value data available.
            </div>
        );
    }

    return (
        <section className="w-full rounded-lg border border-base-300 bg-base-100">
            <div className="grid gap-3 px-3 py-2 lg:grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] lg:items-center">
                <div>
                    <h3 className="text-base font-semibold">Price Change vs Traded Value</h3>
                </div>
                <div className="flex flex-wrap items-center justify-center gap-x-4 gap-y-2 text-xs text-base-content/70">
                    <span>
                        Below -1%: <span className="font-semibold text-error">{formatCount(priceChangeStats.down)}</span>
                    </span>
                    <span>
                        -1% to 1%: <span className="font-semibold text-base-content">{formatCount(priceChangeStats.flat)}</span>
                    </span>
                    <span>
                        Above 1%: <span className="font-semibold text-success">{formatCount(priceChangeStats.up)}</span>
                    </span>
                </div>
                <div className="flex flex-wrap items-center justify-end gap-x-4 gap-y-2 text-xs text-base-content/70">
                    <span className="flex items-center gap-1">
                        <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: POSITIVE_FLOW_COLOR }}></span>
                        Foreign net buy
                    </span>
                    <span className="flex items-center gap-1">
                        <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: NEGATIVE_FLOW_COLOR }}></span>
                        Foreign net sell
                    </span>
                    <span className="flex items-center gap-1">
                        <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: NEUTRAL_FLOW_COLOR }}></span>
                        Neutral / N/A
                    </span>
                </div>
            </div>

            <div className="aspect-[3/2] min-w-0">
                <ResponsiveContainer width="100%" height="100%" debounce={50}>
                    <ScatterChart margin={CHART_MARGIN}>
                        <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.12} />
                        <XAxis
                            type="number"
                            dataKey="x"
                            domain={xDomain}
                            tick={{ fontSize: 11 }}
                            stroke="currentColor"
                            opacity={0.55}
                            tickFormatter={(value: number) => `${value.toFixed(0)}%`}
                            label={{
                                value: 'Daily % Change',
                                position: 'bottom',
                                style: { fontSize: 12, fontWeight: 600 },
                            }}
                        />
                        <YAxis
                            type="number"
                            dataKey="y"
                            domain={yDomain}
                            scale="log"
                            allowDataOverflow
                            ticks={yTicks}
                            width={62}
                            tick={{ fontSize: 11 }}
                            stroke="currentColor"
                            opacity={0.55}
                            tickFormatter={(value: number) => formatBilVnd(value)}
                            label={{
                                value: 'Traded Value (Bil VND)',
                                angle: -90,
                                position: 'insideLeft',
                                style: { fontSize: 12, fontWeight: 600 },
                            }}
                        />
                        <ZAxis
                            type="number"
                            dataKey="z"
                            domain={marketCapDomain}
                            range={[90, 1400]}
                        />
                        <Tooltip
                            content={<ScatterTooltip />}
                            cursor={false}
                            isAnimationActive={false}
                        />

                        <ReferenceLine x={0} stroke="#94a3b8" strokeOpacity={0.65} />
                        {activePoint ? (
                            <>
                                <ReferenceLine x={activePoint.x} stroke="#0f172a" strokeDasharray="3 3" strokeOpacity={0.45} />
                                <ReferenceLine y={activePoint.y} stroke="#0f172a" strokeDasharray="3 3" strokeOpacity={0.45} />
                            </>
                        ) : null}

                        <Scatter
                            name="Stocks"
                            data={chartData}
                            shape={<BubblePoint />}
                            isAnimationActive={false}
                            onMouseLeave={() => setActivePoint(null)}
                            onMouseMove={(event: ScatterPoint) => setActivePoint(event)}
                        >
                        </Scatter>

                    </ScatterChart>
                </ResponsiveContainer>
            </div>
        </section>
    );
};

export default PriceChangeTradedValueScatter;
