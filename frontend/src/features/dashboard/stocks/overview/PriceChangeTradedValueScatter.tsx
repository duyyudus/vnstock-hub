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
import type { Stock } from '../../../../api/stockApi';
import type { PortfolioHoldingSummary, TradingPositionSummary } from './StocksTable';

interface PriceChangeTradedValueScatterProps {
    stocks: Stock[];
    portfolioHoldings?: Record<string, PortfolioHoldingSummary>;
    openTradingPositions?: Record<string, TradingPositionSummary>;
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
    holding?: PortfolioHoldingSummary;
    tradingPosition?: TradingPositionSummary;
    tradingValue?: number;
    tradingPnl?: number;
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
const MAJOR_PRICE_CHANGE_POINTS = [-7, -5, -3, -1, 1, 3, 5, 7];
const PRICE_CHANGE_AXIS_TICKS = [-7, -5, -3, -1, 0, 1, 3, 5, 7];
const LOW_VALUE_REFERENCE_TICKS = [20, 50, 100, 200, 300, 400, 500];

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

const formatHoldingPercent = (value: number): string => {
    return new Intl.NumberFormat('en-US', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    }).format(value);
};

const formatHoldingValue = (value: number): string => {
    return new Intl.NumberFormat('en-US', {
        maximumFractionDigits: 2,
    }).format(value);
};

const formatSignedValue = (value: number): string => {
    const formatted = new Intl.NumberFormat('en-US', {
        maximumFractionDigits: 2,
    }).format(Math.abs(value));
    const prefix = value > 0 ? '+' : value < 0 ? '-' : '';
    return `${prefix}${formatted}`;
};

const formatSignedPercent = (value: number): string => {
    const formatted = new Intl.NumberFormat('en-US', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    }).format(Math.abs(value));
    const prefix = value > 0 ? '+' : value < 0 ? '-' : '';
    return `${prefix}${formatted}%`;
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
    const amplitude = Math.max(
        Math.abs(min),
        Math.abs(max),
        Math.max(...MAJOR_PRICE_CHANGE_POINTS.map((value) => Math.abs(value))),
    );
    const padded = amplitude * 1.15;
    return [-Number(padded.toFixed(2)), Number(padded.toFixed(2))];
};

const getValueDomain = (points: ScatterPoint[]): [number, number] => {
    const values = points.map((point) => point.y).filter((value) => value > 0);
    if (values.length === 0) {
        return [0, 10];
    }

    const min = Math.min(...values);
    const max = Math.max(...values);
    if (min === max) {
        return [0, max * 1.15];
    }
    return [0, max * 1.1];
};

const getNiceStep = (rawStep: number): number => {
    if (!Number.isFinite(rawStep) || rawStep <= 0) {
        return 1;
    }

    const magnitude = Math.pow(10, Math.floor(Math.log10(rawStep)));
    const normalized = rawStep / magnitude;

    if (normalized <= 1.5) return magnitude;
    if (normalized <= 2.25) return 2 * magnitude;
    if (normalized <= 3.75) return 2.5 * magnitude;
    if (normalized <= 7.5) return 5 * magnitude;
    return 10 * magnitude;
};

const buildValueTicks = (domain: [number, number], targetTickCount = 7): number[] => {
    const [min, max] = domain;
    if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) {
        return [0, Math.max(1, Number(max.toFixed(2)) || 1)];
    }

    const buildTicksForStep = (step: number): number[] => {
        const tickStart = Math.ceil(min / step) * step;
        const ticks: number[] = [];

        for (let value = tickStart; value <= max + (step * 0.001); value += step) {
            const roundedValue = Number(value.toFixed(6));
            if (roundedValue >= min && roundedValue <= max) {
                ticks.push(roundedValue);
            }
        }

        return ticks;
    };

    const lowValueTicks = LOW_VALUE_REFERENCE_TICKS.filter((value) => value >= min && value <= max);
    const highTickStart = Math.max(min, LOW_VALUE_REFERENCE_TICKS[LOW_VALUE_REFERENCE_TICKS.length - 1]);
    const highTickRange = max - highTickStart;
    const highStep = getNiceStep(highTickRange / 4);
    let ticks = highTickRange > 0
        ? [...lowValueTicks, ...buildTicksForStep(highStep).filter((value) => value > highTickStart)]
        : lowValueTicks;

    if (ticks.length < 4) {
        const fallbackStep = getNiceStep((max - min) / Math.max(targetTickCount - 1, 1));
        ticks = buildTicksForStep(fallbackStep / 2);
    }

    return Array.from(new Set(ticks)).sort((a, b) => a - b);
};

const getPortfolioTooltipLines = (point: ScatterPoint): string[] => {
    const lines: string[] = [];
    const holding = point.holding;
    const tradingPosition = point.tradingPosition;
    const hasHolding = Boolean(
        holding
        && Number.isFinite(holding.marketValue)
        && Number.isFinite(holding.allocationPercent)
    );
    const hasHoldingPnl = Boolean(
        holding
        && Number.isFinite(holding.pnl)
    );
    const holdingPnlPercent = holding?.costBasis != null && holding.costBasis > 0 && holding.pnl != null
        ? (holding.pnl / holding.costBasis) * 100
        : null;
    const hasTradingValue = Number.isFinite(point.tradingValue);
    const hasTradingPnl = Number.isFinite(point.tradingPnl);
    const tradingPnlPercent = tradingPosition?.costBasis != null && tradingPosition.costBasis > 0 && point.tradingPnl != null
        ? (point.tradingPnl / tradingPosition.costBasis) * 100
        : null;

    if (hasHolding && holding) {
        lines.push(`Holding: ${formatHoldingPercent(holding.allocationPercent)}%`);
        lines.push(`Value: ${formatHoldingValue(holding.marketValue)}`);
    }
    if (hasHoldingPnl && holding && holding.pnl != null) {
        const holdingPnlLabel = holdingPnlPercent != null
            ? `${formatSignedValue(holding.pnl)} (${formatSignedPercent(holdingPnlPercent)})`
            : formatSignedValue(holding.pnl);
        lines.push(`Holding P&L: ${holdingPnlLabel}`);
    }
    if (hasTradingValue && point.tradingValue != null) {
        lines.push(`Trading value: ${formatHoldingValue(point.tradingValue)}`);
    }
    if (hasTradingPnl && point.tradingPnl != null) {
        const tradingPnlLabel = tradingPnlPercent != null
            ? `${formatSignedValue(point.tradingPnl)} (${formatSignedPercent(tradingPnlPercent)})`
            : formatSignedValue(point.tradingPnl);
        lines.push(`Trading P&L: ${tradingPnlLabel}`);
    }

    return lines;
};

const ScatterTooltip: React.FC<ScatterTooltipProps> = ({ active, payload }) => {
    const point = payload?.[0]?.payload;
    if (!active || !point) {
        return null;
    }
    const portfolioLines = getPortfolioTooltipLines(point);

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
            {portfolioLines.length > 0 ? (
                <div className="mt-2 border-t border-base-300 pt-2">
                    {portfolioLines.map((line) => (
                        <p key={line} className="mt-1 text-xs">
                            {line}
                        </p>
                    ))}
                </div>
            ) : null}
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

export const PriceChangeTradedValueScatter: React.FC<PriceChangeTradedValueScatterProps> = ({
    stocks,
    portfolioHoldings = {},
    openTradingPositions = {},
}) => {
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
                const normalizedTicker = stock.ticker.toUpperCase();
                const holding = portfolioHoldings[normalizedTicker];
                const tradingPosition = openTradingPositions[normalizedTicker];
                const tradingValue = tradingPosition ? stock.price * tradingPosition.quantity : undefined;
                const tradingPnl = tradingPosition?.costBasis != null && tradingValue != null
                    ? tradingValue - tradingPosition.costBasis
                    : undefined;

                return {
                    ticker: normalizedTicker,
                    name: stock.company_name || normalizedTicker,
                    x: change,
                    y: tradedValue,
                    z: safeMarketCap,
                    marketCap: safeMarketCap,
                    foreignNet,
                    color: getFlowColor(foreignNet),
                    fillOpacity,
                    strokeOpacity: Math.min(fillOpacity + 0.12, 1),
                    holding,
                    tradingPosition,
                    tradingValue,
                    tradingPnl,
                };
            })
            .filter((entry): entry is ScatterPoint => entry !== null)
            .sort((a, b) => b.marketCap - a.marketCap);
    }, [openTradingPositions, portfolioHoldings, stocks]);

    const xDomain = useMemo(() => getPercentDomain(chartData), [chartData]);
    const yDomain = useMemo(() => getValueDomain(chartData), [chartData]);
    const yTicks = useMemo(() => buildValueTicks(yDomain), [yDomain]);
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
                            ticks={PRICE_CHANGE_AXIS_TICKS}
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
                            scale="sqrt"
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

                        {MAJOR_PRICE_CHANGE_POINTS.map((value) => (
                            <ReferenceLine
                                key={value}
                                x={value}
                                stroke="currentColor"
                                strokeDasharray="3 3"
                                strokeOpacity={0.12}
                            />
                        ))}
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
