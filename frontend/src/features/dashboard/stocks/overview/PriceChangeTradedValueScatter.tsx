import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
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
    usePlotArea,
} from 'recharts';
import type { PlotArea } from 'recharts';
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
    price: number;
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
    recentTrend?: RecentTrend | null;
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

type RecentTrendDirection = 'up' | 'down' | 'sideways';

interface RecentTrend {
    direction: RecentTrendDirection;
    returnPercent: number;
}

interface ZoomDomains {
    x: [number, number];
    y: [number, number];
}

interface SelectionRect {
    startX: number;
    startY: number;
    currentX: number;
    currentY: number;
}

type ForeignFlowFilter = 'all' | 'buy' | 'sell' | 'neutral';

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
const TREND_UP_COLOR = '#16a34a';
const TREND_DOWN_COLOR = '#dc2626';
const TREND_RING_GAP = 3;
const TREND_RING_STROKE_WIDTH = 1.25;
const MIN_X_ZOOM_SPAN = 0.5;
const MIN_Y_ZOOM_SQRT_SPAN = 0.5;
const BUTTON_ZOOM_FACTOR = 0.6;
const WHEEL_ZOOM_FACTOR = 0.82;
const MIN_DRAG_SELECT_PIXELS = 8;
const PAN_STEP_RATIO = 0.2;

const formatPercent = (value: number): string => {
    const prefix = value > 0 ? '+' : '';
    return `${prefix}${value.toFixed(2)}%`;
};

const formatPrice = (value: number): string => {
    if (!Number.isFinite(value)) {
        return 'N/A';
    }
    return new Intl.NumberFormat('en-US').format(value);
};

const formatTrendLabel = (trend: RecentTrend | null | undefined): string => {
    if (!trend) {
        return 'N/A';
    }

    const label = trend.direction === 'up'
        ? 'Up'
        : trend.direction === 'down'
            ? 'Down'
            : 'Sideways';
    return `${label} (${formatPercent(trend.returnPercent)})`;
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

const formatAllocationSuffix = (allocationPercent: number | null | undefined): string => {
    if (typeof allocationPercent !== 'number' || !Number.isFinite(allocationPercent)) {
        return '';
    }
    return ` | ${formatHoldingPercent(allocationPercent)}%`;
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

const getRecentTrend = (stock: Stock): RecentTrend | null => {
    const direction = stock.recent_trend_3d;
    const returnPercent = stock.recent_trend_3d_return;
    if (
        (direction !== 'up' && direction !== 'down' && direction !== 'sideways')
        || typeof returnPercent !== 'number'
        || !Number.isFinite(returnPercent)
    ) {
        return null;
    }
    return { direction, returnPercent };
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

const buildPercentTicks = (domain: [number, number]): number[] => {
    const presetTicks = PRICE_CHANGE_AXIS_TICKS.filter((value) => value >= domain[0] && value <= domain[1]);
    if (presetTicks.length >= 4) {
        return presetTicks;
    }

    const step = getNiceStep((domain[1] - domain[0]) / 6);
    const ticks: number[] = [];
    for (let value = Math.ceil(domain[0] / step) * step; value <= domain[1] + (step * 0.001); value += step) {
        ticks.push(Number(value.toFixed(6)));
    }
    return ticks;
};

const clampRatio = (value: number): number => Math.min(1, Math.max(0, value));

const pixelToDomainPoint = (
    px: number,
    py: number,
    plotArea: PlotArea,
    xDomain: [number, number],
    yDomain: [number, number],
): { x: number; y: number } => {
    const xRatio = clampRatio((px - plotArea.x) / plotArea.width);
    const yRatio = clampRatio((py - plotArea.y) / plotArea.height);
    const sqrtMin = Math.sqrt(Math.max(yDomain[0], 0));
    const sqrtMax = Math.sqrt(Math.max(yDomain[1], 0));
    const sqrtValue = sqrtMax - (yRatio * (sqrtMax - sqrtMin));
    return {
        x: xDomain[0] + (xRatio * (xDomain[1] - xDomain[0])),
        y: sqrtValue * sqrtValue,
    };
};

const PlotAreaCapture: React.FC<{ onPlotAreaChange: (plotArea: PlotArea | null) => void }> = ({ onPlotAreaChange }) => {
    const plotArea = usePlotArea();

    useEffect(() => {
        onPlotAreaChange(plotArea ?? null);
    }, [onPlotAreaChange, plotArea]);

    return null;
};

const ExpandIcon: React.FC = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" className="h-3.5 w-3.5" aria-hidden="true">
        <path d="M8 3H5a2 2 0 0 0-2 2v3" />
        <path d="M21 8V5a2 2 0 0 0-2-2h-3" />
        <path d="M3 16v3a2 2 0 0 0 2 2h3" />
        <path d="M16 21h3a2 2 0 0 0 2-2v-3" />
    </svg>
);

const ZoomInIcon: React.FC = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" className="h-3.5 w-3.5" aria-hidden="true">
        <circle cx="11" cy="11" r="8" />
        <path d="m21 21-4.35-4.35" />
        <path d="M11 8v6" />
        <path d="M8 11h6" />
    </svg>
);

const ZoomOutIcon: React.FC = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" className="h-3.5 w-3.5" aria-hidden="true">
        <circle cx="11" cy="11" r="8" />
        <path d="m21 21-4.35-4.35" />
        <path d="M8 11h6" />
    </svg>
);

const CloseIcon: React.FC = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" className="h-3.5 w-3.5" aria-hidden="true">
        <path d="M18 6 6 18" />
        <path d="m6 6 12 12" />
    </svg>
);

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
        lines.push(
            `Portfolio value: ${formatHoldingValue(holding.marketValue)}${formatAllocationSuffix(holding.allocationPercent)}`
        );
    }
    if (hasTradingValue && point.tradingValue != null) {
        lines.push(
            `Trading value: ${formatHoldingValue(point.tradingValue)}${formatAllocationSuffix(tradingPosition?.allocationPercent)}`
        );
    }
    if (hasHolding && holding && hasTradingValue && point.tradingValue != null) {
        lines.push(`Total value: ${formatHoldingValue(holding.marketValue + point.tradingValue)}`);
    }
    if (hasHoldingPnl && holding && holding.pnl != null) {
        const holdingPnlLabel = holdingPnlPercent != null
            ? `${formatSignedValue(holding.pnl)} (${formatSignedPercent(holdingPnlPercent)})`
            : formatSignedValue(holding.pnl);
        lines.push(`Portfolio P&L: ${holdingPnlLabel}`);
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
                Market price: <span className="font-medium">{formatPrice(point.price)}{Number.isFinite(point.price) ? ' VND' : ''}</span>
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
            <p className="mt-1 text-xs">
                3-day trend: <span className="font-medium">{formatTrendLabel(point.recentTrend)}</span>
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
    const shouldShowTrendRing = payload.recentTrend?.direction === 'up' || payload.recentTrend?.direction === 'down';
    const trendRingColor = payload.recentTrend?.direction === 'up' ? TREND_UP_COLOR : TREND_DOWN_COLOR;

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
            {shouldShowTrendRing ? (
                <circle
                    cx={cx}
                    cy={cy}
                    r={radius + TREND_RING_GAP}
                    fill="none"
                    stroke={trendRingColor}
                    strokeOpacity={0.95}
                    strokeWidth={TREND_RING_STROKE_WIDTH}
                    pointerEvents="none"
                />
            ) : null}
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
    const [isFullscreen, setIsFullscreen] = useState(false);
    const [zoomDomains, setZoomDomains] = useState<ZoomDomains | null>(null);
    const [selection, setSelection] = useState<SelectionRect | null>(null);
    const [plotArea, setPlotArea] = useState<PlotArea | null>(null);
    const [foreignFlowFilter, setForeignFlowFilter] = useState<ForeignFlowFilter>('all');

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
                    price: Number(stock.price),
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
                    recentTrend: getRecentTrend(stock),
                };
            })
            .filter((entry): entry is ScatterPoint => entry !== null)
            .sort((a, b) => b.marketCap - a.marketCap);
    }, [openTradingPositions, portfolioHoldings, stocks]);

    const foreignFlowCounts = useMemo(() => {
        return chartData.reduce(
            (counts, point) => {
                if (point.foreignNet !== null && point.foreignNet > 0) {
                    counts.buy += 1;
                } else if (point.foreignNet !== null && point.foreignNet < 0) {
                    counts.sell += 1;
                } else {
                    counts.neutral += 1;
                }
                return counts;
            },
            { buy: 0, sell: 0, neutral: 0 },
        );
    }, [chartData]);
    const displayedChartData = useMemo(() => {
        if (foreignFlowFilter === 'all') {
            return chartData;
        }
        return chartData.filter((point) => {
            if (foreignFlowFilter === 'buy') {
                return point.foreignNet !== null && point.foreignNet > 0;
            }
            if (foreignFlowFilter === 'sell') {
                return point.foreignNet !== null && point.foreignNet < 0;
            }
            return point.foreignNet === null || point.foreignNet === 0;
        });
    }, [chartData, foreignFlowFilter]);

    const xDomain = useMemo(() => getPercentDomain(chartData), [chartData]);
    const yDomain = useMemo(() => getValueDomain(chartData), [chartData]);
    const effectiveXDomain = zoomDomains?.x ?? xDomain;
    const effectiveYDomain = zoomDomains?.y ?? yDomain;
    const xTicks = useMemo(() => buildPercentTicks(effectiveXDomain), [effectiveXDomain]);
    const yTicks = useMemo(() => buildValueTicks(effectiveYDomain), [effectiveYDomain]);
    const marketCapDomain = useMemo<[number, number]>(() => {
        if (chartData.length === 0) {
            return [MIN_BUBBLE_MARKET_CAP, MIN_BUBBLE_MARKET_CAP];
        }

        const values = chartData.map((point) => point.marketCap);
        return [Math.min(...values), Math.max(...values)];
    }, [chartData]);
    const priceChangeStats = useMemo(() => {
        return displayedChartData.reduce(
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
    }, [displayedChartData]);

    const toggleForeignFlowFilter = useCallback((filter: Exclude<ForeignFlowFilter, 'all'>) => {
        setForeignFlowFilter((current) => current === filter ? 'all' : filter);
        setActivePoint(null);
    }, []);

    const handlePlotAreaChange = useCallback((nextPlotArea: PlotArea | null) => {
        setPlotArea((previous) => {
            if (
                previous
                && nextPlotArea
                && previous.x === nextPlotArea.x
                && previous.y === nextPlotArea.y
                && previous.width === nextPlotArea.width
                && previous.height === nextPlotArea.height
            ) {
                return previous;
            }
            return nextPlotArea;
        });
    }, []);

    const closeFullscreen = useCallback(() => {
        setIsFullscreen(false);
        setZoomDomains(null);
        setSelection(null);
        setPlotArea(null);
        setActivePoint(null);
    }, []);

    const panBy = useCallback((xDirection: number, yDirection: number) => {
        setZoomDomains((current) => {
            if (!current) {
                return current;
            }

            const [currentX0, currentX1] = current.x;
            const xShift = xDirection * (currentX1 - currentX0) * PAN_STEP_RATIO;
            const clampedXShift = Math.min(Math.max(xShift, xDomain[0] - currentX0), xDomain[1] - currentX1);

            const sqrt0 = Math.sqrt(Math.max(current.y[0], 0));
            const sqrt1 = Math.sqrt(Math.max(current.y[1], 0));
            const baseSqrt0 = Math.sqrt(Math.max(yDomain[0], 0));
            const baseSqrt1 = Math.sqrt(Math.max(yDomain[1], 0));
            const sqrtShift = yDirection * (sqrt1 - sqrt0) * PAN_STEP_RATIO;
            const clampedSqrtShift = Math.min(Math.max(sqrtShift, baseSqrt0 - sqrt0), baseSqrt1 - sqrt1);

            if (clampedXShift === 0 && clampedSqrtShift === 0) {
                return current;
            }
            const nextSqrt0 = sqrt0 + clampedSqrtShift;
            const nextSqrt1 = sqrt1 + clampedSqrtShift;
            return {
                x: [currentX0 + clampedXShift, currentX1 + clampedXShift],
                y: [nextSqrt0 * nextSqrt0, nextSqrt1 * nextSqrt1],
            };
        });
    }, [xDomain, yDomain]);

    useEffect(() => {
        if (!isFullscreen) {
            return;
        }

        const panDirections: Record<string, [number, number]> = {
            ArrowLeft: [-1, 0],
            ArrowRight: [1, 0],
            ArrowUp: [0, 1],
            ArrowDown: [0, -1],
        };
        const handleKeyDown = (event: KeyboardEvent) => {
            if (event.key === 'Escape') {
                closeFullscreen();
                return;
            }
            const direction = panDirections[event.key];
            if (direction) {
                event.preventDefault();
                panBy(direction[0], direction[1]);
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        const previousOverflow = document.body.style.overflow;
        document.body.style.overflow = 'hidden';
        return () => {
            window.removeEventListener('keydown', handleKeyDown);
            document.body.style.overflow = previousOverflow;
        };
    }, [closeFullscreen, isFullscreen, panBy]);

    const applyZoom = useCallback((factor: number, anchor?: { x: number; y: number }) => {
        setZoomDomains((current) => {
            const [currentX0, currentX1] = current?.x ?? xDomain;
            const [currentY0, currentY1] = current?.y ?? yDomain;
            const sqrt0 = Math.sqrt(Math.max(currentY0, 0));
            const sqrt1 = Math.sqrt(Math.max(currentY1, 0));
            if (
                factor < 1
                && ((currentX1 - currentX0) <= MIN_X_ZOOM_SPAN || (sqrt1 - sqrt0) <= MIN_Y_ZOOM_SQRT_SPAN)
            ) {
                return current;
            }

            const anchorX = anchor ? Math.min(Math.max(anchor.x, currentX0), currentX1) : (currentX0 + currentX1) / 2;
            const anchorSqrt = anchor
                ? Math.min(Math.max(Math.sqrt(Math.max(anchor.y, 0)), sqrt0), sqrt1)
                : (sqrt0 + sqrt1) / 2;
            const baseSqrt0 = Math.sqrt(Math.max(yDomain[0], 0));
            const baseSqrt1 = Math.sqrt(Math.max(yDomain[1], 0));

            const nextX: [number, number] = [
                Math.max(xDomain[0], anchorX - ((anchorX - currentX0) * factor)),
                Math.min(xDomain[1], anchorX + ((currentX1 - anchorX) * factor)),
            ];
            const nextSqrt0 = Math.max(baseSqrt0, anchorSqrt - ((anchorSqrt - sqrt0) * factor));
            const nextSqrt1 = Math.min(baseSqrt1, anchorSqrt + ((sqrt1 - anchorSqrt) * factor));
            const nextY: [number, number] = [nextSqrt0 * nextSqrt0, nextSqrt1 * nextSqrt1];

            const coversFullDomain = nextX[0] <= xDomain[0] && nextX[1] >= xDomain[1]
                && nextY[0] <= yDomain[0] && nextY[1] >= yDomain[1];
            return coversFullDomain ? null : { x: nextX, y: nextY };
        });
    }, [xDomain, yDomain]);

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

    const getRelativePoint = (event: React.MouseEvent<HTMLDivElement>): { x: number; y: number } => {
        const rect = event.currentTarget.getBoundingClientRect();
        return { x: event.clientX - rect.left, y: event.clientY - rect.top };
    };

    const handleChartMouseDown = (event: React.MouseEvent<HTMLDivElement>) => {
        if (event.button !== 0) {
            return;
        }
        event.preventDefault();
        const point = getRelativePoint(event);
        setSelection({ startX: point.x, startY: point.y, currentX: point.x, currentY: point.y });
    };

    const handleChartMouseMove = (event: React.MouseEvent<HTMLDivElement>) => {
        if (!selection) {
            return;
        }
        const point = getRelativePoint(event);
        setSelection({ ...selection, currentX: point.x, currentY: point.y });
    };

    const handleChartMouseUp = () => {
        if (!selection) {
            return;
        }
        setSelection(null);
        if (
            !plotArea
            || Math.abs(selection.currentX - selection.startX) < MIN_DRAG_SELECT_PIXELS
            || Math.abs(selection.currentY - selection.startY) < MIN_DRAG_SELECT_PIXELS
        ) {
            return;
        }

        const pointA = pixelToDomainPoint(selection.startX, selection.startY, plotArea, effectiveXDomain, effectiveYDomain);
        const pointB = pixelToDomainPoint(selection.currentX, selection.currentY, plotArea, effectiveXDomain, effectiveYDomain);

        let x0 = Math.min(pointA.x, pointB.x);
        let x1 = Math.max(pointA.x, pointB.x);
        if (x1 - x0 < MIN_X_ZOOM_SPAN) {
            const centerX = (x0 + x1) / 2;
            x0 = Math.max(xDomain[0], centerX - (MIN_X_ZOOM_SPAN / 2));
            x1 = Math.min(xDomain[1], centerX + (MIN_X_ZOOM_SPAN / 2));
        }

        let sqrtLow = Math.sqrt(Math.max(Math.min(pointA.y, pointB.y), 0));
        let sqrtHigh = Math.sqrt(Math.max(pointA.y, pointB.y, 0));
        if (sqrtHigh - sqrtLow < MIN_Y_ZOOM_SQRT_SPAN) {
            const centerSqrt = (sqrtLow + sqrtHigh) / 2;
            sqrtLow = Math.max(Math.sqrt(Math.max(yDomain[0], 0)), centerSqrt - (MIN_Y_ZOOM_SQRT_SPAN / 2));
            sqrtHigh = Math.min(Math.sqrt(Math.max(yDomain[1], 0)), centerSqrt + (MIN_Y_ZOOM_SQRT_SPAN / 2));
        }

        setZoomDomains({ x: [x0, x1], y: [sqrtLow * sqrtLow, sqrtHigh * sqrtHigh] });
    };

    const handleChartWheel = (event: React.WheelEvent<HTMLDivElement>) => {
        const point = getRelativePoint(event);
        const isInsidePlotArea = plotArea
            && point.x >= plotArea.x && point.x <= plotArea.x + plotArea.width
            && point.y >= plotArea.y && point.y <= plotArea.y + plotArea.height;
        const anchor = plotArea && isInsidePlotArea
            ? pixelToDomainPoint(point.x, point.y, plotArea, effectiveXDomain, effectiveYDomain)
            : undefined;
        applyZoom(event.deltaY < 0 ? WHEEL_ZOOM_FACTOR : 1 / WHEEL_ZOOM_FACTOR, anchor);
    };

    const statsSummary = (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-xs text-base-content/70">
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
    );

    const legend = (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-xs text-base-content/70">
            <button
                type="button"
                className={`flex items-center gap-1 rounded px-1 py-0.5 transition-colors hover:bg-base-200 focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary ${
                    foreignFlowFilter === 'buy' ? 'bg-base-200 font-semibold text-base-content' : ''
                }`}
                onClick={() => toggleForeignFlowFilter('buy')}
                aria-pressed={foreignFlowFilter === 'buy'}
                title="Show only foreign net buy stocks"
            >
                <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: POSITIVE_FLOW_COLOR }}></span>
                Foreign net buy ({formatCount(foreignFlowCounts.buy)})
            </button>
            <button
                type="button"
                className={`flex items-center gap-1 rounded px-1 py-0.5 transition-colors hover:bg-base-200 focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary ${
                    foreignFlowFilter === 'sell' ? 'bg-base-200 font-semibold text-base-content' : ''
                }`}
                onClick={() => toggleForeignFlowFilter('sell')}
                aria-pressed={foreignFlowFilter === 'sell'}
                title="Show only foreign net sell stocks"
            >
                <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: NEGATIVE_FLOW_COLOR }}></span>
                Foreign net sell ({formatCount(foreignFlowCounts.sell)})
            </button>
            <button
                type="button"
                className={`flex items-center gap-1 rounded px-1 py-0.5 transition-colors hover:bg-base-200 focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary ${
                    foreignFlowFilter === 'neutral' ? 'bg-base-200 font-semibold text-base-content' : ''
                }`}
                onClick={() => toggleForeignFlowFilter('neutral')}
                aria-pressed={foreignFlowFilter === 'neutral'}
                title="Show only neutral or unavailable foreign flow stocks"
            >
                <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: NEUTRAL_FLOW_COLOR }}></span>
                Neutral / N/A ({formatCount(foreignFlowCounts.neutral)})
            </button>
            <span className="flex items-center gap-1">
                <span className="h-3 w-3 rounded-full border" style={{ borderColor: TREND_UP_COLOR }}></span>
                3-day up ring
            </span>
            <span className="flex items-center gap-1">
                <span className="h-3 w-3 rounded-full border" style={{ borderColor: TREND_DOWN_COLOR }}></span>
                3-day down ring
            </span>
        </div>
    );

    const chartElement = (
        <ResponsiveContainer width="100%" height="100%" debounce={50}>
            <ScatterChart margin={CHART_MARGIN}>
                <PlotAreaCapture onPlotAreaChange={handlePlotAreaChange} />
                <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.12} />
                <XAxis
                    type="number"
                    dataKey="x"
                    domain={effectiveXDomain}
                    allowDataOverflow
                    ticks={xTicks}
                    tick={{ fontSize: 11 }}
                    stroke="currentColor"
                    opacity={0.55}
                    tickFormatter={(value: number) => `${Number(value.toFixed(1))}%`}
                    label={{
                        value: 'Daily % Change',
                        position: 'bottom',
                        style: { fontSize: 12, fontWeight: 600 },
                    }}
                />
                <YAxis
                    type="number"
                    dataKey="y"
                    domain={effectiveYDomain}
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
                    data={displayedChartData}
                    shape={<BubblePoint />}
                    isAnimationActive={false}
                    onMouseLeave={() => setActivePoint(null)}
                    onMouseMove={(event: ScatterPoint) => setActivePoint(event)}
                >
                </Scatter>

            </ScatterChart>
        </ResponsiveContainer>
    );

    return (
        <section className="w-full rounded-lg border border-base-300 bg-base-100">
            <div className="grid gap-3 px-3 py-2 lg:grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] lg:items-center">
                <div className="flex items-center gap-2">
                    <h3 className="text-base font-semibold">Price Change vs Traded Value</h3>
                    <button
                        type="button"
                        className="btn btn-ghost btn-xs btn-square"
                        onClick={() => setIsFullscreen(true)}
                        title="Open fullscreen"
                        aria-label="Open fullscreen"
                    >
                        <ExpandIcon />
                    </button>
                </div>
                <div className="flex justify-center">{statsSummary}</div>
                <div className="flex justify-end">{legend}</div>
            </div>

            <div className="aspect-[3/2] min-w-0">
                {isFullscreen ? (
                    <div className="flex h-full items-center justify-center text-sm text-base-content/50">
                        Chart is open in fullscreen mode.
                    </div>
                ) : (
                    chartElement
                )}
            </div>

            {isFullscreen && typeof document !== 'undefined'
                ? createPortal(
                    <div className="fixed inset-0 z-[150] flex flex-col bg-base-100">
                        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-base-300 px-4 py-2">
                            <h3 className="text-base font-semibold">Price Change vs Traded Value</h3>
                            {statsSummary}
                            <div className="ml-auto flex flex-wrap items-center gap-x-4 gap-y-2">
                                {legend}
                                <div className="flex items-center gap-1">
                                    <button
                                        type="button"
                                        className="btn btn-ghost btn-xs btn-square"
                                        onClick={() => applyZoom(BUTTON_ZOOM_FACTOR)}
                                        title="Zoom in"
                                        aria-label="Zoom in"
                                    >
                                        <ZoomInIcon />
                                    </button>
                                    <button
                                        type="button"
                                        className="btn btn-ghost btn-xs btn-square"
                                        onClick={() => applyZoom(1 / BUTTON_ZOOM_FACTOR)}
                                        disabled={!zoomDomains}
                                        title="Zoom out"
                                        aria-label="Zoom out"
                                    >
                                        <ZoomOutIcon />
                                    </button>
                                    <button
                                        type="button"
                                        className="btn btn-ghost btn-xs"
                                        onClick={() => setZoomDomains(null)}
                                        disabled={!zoomDomains}
                                    >
                                        Reset
                                    </button>
                                    <button
                                        type="button"
                                        className="btn btn-ghost btn-xs btn-square"
                                        onClick={closeFullscreen}
                                        title="Exit fullscreen"
                                        aria-label="Exit fullscreen"
                                    >
                                        <CloseIcon />
                                    </button>
                                </div>
                            </div>
                        </div>
                        <p className="px-4 pt-1 text-[11px] text-base-content/50">
                            Drag to zoom into an area · Scroll to zoom at the cursor · Arrow keys to pan · Esc to exit
                        </p>
                        <div
                            className="relative min-h-0 flex-1 cursor-crosshair select-none"
                            onMouseDown={handleChartMouseDown}
                            onMouseMove={handleChartMouseMove}
                            onMouseUp={handleChartMouseUp}
                            onMouseLeave={() => setSelection(null)}
                            onWheel={handleChartWheel}
                        >
                            {chartElement}
                            {selection ? (
                                <div
                                    className="pointer-events-none absolute rounded border border-primary/60 bg-primary/10"
                                    style={{
                                        left: Math.min(selection.startX, selection.currentX),
                                        top: Math.min(selection.startY, selection.currentY),
                                        width: Math.abs(selection.currentX - selection.startX),
                                        height: Math.abs(selection.currentY - selection.startY),
                                    }}
                                />
                            ) : null}
                        </div>
                    </div>,
                    document.body,
                )
                : null}
        </section>
    );
};

export default PriceChangeTradedValueScatter;
