import React, { useMemo } from 'react';
import {
    CartesianGrid,
    Line,
    LineChart,
    ReferenceLine,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from 'recharts';
import type { DateRange } from '../utils/dateRange';
import type { TradeFlowAggregatedPoint, TradeFlowChartColors } from './TradeFlow';

interface AggregatedTradeFlowChartProps {
    aggregatedPoints: TradeFlowAggregatedPoint[];
    colors: TradeFlowChartColors;
    dateRange: DateRange;
    stockCount: number;
}

interface SplitPoint {
    x: number;
    date: string;
    foreignAll: number | null;
    foreignPositive: number | null;
    propAll: number | null;
    propPositive: number | null;
    foreignNetTotal: number | null;
    propNetTotal: number | null;
}

interface TooltipPayloadEntry {
    payload?: SplitPoint;
}

interface CustomTooltipProps {
    active?: boolean;
    payload?: TooltipPayloadEntry[];
}

const LINE_WIDTH = 2;

const roundToTwo = (value: number): number => Math.round(value * 100) / 100;

const formatDateShort = (rawValue: number): string => {
    const date = new Date(rawValue);
    if (Number.isNaN(date.getTime())) {
        return '';
    }
    return `${date.getMonth() + 1}/${date.getDate()}`;
};

const formatBilVnd = (value: number): string => {
    const absolute = Math.abs(value);
    if (absolute >= 1000) return `${value < 0 ? '-' : ''}${(absolute / 1000).toFixed(1)}K`;
    if (absolute >= 100) return `${value < 0 ? '-' : ''}${absolute.toFixed(0)}`;
    if (absolute >= 10) return `${value < 0 ? '-' : ''}${absolute.toFixed(1)}`;
    return `${value < 0 ? '-' : ''}${absolute.toFixed(2)}`;
};

const formatRangeNetValue = (value: number | null): string => {
    if (value === null) {
        return 'N/A';
    }
    return `${value >= 0 ? '+' : '-'}${formatBilVnd(Math.abs(value))} Bil`;
};

const getSignedValueClassName = (value: number | null): string => {
    if (value === null) {
        return 'font-medium text-base-content/70';
    }
    return value >= 0 ? 'font-medium text-success' : 'font-medium text-error';
};

const formatTooltipValue = (value: number | null): string => {
    if (value === null) {
        return 'N/A';
    }
    const formatted = new Intl.NumberFormat('en-US', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    }).format(Math.abs(value));
    return `${value >= 0 ? '+' : '-'}${formatted} Bil VND`;
};

const computeDomain = (points: TradeFlowAggregatedPoint[]): [number, number] => {
    const values = points.flatMap((point) => [
        point.foreignNetTotal,
        point.propNetTotal,
    ]).filter((value): value is number => value !== null);

    if (values.length === 0) {
        return [-1, 1];
    }

    const minValue = Math.min(...values, 0);
    const maxValue = Math.max(...values, 0);

    if (minValue === 0 && maxValue === 0) {
        return [-1, 1];
    }

    const amplitude = Math.max(Math.abs(minValue), Math.abs(maxValue));
    const padding = Math.max(amplitude * 0.08, 0.5);
    return [roundToTwo(minValue - padding), roundToTwo(maxValue + padding)];
};

const buildYAxisTicks = (domain: [number, number]): number[] => {
    const [minValue, maxValue] = domain;
    const tickSet = new Set<number>([roundToTwo(minValue), 0, roundToTwo(maxValue)]);

    const span = maxValue - minValue;
    if (span > 0) {
        const midpoint = roundToTwo(minValue + (span / 2));
        const lowerMid = roundToTwo((minValue + 0) / 2);
        const upperMid = roundToTwo((maxValue + 0) / 2);
        tickSet.add(midpoint);
        tickSet.add(lowerMid);
        tickSet.add(upperMid);
    }

    return Array.from(tickSet).sort((a, b) => a - b);
};

const getRangeNetTotal = (
    aggregatedPoints: TradeFlowAggregatedPoint[],
    key: 'foreignNetTotal' | 'propNetTotal'
): number | null => {
    let total = 0;
    let hasValue = false;

    aggregatedPoints.forEach((point) => {
        if (point[key] !== null) {
            total += point[key];
            hasValue = true;
        }
    });

    return hasValue ? roundToTwo(total) : null;
};

const CustomTooltip = ({ active, payload }: CustomTooltipProps) => {
    const point = payload?.[0]?.payload;
    if (!active || !point) {
        return null;
    }

    return (
        <div className="rounded-lg border border-base-300 bg-base-100 p-3 shadow-lg">
            <p className="mb-2 text-sm font-semibold">{point.date}</p>
            <p className="text-xs">
                Foreign net: <span className="font-medium">{formatTooltipValue(point.foreignNetTotal)}</span>
            </p>
            <p className="mt-1 text-xs">
                Proprietary net: <span className="font-medium">{formatTooltipValue(point.propNetTotal)}</span>
            </p>
        </div>
    );
};

export const AggregatedTradeFlowChart: React.FC<AggregatedTradeFlowChartProps> = ({
    aggregatedPoints,
    colors,
    dateRange,
    stockCount,
}) => {
    const chartData = useMemo<SplitPoint[]>(() => {
        return aggregatedPoints.map((point) => ({
            x: point.x,
            date: point.date,
            foreignAll: point.foreignNetTotal,
            foreignPositive: point.foreignNetTotal !== null && point.foreignNetTotal >= 0 ? point.foreignNetTotal : null,
            propAll: point.propNetTotal,
            propPositive: point.propNetTotal !== null && point.propNetTotal >= 0 ? point.propNetTotal : null,
            foreignNetTotal: point.foreignNetTotal,
            propNetTotal: point.propNetTotal,
        }));
    }, [aggregatedPoints]);
    const yDomain = useMemo(() => computeDomain(aggregatedPoints), [aggregatedPoints]);
    const yAxisTicks = useMemo(() => buildYAxisTicks(yDomain), [yDomain]);
    const foreignRangeNetTotal = useMemo(
        () => getRangeNetTotal(aggregatedPoints, 'foreignNetTotal'),
        [aggregatedPoints]
    );
    const propRangeNetTotal = useMemo(
        () => getRangeNetTotal(aggregatedPoints, 'propNetTotal'),
        [aggregatedPoints]
    );

    return (
        <section className="min-h-0 flex-1 space-y-4">
            <div className="flex flex-col gap-2 border-b border-base-300 pb-3">
                <div className="flex flex-wrap items-center gap-3">
                    <div className="text-sm font-medium text-base-content">Aggregated trade flow</div>
                    <div className="text-xs text-base-content/60">
                        Range: {dateRange.startDate} to {dateRange.endDate} | {stockCount} stocks
                    </div>
                </div>
                <div className="flex flex-wrap items-center gap-4 text-xs text-base-content/70">
                    <span className="inline-flex items-center gap-2">
                        <span className="h-0.5 w-5" style={{ backgroundColor: colors.foreign }} />
                        Foreign
                    </span>
                    <span className="inline-flex items-center gap-2">
                        <span className="h-0.5 w-5" style={{ backgroundColor: colors.proprietary }} />
                        Proprietary
                    </span>
                    <span className="inline-flex items-center gap-2">
                        <span className="h-0.5 w-5" style={{ backgroundColor: colors.foreignPositive }} />
                        Positive = brighter
                    </span>
                    <span className="inline-flex items-center gap-2">
                        <span className="h-0.5 w-5" style={{ backgroundColor: colors.foreign }} />
                        Negative = base shade
                    </span>
                    <span className="text-xs">
                        <span style={{ color: colors.foreign }}>Foreign net sum:</span>{' '}
                        <span className={getSignedValueClassName(foreignRangeNetTotal)}>
                            {formatRangeNetValue(foreignRangeNetTotal)}
                        </span>
                    </span>
                    <span className="text-xs">
                        <span style={{ color: colors.proprietary }}>Proprietary net sum:</span>{' '}
                        <span className={getSignedValueClassName(propRangeNetTotal)}>
                            {formatRangeNetValue(propRangeNetTotal)}
                        </span>
                    </span>
                </div>
            </div>
            <ResponsiveContainer width="100%" height={560} debounce={50}>
                <LineChart
                    data={chartData}
                    margin={{ top: 10, right: 30, left: 10, bottom: 30 }}
                >
                    <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} />
                    <XAxis
                        type="number"
                        dataKey="x"
                        domain={['dataMin', 'dataMax']}
                        tickFormatter={formatDateShort}
                        tick={{ fontSize: 11 }}
                        stroke="currentColor"
                        opacity={0.5}
                    />
                    <YAxis
                        domain={yDomain}
                        ticks={yAxisTicks}
                        tickFormatter={formatBilVnd}
                        tick={{ fontSize: 11 }}
                        stroke="currentColor"
                        opacity={0.5}
                        label={{ value: 'Net Flow (Bil VND)', angle: -90, position: 'insideLeft', style: { fontSize: 11 } }}
                    />
                    <Tooltip content={<CustomTooltip />} isAnimationActive={false} />
                    <ReferenceLine y={0} stroke="currentColor" strokeOpacity={0.3} />

                    <Line
                        type="linear"
                        dataKey="foreignAll"
                        stroke={colors.foreign}
                        strokeWidth={LINE_WIDTH}
                        dot={false}
                        connectNulls={false}
                        isAnimationActive={false}
                        strokeLinecap="round"
                        strokeLinejoin="round"
                    />
                    <Line
                        type="linear"
                        dataKey="foreignPositive"
                        stroke={colors.foreignPositive}
                        strokeWidth={LINE_WIDTH}
                        dot={false}
                        connectNulls={false}
                        isAnimationActive={false}
                        strokeLinecap="round"
                        strokeLinejoin="round"
                    />
                    <Line
                        type="linear"
                        dataKey="propAll"
                        stroke={colors.proprietary}
                        strokeWidth={LINE_WIDTH}
                        dot={false}
                        connectNulls={false}
                        isAnimationActive={false}
                        strokeLinecap="round"
                        strokeLinejoin="round"
                    />
                    <Line
                        type="linear"
                        dataKey="propPositive"
                        stroke={colors.proprietaryPositive}
                        strokeWidth={LINE_WIDTH}
                        dot={false}
                        connectNulls={false}
                        isAnimationActive={false}
                        strokeLinecap="round"
                        strokeLinejoin="round"
                    />
                </LineChart>
            </ResponsiveContainer>
        </section>
    );
};

export default AggregatedTradeFlowChart;
