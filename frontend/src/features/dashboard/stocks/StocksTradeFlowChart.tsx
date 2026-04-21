import React, { useEffect, useMemo, useRef, useState } from 'react';
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
import { stockApi } from '../../../api/stockApi';
import type { Stock, StocksVolumeSeriesResponse } from '../../../api/stockApi';
import type { DateRange } from './dateRange';

interface StocksTradeFlowChartProps {
    stocks: Stock[];
    dateRange: DateRange;
}

interface AggregatedPoint {
    x: number;
    date: string;
    foreignNetTotal: number | null;
    propNetTotal: number | null;
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

const FOREIGN_COLOR = '#2563eb';
const FOREIGN_POSITIVE_COLOR = '#60a5fa';
const PROPRIETARY_COLOR = '#d97706';
const PROPRIETARY_POSITIVE_COLOR = '#fbbf24';
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

const formatRangeNetValue = (value: number | null): string => {
    if (value === null) {
        return 'N/A';
    }
    return `${value >= 0 ? '+' : '-'}${formatBilVnd(Math.abs(value))} Bil`;
};

const computeDomain = (points: AggregatedPoint[]): [number, number] => {
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

export const StocksTradeFlowChart: React.FC<StocksTradeFlowChartProps> = ({ stocks, dateRange }) => {
    const requestIdRef = useRef(0);
    const [flowResponse, setFlowResponse] = useState<StocksVolumeSeriesResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const symbols = useMemo(() => {
        return Array.from(new Set(stocks.map((stock) => stock.ticker.toUpperCase())));
    }, [stocks]);

    useEffect(() => {
        if (symbols.length === 0) {
            requestIdRef.current += 1;
            return;
        }

        const requestId = requestIdRef.current + 1;
        requestIdRef.current = requestId;
        window.setTimeout(() => {
            if (requestId !== requestIdRef.current) {
                return;
            }
            setLoading(true);
            setError(null);
        }, 0);

        stockApi.getStocksVolumeSeries(symbols, dateRange.startDate, dateRange.endDate)
            .then((response) => {
                if (requestId !== requestIdRef.current) {
                    return;
                }
                setFlowResponse(response);
            })
            .catch((err) => {
                if (requestId !== requestIdRef.current) {
                    return;
                }
                console.error('Error fetching stocks trade flow series:', err);
                setError('Failed to load trade flow series. Please try again.');
            })
            .finally(() => {
                if (requestId !== requestIdRef.current) {
                    return;
                }
                setLoading(false);
            });
    }, [dateRange.endDate, dateRange.startDate, symbols]);

    const aggregatedPoints = useMemo<AggregatedPoint[]>(() => {
        if (!flowResponse) {
            return [];
        }

        const byDate = new Map<string, {
            x: number;
            foreignSum: number;
            foreignCount: number;
            propSum: number;
            propCount: number;
        }>();

        flowResponse.stocks.forEach((stockSeries) => {
            stockSeries.data.forEach((point) => {
                const x = new Date(`${point.date}T00:00:00`).getTime();
                if (Number.isNaN(x)) {
                    return;
                }
                const entry = byDate.get(point.date) ?? {
                    x,
                    foreignSum: 0,
                    foreignCount: 0,
                    propSum: 0,
                    propCount: 0,
                };

                if (point.foreign_net_value !== null) {
                    entry.foreignSum += point.foreign_net_value;
                    entry.foreignCount += 1;
                }
                if (point.prop_net_value !== null) {
                    entry.propSum += point.prop_net_value;
                    entry.propCount += 1;
                }

                byDate.set(point.date, entry);
            });
        });

        return Array.from(byDate.entries())
            .map(([date, entry]) => ({
                x: entry.x,
                date,
                foreignNetTotal: entry.foreignCount > 0 ? roundToTwo(entry.foreignSum) : null,
                propNetTotal: entry.propCount > 0 ? roundToTwo(entry.propSum) : null,
            }))
            .sort((a, b) => a.x - b.x);
    }, [flowResponse]);

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
    const foreignRangeNetTotal = useMemo(() => {
        let total = 0;
        let hasValue = false;
        aggregatedPoints.forEach((point) => {
            if (point.foreignNetTotal !== null) {
                total += point.foreignNetTotal;
                hasValue = true;
            }
        });
        return hasValue ? roundToTwo(total) : null;
    }, [aggregatedPoints]);
    const propRangeNetTotal = useMemo(() => {
        let total = 0;
        let hasValue = false;
        aggregatedPoints.forEach((point) => {
            if (point.propNetTotal !== null) {
                total += point.propNetTotal;
                hasValue = true;
            }
        });
        return hasValue ? roundToTwo(total) : null;
    }, [aggregatedPoints]);

    const hasForeignFlowData = useMemo(
        () => aggregatedPoints.some((point) => point.foreignNetTotal !== null),
        [aggregatedPoints]
    );
    const hasPropFlowData = useMemo(
        () => aggregatedPoints.some((point) => point.propNetTotal !== null),
        [aggregatedPoints]
    );

    const flowAvailabilityNote = useMemo(() => {
        if (!hasForeignFlowData && !hasPropFlowData) {
            return 'Foreign and proprietary flow unavailable for this range.';
        }
        if (!hasForeignFlowData) {
            return 'Foreign flow unavailable for this range.';
        }
        if (!hasPropFlowData) {
            return 'Proprietary flow unavailable for this range.';
        }
        return null;
    }, [hasForeignFlowData, hasPropFlowData]);

    if (symbols.length === 0) {
        return (
            <div className="flex h-64 items-center justify-center text-base-content/50">
                No stocks available for trade flow analysis.
            </div>
        );
    }

    return (
        <div className="flex h-full w-full flex-col space-y-4">
            <div className="flex flex-col gap-2 border-b border-base-300 pb-3">
                <div className="flex flex-wrap items-center gap-3">
                    <div className="text-sm font-medium text-base-content">Aggregated trade flow</div>
                    <div className="text-xs text-base-content/60">
                        Range: {dateRange.startDate} to {dateRange.endDate} | {symbols.length} stocks
                    </div>
                </div>
                <div className="flex flex-wrap items-center gap-4 text-xs text-base-content/70">
                    <span className="inline-flex items-center gap-2">
                        <span className="h-0.5 w-5" style={{ backgroundColor: FOREIGN_COLOR }} />
                        Foreign
                    </span>
                    <span className="inline-flex items-center gap-2">
                        <span className="h-0.5 w-5" style={{ backgroundColor: PROPRIETARY_COLOR }} />
                        Proprietary
                    </span>
                    <span className="inline-flex items-center gap-2">
                        <span className="h-0.5 w-5" style={{ backgroundColor: FOREIGN_POSITIVE_COLOR }} />
                        Positive = brighter
                    </span>
                    <span className="inline-flex items-center gap-2">
                        <span className="h-0.5 w-5" style={{ backgroundColor: FOREIGN_COLOR }} />
                        Negative = base shade
                    </span>
                    <span className="text-xs text-base-content/70">
                        Foreign net sum: <span className="font-medium" style={{ color: FOREIGN_COLOR }}>{formatRangeNetValue(foreignRangeNetTotal)}</span>
                    </span>
                    <span className="text-xs text-base-content/70">
                        Proprietary net sum: <span className="font-medium" style={{ color: PROPRIETARY_COLOR }}>{formatRangeNetValue(propRangeNetTotal)}</span>
                    </span>
                </div>
            </div>

            {flowResponse?.is_stale && !flowResponse.is_syncing ? (
                <div className="alert alert-info py-2">
                    <span className="text-xs">Showing cached data. Fresh data sync is pending.</span>
                </div>
            ) : null}

            {flowResponse?.is_syncing ? (
                <div className="flex items-center gap-2 text-xs text-warning">
                    <span className="loading loading-spinner loading-xs" />
                    Syncing latest trade-flow data...
                </div>
            ) : null}

            {flowAvailabilityNote ? (
                <div className="text-xs text-base-content/60">{flowAvailabilityNote}</div>
            ) : null}

            {loading ? (
                <div className="flex h-96 flex-col items-center justify-center gap-4">
                    <span className="loading loading-spinner loading-lg text-primary"></span>
                    <p className="text-base-content/70">Loading trade flow series...</p>
                </div>
            ) : error ? (
                <div className="flex h-96 flex-col items-center justify-center gap-4">
                    <div className="alert alert-error max-w-md">
                        <span>{error}</span>
                    </div>
                </div>
            ) : !hasForeignFlowData && !hasPropFlowData ? (
                <div className="flex h-96 items-center justify-center text-base-content/50">
                    No trade flow data available for the selected range.
                </div>
            ) : (
                <div className="min-h-0 flex-1">
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
                                stroke={FOREIGN_COLOR}
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
                                stroke={FOREIGN_POSITIVE_COLOR}
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
                                stroke={PROPRIETARY_COLOR}
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
                                stroke={PROPRIETARY_POSITIVE_COLOR}
                                strokeWidth={LINE_WIDTH}
                                dot={false}
                                connectNulls={false}
                                isAnimationActive={false}
                                strokeLinecap="round"
                                strokeLinejoin="round"
                            />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            )}
        </div>
    );
};

export default StocksTradeFlowChart;
