import React, { useEffect, useMemo, useRef, useState } from 'react';
import { stockApi } from '../../../../api/stockApi';
import type { Stock, StocksVolumeSeriesResponse } from '../../../../api/stockApi';
import type { DateRange } from '../dateRange';
import { AggregatedTradeFlowChart } from './AggregatedTradeFlowChart';

interface TradeFlowProps {
    stocks: Stock[];
    dateRange: DateRange;
}

export interface TradeFlowAggregatedPoint {
    x: number;
    date: string;
    foreignNetTotal: number | null;
    propNetTotal: number | null;
}

export interface TradeFlowChartColors {
    foreign: string;
    foreignPositive: string;
    proprietary: string;
    proprietaryPositive: string;
}

const TRADE_FLOW_COLORS: TradeFlowChartColors = {
    foreign: '#2563eb',
    foreignPositive: '#60a5fa',
    proprietary: '#d97706',
    proprietaryPositive: '#fbbf24',
};

const roundToTwo = (value: number): number => Math.round(value * 100) / 100;

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

export const TradeFlow: React.FC<TradeFlowProps> = ({ stocks, dateRange }) => {
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

    const aggregatedPoints = useMemo<TradeFlowAggregatedPoint[]>(() => {
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

    const foreignRangeNetTotal = useMemo(
        () => getRangeNetTotal(aggregatedPoints, 'foreignNetTotal'),
        [aggregatedPoints]
    );
    const propRangeNetTotal = useMemo(
        () => getRangeNetTotal(aggregatedPoints, 'propNetTotal'),
        [aggregatedPoints]
    );

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
                        <span className="h-0.5 w-5" style={{ backgroundColor: TRADE_FLOW_COLORS.foreign }} />
                        Foreign
                    </span>
                    <span className="inline-flex items-center gap-2">
                        <span className="h-0.5 w-5" style={{ backgroundColor: TRADE_FLOW_COLORS.proprietary }} />
                        Proprietary
                    </span>
                    <span className="inline-flex items-center gap-2">
                        <span className="h-0.5 w-5" style={{ backgroundColor: TRADE_FLOW_COLORS.foreignPositive }} />
                        Positive = brighter
                    </span>
                    <span className="inline-flex items-center gap-2">
                        <span className="h-0.5 w-5" style={{ backgroundColor: TRADE_FLOW_COLORS.foreign }} />
                        Negative = base shade
                    </span>
                    <span className="text-xs text-base-content/70">
                        Foreign net sum: <span className="font-medium" style={{ color: TRADE_FLOW_COLORS.foreign }}>{formatRangeNetValue(foreignRangeNetTotal)}</span>
                    </span>
                    <span className="text-xs text-base-content/70">
                        Proprietary net sum: <span className="font-medium" style={{ color: TRADE_FLOW_COLORS.proprietary }}>{formatRangeNetValue(propRangeNetTotal)}</span>
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
                <AggregatedTradeFlowChart
                    aggregatedPoints={aggregatedPoints}
                    colors={TRADE_FLOW_COLORS}
                />
            )}
        </div>
    );
};

export default TradeFlow;
