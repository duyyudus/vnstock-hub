import React, { useEffect, useMemo, useRef, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import type { FundPerformanceMetrics } from '../../../api/stockApi';

interface CumulativeGrowthChartProps {
    funds: FundPerformanceMetrics[];
    benchmark: FundPerformanceMetrics | null;
    startYear: number;
}

interface GrowthChartRecord extends Record<string, number | string> {
    date: string;
}

interface GrowthTooltipEntry {
    color?: string;
    dataKey?: string | number;
    name?: string | number;
    value?: number | string | null;
}

interface GrowthTooltipProps {
    active?: boolean;
    payload?: GrowthTooltipEntry[];
    label?: string | number;
}

interface GrowthTooltipCardProps {
    label?: string | number;
    payload?: GrowthTooltipEntry[];
    maxEntries?: number;
    onClose?: () => void;
    footerText?: string;
    scrollable?: boolean;
}

interface ChartInteractionState {
    activeTooltipIndex?: number | string;
    activeLabel?: string | number;
    isTooltipActive?: boolean;
}

// Color palette for top funds
const TOP_COLORS = [
    '#3b82f6', // blue
    '#10b981', // green
    '#f59e0b', // amber
    '#ef4444', // red
    '#8b5cf6', // purple
    '#ec4899', // pink
    '#06b6d4', // cyan
    '#84cc16', // lime
    '#f97316', // orange
    '#6366f1', // indigo
];

const GRAY_COLOR = '#6b7280';
const BENCHMARK_COLOR = '#fbbf24';

const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return `${date.getMonth() + 1}/${date.getFullYear().toString().slice(2)}`;
};

const formatValue = (value: number) => {
    return value.toFixed(0);
};

const getFundColor = (index: number) => {
    return index < 10 ? TOP_COLORS[index] : GRAY_COLOR;
};

const getLatestGrowthValue = (fund: FundPerformanceMetrics, startStr: string) => {
    const history = [...fund.nav_history]
        .filter((point) => point.date >= startStr)
        .sort((a, b) => a.date.localeCompare(b.date));

    if (history.length === 0) {
        return null;
    }

    const baseVal = history[0].normalized_nav;
    const latestVal = history[history.length - 1].normalized_nav;
    return baseVal > 0 ? (latestVal / baseVal) * 100 : 100;
};

const sortGrowthPayload = (payload?: GrowthTooltipEntry[]) => {
    return [...(payload ?? [])].sort((a, b) => {
        const aValue = typeof a.value === 'number' ? a.value : 0;
        const bValue = typeof b.value === 'number' ? b.value : 0;
        return bValue - aValue;
    });
};

const GrowthTooltipCard: React.FC<GrowthTooltipCardProps> = ({
    payload,
    label,
    maxEntries,
    onClose,
    footerText,
    scrollable = false,
}) => {
    const sorted = sortGrowthPayload(payload);

    if (sorted.length === 0) {
        return null;
    }

    const visibleEntries = maxEntries != null ? sorted.slice(0, maxEntries) : sorted;
    const hiddenCount = sorted.length - visibleEntries.length;

    return (
        <div
            className={`bg-base-100 border border-base-300 p-3 rounded-lg shadow-lg w-[26rem] max-w-[min(26rem,calc(100vw-2rem))] ${scrollable ? 'max-h-[32rem] overflow-y-auto' : ''}`}
        >
            <div className="flex items-start justify-between gap-3 mb-2">
                <div>
                    <p className="text-sm font-semibold">{label}</p>
                    <p className="text-[11px] text-base-content/50 mt-1">
                        Colors are fixed by latest available performance. Entries below are sorted for this date.
                    </p>
                </div>
                {onClose ? (
                    <button type="button" className="btn btn-ghost btn-xs" onClick={onClose}>
                        Close
                    </button>
                ) : null}
            </div>
            {visibleEntries.map((entry, index) => {
                const dataKey = entry.dataKey != null ? String(entry.dataKey) : 'N/A';
                const displayName = entry.name != null ? String(entry.name) : '';
                const isBenchmark = dataKey === 'benchmark';
                const primaryLabel = isBenchmark ? (displayName || dataKey) : dataKey;
                const secondaryLabel =
                    !isBenchmark && displayName && displayName !== dataKey ? displayName : null;
                const value = typeof entry.value === 'number' ? entry.value.toFixed(1) : 'N/A';
                return (
                    <div key={`${dataKey}-${index}`} className="mb-1 last:mb-0">
                        <p className="text-xs" style={{ color: entry.color }}>
                            {isBenchmark ? '📊 ' : ''}{primaryLabel}: {value}
                        </p>
                        {secondaryLabel ? (
                            <p className="text-[11px] text-base-content/55 leading-tight pl-2">
                                {secondaryLabel}
                            </p>
                        ) : null}
                    </div>
                );
            })}
            {hiddenCount > 0 ? (
                <p className="text-[11px] text-base-content/50 mt-2">
                    +{hiddenCount} more. Click the chart to pin the full list.
                </p>
            ) : null}
            {footerText ? (
                <p className="text-[11px] text-base-content/50 mt-2">
                    {footerText}
                </p>
            ) : null}
        </div>
    );
};

const GrowthTooltip: React.FC<GrowthTooltipProps> = ({
    active,
    payload,
    label,
}) => {
    if (!active || !payload?.length) {
        return null;
    }

    return (
        <GrowthTooltipCard
            label={label}
            payload={payload}
            maxEntries={10}
            footerText="Click the chart to pin this list."
        />
    );
};

export const CumulativeGrowthChart: React.FC<CumulativeGrowthChartProps> = ({
    funds,
    benchmark,
    startYear,
}) => {
    const chartContainerRef = useRef<HTMLDivElement | null>(null);
    const [pinnedDate, setPinnedDate] = useState<string | null>(null);

    // Process data for chart - merge all NAV histories by date
    const chartData = useMemo(() => {
        if (!funds.length) return [];

        // Calculate start date based on startYear (Jan 1st)
        const startStr = `${startYear}-01-01`;

        // Get all unique dates
        const dateMap = new Map<string, GrowthChartRecord>();

        // Add fund data
        funds.forEach((fund) => {
            // Filter and sort history for the period
            const history = [...fund.nav_history]
                .filter(p => p.date >= startStr)
                .sort((a, b) => a.date.localeCompare(b.date));

            if (history.length > 0) {
                // Re-normalize to 100 at the start of the timeframe
                const baseVal = history[0].normalized_nav;
                history.forEach((point) => {
                    if (!dateMap.has(point.date)) {
                        dateMap.set(point.date, { date: point.date });
                    }
                    const record = dateMap.get(point.date)!;
                    // Protect against division by zero just in case
                    record[fund.symbol] = baseVal > 0 ? (point.normalized_nav / baseVal) * 100 : 100;
                });
            }
        });

        // Add benchmark data
        if (benchmark) {
            const history = [...benchmark.nav_history]
                .filter(p => p.date >= startStr)
                .sort((a, b) => a.date.localeCompare(b.date));

            if (history.length > 0) {
                const baseVal = history[0].normalized_nav;
                history.forEach((point) => {
                    if (!dateMap.has(point.date)) {
                        dateMap.set(point.date, { date: point.date });
                    }
                    const record = dateMap.get(point.date)!;
                    record['benchmark'] = baseVal > 0 ? (point.normalized_nav / baseVal) * 100 : 100;
                });
            }
        }

        // Sort by date and convert to array
        return Array.from(dateMap.values()).sort((a, b) =>
            String(a.date).localeCompare(String(b.date))
        );
    }, [funds, benchmark, startYear]);

    // Rank funds by each fund's own latest available value in the selected range
    const sortedFunds = useMemo(() => {
        const startStr = `${startYear}-01-01`;
        return [...funds].sort((a, b) => {
            const aVal = getLatestGrowthValue(a, startStr) ?? 0;
            const bVal = getLatestGrowthValue(b, startStr) ?? 0;
            return bVal - aVal;
        });
    }, [funds, startYear]);

    const buildTooltipPayload = (record: GrowthChartRecord): GrowthTooltipEntry[] => {
        const nextPayload: GrowthTooltipEntry[] = [];

        sortedFunds.forEach((fund, index) => {
            const value = record[fund.symbol];
            if (typeof value === 'number') {
                nextPayload.push({
                    color: getFundColor(index),
                    dataKey: fund.symbol,
                    name: fund.name,
                    value,
                });
            }
        });

        if (benchmark) {
            const benchmarkValue = record.benchmark;
            if (typeof benchmarkValue === 'number') {
                nextPayload.push({
                    color: BENCHMARK_COLOR,
                    dataKey: 'benchmark',
                    name: benchmark.name,
                    value: benchmarkValue,
                });
            }
        }

        return nextPayload;
    };

    const pinnedTooltip = (() => {
        if (!pinnedDate) {
            return null;
        }

        const record = chartData.find((entry) => String(entry.date) === pinnedDate);
        if (!record) {
            return null;
        }

        const payload = buildTooltipPayload(record);
        if (payload.length === 0) {
            return null;
        }

        return {
            label: pinnedDate,
            payload,
        };
    })();

    useEffect(() => {
        if (!pinnedDate) {
            return;
        }

        const handlePointerDown = (event: MouseEvent) => {
            const container = chartContainerRef.current;
            const target = event.target;
            if (!(target instanceof Node) || !container) {
                return;
            }

            if (!container.contains(target)) {
                setPinnedDate(null);
            }
        };

        document.addEventListener('mousedown', handlePointerDown);

        return () => {
            document.removeEventListener('mousedown', handlePointerDown);
        };
    }, [pinnedDate]);

    const handleChartClick = (state: ChartInteractionState) => {
        if (!state.isTooltipActive || state.activeTooltipIndex == null) {
            setPinnedDate(null);
            return;
        }

        const recordIndex = Number(state.activeTooltipIndex);
        if (!Number.isInteger(recordIndex) || recordIndex < 0 || recordIndex >= chartData.length) {
            setPinnedDate(null);
            return;
        }

        const record = chartData[recordIndex];
        const payload = buildTooltipPayload(record);
        if (payload.length === 0) {
            setPinnedDate(null);
            return;
        }

        setPinnedDate(String(state.activeLabel ?? record.date));
    };

    if (chartData.length === 0) {
        return (
            <div className="flex items-center justify-center h-full text-base-content/50">
                No data available for the selected timeframe
            </div>
        );
    }

    return (
        <div className="w-full h-full flex flex-col">
            <div ref={chartContainerRef} className="flex-1 min-h-0 relative">
                {pinnedTooltip ? (
                    <div className="absolute right-4 top-4 z-30">
                        <GrowthTooltipCard
                            label={pinnedTooltip.label}
                            payload={pinnedTooltip.payload}
                            onClose={() => setPinnedDate(null)}
                            scrollable={true}
                        />
                    </div>
                ) : null}
                <ResponsiveContainer width="100%" height={680} debounce={50}>
                    <LineChart
                        data={chartData}
                        margin={{ top: 10, right: 30, left: 10, bottom: 30 }}
                        onClick={handleChartClick}
                    >
                        <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} />
                        <XAxis
                            dataKey="date"
                            tickFormatter={formatDate}
                            tick={{ fontSize: 11 }}
                            stroke="currentColor"
                            opacity={0.5}
                            interval="preserveStartEnd"
                        />
                        <YAxis
                            tickFormatter={formatValue}
                            tick={{ fontSize: 11 }}
                            stroke="currentColor"
                            opacity={0.5}
                            domain={['auto', 'auto']}
                            label={{ value: 'Growth (Base=100)', angle: -90, position: 'insideLeft', style: { fontSize: 11 } }}
                        />
                        <Tooltip
                            content={<GrowthTooltip />}
                            isAnimationActive={false}
                            allowEscapeViewBox={{ x: true, y: true }}
                            offset={16}
                            wrapperStyle={{ zIndex: 20, pointerEvents: 'auto' }}
                        />

                        {/* Render funds - top 10 get colors, rest are gray */}
                        {sortedFunds.map((fund, idx) => (
                            <Line
                                key={fund.symbol}
                                type="monotone"
                                dataKey={fund.symbol}
                                stroke={getFundColor(idx)}
                                strokeWidth={idx < 3 ? 2.5 : idx < 10 ? 1.5 : 0.8}
                                strokeOpacity={idx < 10 ? 1 : 0.3}
                                dot={false}
                                name={fund.name}
                                connectNulls={true}
                            />
                        ))}

                        {/* Benchmark line - dashed, prominent */}
                        {benchmark && (
                            <Line
                                type="monotone"
                                dataKey="benchmark"
                                stroke={BENCHMARK_COLOR}
                                strokeWidth={2.5}
                                strokeDasharray="5 5"
                                dot={false}
                                name={benchmark.name}
                                connectNulls={true}
                            />
                        )}
                    </LineChart>
                </ResponsiveContainer>
            </div>

            {/* Legend for top funds */}
            <div className="flex flex-col items-center gap-2 mt-4 mb-2 text-xs">
                <p className="text-[11px] text-base-content/55 text-center">
                    Highlight colors rank funds by their latest available value in the selected period.
                </p>
                <div className="flex flex-wrap justify-center gap-3">
                    {sortedFunds.slice(0, 5).map((fund, idx) => (
                        <div key={fund.symbol} className="flex items-center gap-1">
                            <div className="w-3 h-3 rounded" style={{ backgroundColor: TOP_COLORS[idx] }}></div>
                            <span>{fund.symbol}</span>
                        </div>
                    ))}
                    {benchmark && (
                        <div className="flex items-center gap-1">
                            <div className="w-3 h-0.5" style={{ backgroundColor: BENCHMARK_COLOR, borderStyle: 'dashed' }}></div>
                            <span>{benchmark.name}</span>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default CumulativeGrowthChart;
