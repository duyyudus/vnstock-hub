import React, { useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import type { FundPerformanceMetrics } from '../../api/stockApi';

interface CumulativeGrowthChartProps {
    funds: FundPerformanceMetrics[];
    benchmark: FundPerformanceMetrics | null;
    startYear: number;
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

export const CumulativeGrowthChart: React.FC<CumulativeGrowthChartProps> = ({
    funds,
    benchmark,
    startYear,
}) => {
    // Process data for chart - merge all NAV histories by date
    const chartData = useMemo(() => {
        if (!funds.length) return [];

        // Calculate start date based on startYear (Jan 1st)
        const startStr = `${startYear}-01-01`;

        // Get all unique dates
        const dateMap = new Map<string, Record<string, number>>();

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
                        dateMap.set(point.date, { date: point.date as unknown as number });
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
                        dateMap.set(point.date, { date: point.date as unknown as number });
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

    // Sort funds by final performance (last data point) to determine colors
    const sortedFunds = useMemo(() => {
        if (chartData.length === 0) return [...funds];

        const lastRecord = chartData[chartData.length - 1];
        return [...funds].sort((a, b) => {
            const aVal = lastRecord[a.symbol] ?? 0;
            const bVal = lastRecord[b.symbol] ?? 0;
            return bVal - aVal;
        });
    }, [funds, chartData]);

    const formatDate = (dateStr: string) => {
        const date = new Date(dateStr);
        return `${date.getMonth() + 1}/${date.getFullYear().toString().slice(2)}`;
    };

    const formatValue = (value: number) => {
        return value.toFixed(0);
    };

    const CustomTooltip = ({ active, payload, label }: any) => {
        if (active && payload && payload.length) {
            // Sort payload by value descending
            const sorted = [...payload].sort((a, b) => (b.value || 0) - (a.value || 0));
            return (
                <div className="bg-base-100 border border-base-300 p-3 rounded-lg shadow-lg max-h-64 overflow-y-auto">
                    <p className="text-sm font-semibold mb-2">{label}</p>
                    {sorted.slice(0, 10).map((entry: any, index: number) => (
                        <p key={index} className="text-xs" style={{ color: entry.color }}>
                            {entry.dataKey === 'benchmark' ? '📊 ' : ''}{entry.dataKey}: {entry.value?.toFixed(1) || 'N/A'}
                        </p>
                    ))}
                    {sorted.length > 10 && (
                        <p className="text-xs text-base-content/50">...and {sorted.length - 10} more</p>
                    )}
                </div>
            );
        }
        return null;
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
            <div className="flex-1 min-h-0 relative">
                <ResponsiveContainer width="100%" height={680} debounce={50}>
                    <LineChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 20 }}>
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
                        <Tooltip content={<CustomTooltip />} isAnimationActive={false} />

                        {/* Render funds - top 10 get colors, rest are gray */}
                        {sortedFunds.map((fund, idx) => (
                            <Line
                                key={fund.symbol}
                                type="monotone"
                                dataKey={fund.symbol}
                                stroke={idx < 10 ? TOP_COLORS[idx] : GRAY_COLOR}
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
            <div className="flex flex-wrap justify-center gap-3 mt-4 mb-2 text-xs">
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
    );
};

export default CumulativeGrowthChart;
