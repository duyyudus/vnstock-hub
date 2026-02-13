import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LabelList } from 'recharts';

interface StockAllocationItem {
    ticker: string;
    allocation: number;
    companyName?: string;
}

interface StockAllocationChartProps {
    data: StockAllocationItem[];
    loading?: boolean;
}

interface StockAllocationTooltipEntry {
    payload: StockAllocationItem;
}

interface StockAllocationTooltipProps {
    active?: boolean;
    payload?: StockAllocationTooltipEntry[];
}

const ROW_HEIGHT = 30;
const MIN_CHART_HEIGHT = 320;

const formatPercent = (value: number) => {
    return `${value.toFixed(1)}%`;
};

const formatLabelPercent = (value: number | string | boolean | undefined | null) => {
    const numericValue = typeof value === 'number' ? value : Number(value);
    if (!Number.isFinite(numericValue)) {
        return '';
    }
    return `${numericValue.toFixed(1)}%`;
};

const StockAllocationTooltip: React.FC<StockAllocationTooltipProps> = ({ active, payload }) => {
    if (active && payload && payload.length) {
        const data = payload[0].payload;
        return (
            <div className="bg-base-100 border border-base-300 p-3 rounded-lg shadow-lg">
                <p className="text-sm font-semibold mb-1">{data.ticker}</p>
                {data.companyName && (
                    <p className="text-xs text-base-content/70 mb-1">
                        {data.companyName}
                    </p>
                )}
                <p className="text-xs text-primary">
                    Allocation: {formatPercent(data.allocation)}
                </p>
            </div>
        );
    }
    return null;
};

export const StockAllocationChart: React.FC<StockAllocationChartProps> = ({ data, loading = false }) => {
    if (loading) {
        return (
            <div className="flex items-center justify-center h-full">
                <span className="loading loading-spinner loading-lg text-primary"></span>
            </div>
        );
    }

    if (!data || data.length === 0) {
        return (
            <div className="flex items-center justify-center h-full text-base-content/50">
                No stock data available
            </div>
        );
    }

    const chartHeight = Math.max(MIN_CHART_HEIGHT, data.length * ROW_HEIGHT);

    return (
        <div className="w-full h-full">
            <div style={{ height: `${chartHeight}px` }}>
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                        data={data}
                        layout="vertical"
                        margin={{ top: 10, right: 20, left: 4, bottom: 10 }}
                    >
                        <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} />
                        <XAxis
                            type="number"
                            tickFormatter={formatPercent}
                            tick={{ fontSize: 12 }}
                            stroke="currentColor"
                            opacity={0.5}
                        />
                        <YAxis
                            type="category"
                            dataKey="ticker"
                            tick={{ fontSize: 12 }}
                            stroke="currentColor"
                            opacity={0.5}
                            width={38}
                            tickMargin={4}
                        />
                        <Tooltip content={<StockAllocationTooltip />} isAnimationActive={false} />
                        <Bar
                            dataKey="allocation"
                            fill="#10b981"
                            radius={[0, 4, 4, 0]}
                        >
                            <LabelList
                                dataKey="allocation"
                                position="right"
                                formatter={formatLabelPercent}
                                fill="currentColor"
                                fontSize={12}
                            />
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};
