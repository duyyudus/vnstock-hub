import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface TopHoldingChartProps {
    data: any[];
    loading?: boolean;
}

export const TopHoldingChart: React.FC<TopHoldingChartProps> = ({ data, loading = false }) => {
    const formatPercent = (value: number) => {
        return `${value.toFixed(1)}%`;
    };

    const CustomTooltip = ({ active, payload }: any) => {
        if (active && payload && payload.length) {
            const data = payload[0].payload;
            const ticker = data.ticker || data.symbol || data.stock_code || 'N/A';
            const allocation = data.allocation || data.weight || data.percentage || 0;
            return (
                <div className="bg-base-100 border border-base-300 p-3 rounded-lg shadow-lg">
                    <p className="text-sm font-semibold mb-1">{ticker}</p>
                    <p className="text-xs text-primary">
                        Allocation: {formatPercent(allocation)}
                    </p>
                </div>
            );
        }
        return null;
    };

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
                No holdings data available
            </div>
        );
    }

    // Sort by allocation and take top 10
    const sortedData = [...data]
        .sort((a, b) => {
            const aVal = a.allocation || a.weight || a.percentage || 0;
            const bVal = b.allocation || b.weight || b.percentage || 0;
            return bVal - aVal;
        })
        .slice(0, 10);

    return (
        <div className="w-full h-full">
            <ResponsiveContainer width="100%" height="100%">
                <BarChart
                    data={sortedData}
                    layout="vertical"
                    margin={{ top: 10, right: 30, left: 60, bottom: 0 }}
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
                        dataKey={(item) => item.ticker || item.symbol || item.stock_code}
                        tick={{ fontSize: 12 }}
                        stroke="currentColor"
                        opacity={0.5}
                        width={50}
                    />
                    <Tooltip content={<CustomTooltip />} isAnimationActive={false} />
                    <Bar
                        dataKey={(item) => item.allocation || item.weight || item.percentage}
                        fill="#10b981"
                        radius={[0, 4, 4, 0]}
                    />
                </BarChart>
            </ResponsiveContainer>
        </div>
    );
};
