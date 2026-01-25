import React from 'react';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from 'recharts';

interface IndustryHoldingChartProps {
    data: any[];
    loading?: boolean;
}

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316'];

export const IndustryHoldingChart: React.FC<IndustryHoldingChartProps> = ({ data, loading = false }) => {
    const formatPercent = (value: number) => {
        return `${value.toFixed(1)}%`;
    };

    const CustomTooltip = ({ active, payload }: any) => {
        if (active && payload && payload.length) {
            const data = payload[0];
            return (
                <div className="bg-base-100 border border-base-300 p-3 rounded-lg shadow-lg">
                    <p className="text-sm font-semibold mb-1">{data.name}</p>
                    <p className="text-xs text-primary">
                        Allocation: {formatPercent(data.value)}
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
                No industry data available
            </div>
        );
    }

    // Transform data for pie chart
    const chartData = data.map(item => ({
        name: item.industry || item.sector || item.industry_name || 'Other',
        value: item.allocation || item.weight || item.percentage || 0
    }));

    return (
        <div className="w-full h-full">
            <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                    <Pie
                        data={chartData}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ x, y, cx, name, percent }) => (
                            <text
                                x={x}
                                y={y}
                                fill="currentColor"
                                textAnchor={x > cx ? 'start' : 'end'}
                                dominantBaseline="central"
                                className="text-[10px]"
                                style={{ fontSize: '10px' }}
                            >
                                {`${name}: ${((percent || 0) * 100).toFixed(0)}%`}
                            </text>
                        )}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                    >
                        {chartData.map((_, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                    </Pie>
                    <Tooltip content={<CustomTooltip />} isAnimationActive={false} />
                </PieChart>
            </ResponsiveContainer>
        </div>
    );
};
