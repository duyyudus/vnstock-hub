import React from 'react';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from 'recharts';

interface IndustryHoldingStockItem {
    ticker: string;
    companyName?: string;
}

interface IndustryHoldingDataPoint {
    industry?: string | null;
    sector?: string | null;
    industry_name?: string | null;
    allocation?: number | null;
    weight?: number | null;
    percentage?: number | null;
    stocks?: IndustryHoldingStockItem[] | null;
}

interface PieDatum {
    name: string;
    value: number;
    stocks?: IndustryHoldingStockItem[];
}

interface PieTooltipProps {
    active?: boolean;
    payload?: Array<{
        payload: PieDatum;
    }>;
}

interface IndustryHoldingChartProps {
    data: IndustryHoldingDataPoint[];
    loading?: boolean;
}

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316'];
const formatPercent = (value: number) => {
    return `${value.toFixed(1)}%`;
};

const IndustryHoldingTooltip: React.FC<PieTooltipProps> = ({ active, payload }) => {
    if (active && payload && payload.length) {
        const datum = payload[0].payload;
        return (
            <div className="bg-base-100 border border-base-300 p-3 rounded-lg shadow-lg">
                <p className="text-sm font-semibold mb-1">{datum.name}</p>
                <p className="text-xs text-primary">
                    Allocation: {formatPercent(datum.value)}
                </p>
                {datum.stocks && datum.stocks.length > 0 && (
                    <div className="mt-2 max-h-32 overflow-y-auto pr-1">
                        <ul className="text-xs text-base-content/80 space-y-1">
                            {datum.stocks.map((stock) => (
                                <li key={stock.ticker}>
                                    <span className="font-medium">{stock.ticker}</span>
                                    {stock.companyName ? ` - ${stock.companyName}` : ''}
                                </li>
                            ))}
                        </ul>
                    </div>
                )}
            </div>
        );
    }
    return null;
};

export const IndustryHoldingChart: React.FC<IndustryHoldingChartProps> = ({ data, loading = false }) => {
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
    const chartData: PieDatum[] = data.map((item) => ({
        name: item.industry || item.sector || item.industry_name || 'Other',
        value: item.allocation || item.weight || item.percentage || 0,
        stocks: item.stocks || undefined,
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
                        outerRadius="60%"
                        fill="#8884d8"
                        dataKey="value"
                    >
                        {chartData.map((_, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                    </Pie>
                    <Tooltip content={<IndustryHoldingTooltip />} isAnimationActive={false} />
                </PieChart>
            </ResponsiveContainer>
        </div>
    );
};
