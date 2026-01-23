import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface NavReportChartProps {
    data: any[];
    loading?: boolean;
}

export const NavReportChart: React.FC<NavReportChartProps> = ({ data, loading = false }) => {
    const formatDate = (dateStr: string) => {
        const date = new Date(dateStr);
        return `${date.getMonth() + 1}/${date.getDate()}`;
    };

    const formatValue = (value: number) => {
        return value.toLocaleString();
    };

    const CustomTooltip = ({ active, payload }: any) => {
        if (active && payload && payload.length) {
            const data = payload[0].payload;
            return (
                <div className="bg-base-100 border border-base-300 p-3 rounded-lg shadow-lg">
                    <p className="text-sm font-semibold mb-1">{data.date || data.nav_date || 'N/A'}</p>
                    <p className="text-xs text-primary">
                        NAV: {formatValue(data.nav || data.value || 0)}
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
                No NAV data available
            </div>
        );
    }

    return (
        <div className="w-full h-full">
            <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} />
                    <XAxis
                        dataKey={(item) => item.date || item.nav_date}
                        tickFormatter={formatDate}
                        tick={{ fontSize: 12 }}
                        stroke="currentColor"
                        opacity={0.5}
                    />
                    <YAxis
                        tickFormatter={formatValue}
                        tick={{ fontSize: 12 }}
                        stroke="currentColor"
                        opacity={0.5}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Line
                        type="monotone"
                        dataKey={(item) => item.nav || item.value}
                        stroke="#3b82f6"
                        strokeWidth={2}
                        dot={false}
                    />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
};
