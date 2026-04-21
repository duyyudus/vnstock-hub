import React, { useState, useMemo } from 'react';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid,
    Tooltip, ResponsiveContainer, Legend,
} from 'recharts';
import type { Stock } from '../../../api/stockApi';

interface StocksComparisonChartProps {
    stocks: Stock[];
}

type SortMetric = 'market_cap' | 'charter_capital' | 'pe_ratio';

const STOCK_COUNT_PRESETS = [10, 15, 20, 30];

const formatBillionVND = (value: number): string => {
    if (value >= 1000) return `${(value / 1000).toFixed(1)}K`;
    return value.toFixed(0);
};

interface ComparisonChartDatum {
    ticker: string;
    company_name: string;
    market_cap: number;
    charter_capital: number;
    pe_ratio: number | null;
}

interface ComparisonTooltipPayload {
    payload: ComparisonChartDatum;
}

interface ComparisonTooltipProps {
    active?: boolean;
    payload?: ComparisonTooltipPayload[];
}

const CustomTooltip: React.FC<ComparisonTooltipProps> = ({ active, payload }) => {
    if (active && payload && payload.length) {
        const data = payload[0].payload;
        return (
            <div className="bg-base-100 border border-base-300 p-3 rounded-lg shadow-lg">
                <p className="text-sm font-semibold mb-1">{data.ticker}</p>
                <p className="text-xs text-base-content/70 mb-2">{data.company_name}</p>
                <p className="text-xs" style={{ color: '#3b82f6' }}>
                    Market Cap: {formatBillionVND(data.market_cap)} B
                </p>
                <p className="text-xs" style={{ color: '#10b981' }}>
                    Charter Capital: {formatBillionVND(data.charter_capital)} B
                </p>
                <p className="text-xs" style={{ color: '#f59e0b' }}>
                    P/E Ratio: {data.pe_ratio != null ? data.pe_ratio.toFixed(1) : 'N/A'}
                </p>
            </div>
        );
    }
    return null;
};

export const StocksComparisonChart: React.FC<StocksComparisonChartProps> = ({ stocks }) => {
    const [sortMetric, setSortMetric] = useState<SortMetric>('market_cap');
    const [stockCount, setStockCount] = useState<number>(15);

    const validStocks = useMemo(
        () => stocks.filter(s => s.market_cap > 0),
        [stocks],
    );

    const countOptions = useMemo(() => {
        const total = validStocks.length;
        const opts = STOCK_COUNT_PRESETS.filter(n => n < total);
        // Always include total as the last "All" option
        if (total > 0) opts.push(total);
        return opts;
    }, [validStocks.length]);

    // Clamp stockCount to available range when stocks change
    const effectiveCount = useMemo(() => {
        if (countOptions.length === 0) return 0;
        if (stockCount > validStocks.length) return validStocks.length;
        // Snap to nearest available option
        const closest = countOptions.reduce((prev, curr) =>
            Math.abs(curr - stockCount) < Math.abs(prev - stockCount) ? curr : prev
        );
        return closest;
    }, [stockCount, countOptions, validStocks.length]);

    const chartData = useMemo(() => {
        const sorted = [...validStocks]
            .sort((a, b) => {
                const aVal = a[sortMetric] ?? 0;
                const bVal = b[sortMetric] ?? 0;
                return bVal - aVal;
            })
            .slice(0, effectiveCount);

        return sorted.map(stock => ({
            ticker: stock.ticker,
            company_name: stock.company_name,
            market_cap: stock.market_cap,
            charter_capital: stock.charter_capital,
            pe_ratio: stock.pe_ratio,
        }));
    }, [validStocks, sortMetric, effectiveCount]);

    if (stocks.length === 0) {
        return (
            <div className="flex items-center justify-center h-64 text-base-content/50">
                No comparison data available.
            </div>
        );
    }

    return (
        <div className="w-full h-full flex flex-col space-y-4">
            {/* Controls Bar */}
            <div className="flex flex-wrap items-center gap-4 border-b border-base-300 pb-2">
                <label className="flex items-center gap-2 text-sm">
                    Sort by:
                    <select
                        className="select select-sm select-bordered"
                        value={sortMetric}
                        onChange={(e) => setSortMetric(e.target.value as SortMetric)}
                    >
                        <option value="market_cap">Market Cap</option>
                        <option value="charter_capital">Charter Capital</option>
                        <option value="pe_ratio">P/E Ratio</option>
                    </select>
                </label>
                <label className="flex items-center gap-2 text-sm">
                    Show:
                    <select
                        className="select select-sm select-bordered"
                        value={effectiveCount}
                        onChange={(e) => setStockCount(Number(e.target.value))}
                    >
                        {countOptions.map(n => (
                            <option key={n} value={n}>
                                {n === validStocks.length ? `All ${n}` : `Top ${n}`}
                            </option>
                        ))}
                    </select>
                </label>
                <span className="ml-auto text-xs text-base-content/50">
                    {chartData.length} of {validStocks.length} stocks
                </span>
            </div>

            {/* Chart */}
            <div className="flex-1 min-h-0">
                <ResponsiveContainer width="100%" height={600} debounce={50}>
                    <BarChart
                        data={chartData}
                        margin={{ top: 10, right: 10, left: 10, bottom: 20 }}
                        barCategoryGap="10%"
                        barGap={1}
                    >
                        <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} />

                        <XAxis
                            dataKey="ticker"
                            tick={{ fontSize: 11 }}
                            stroke="currentColor"
                            opacity={0.5}
                            angle={-45}
                            textAnchor="end"
                            height={60}
                            interval={0}
                        />

                        {/* Left Y-Axis: Capital values (Billion VND) */}
                        <YAxis
                            yAxisId="left"
                            orientation="left"
                            width={55}
                            tick={{ fontSize: 11 }}
                            stroke="currentColor"
                            opacity={0.5}
                            tickFormatter={(value: number) => formatBillionVND(value)}
                            label={{
                                value: 'Billion VND',
                                angle: -90,
                                position: 'insideLeft',
                                offset: 0,
                                style: { fontSize: 10, fill: 'currentColor', opacity: 0.6 },
                            }}
                        />

                        {/* Right Y-Axis: P/E Ratio */}
                        <YAxis
                            yAxisId="right"
                            orientation="right"
                            width={45}
                            tick={{ fontSize: 11 }}
                            stroke="currentColor"
                            opacity={0.5}
                            domain={[0, 'auto']}
                            label={{
                                value: 'P/E',
                                angle: 90,
                                position: 'insideRight',
                                offset: 0,
                                style: { fontSize: 10, fill: 'currentColor', opacity: 0.6 },
                            }}
                        />

                        <Tooltip content={<CustomTooltip />} isAnimationActive={false} />
                        <Legend verticalAlign="top" height={36} />

                        <Bar
                            yAxisId="left"
                            dataKey="market_cap"
                            name="Market Cap"
                            fill="#3b82f6"
                            radius={[2, 2, 0, 0]}
                        />
                        <Bar
                            yAxisId="left"
                            dataKey="charter_capital"
                            name="Charter Capital"
                            fill="#10b981"
                            radius={[2, 2, 0, 0]}
                        />
                        <Bar
                            yAxisId="right"
                            dataKey="pe_ratio"
                            name="P/E Ratio"
                            fill="#f59e0b"
                            radius={[2, 2, 0, 0]}
                        />
                    </BarChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export default StocksComparisonChart;
