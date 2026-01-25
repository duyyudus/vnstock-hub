import React, { useMemo } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Cell, ZAxis } from 'recharts';
import type { FundPerformanceMetrics } from '../../api/stockApi';

interface RiskReturnScatterPlotProps {
    funds: FundPerformanceMetrics[];
    benchmark: FundPerformanceMetrics | null;
    startYear: number;
}

// Color based on Sharpe ratio (efficiency)
const getColor = (sharpe: number | null): string => {
    if (sharpe === null) return '#6b7280'; // gray
    if (sharpe >= 1) return '#10b981';      // green - excellent
    if (sharpe >= 0.5) return '#3b82f6';    // blue - good
    if (sharpe >= 0) return '#f59e0b';      // yellow - neutral
    return '#ef4444';                        // red - poor
};

// Helper to calculate return and volatility from history
const calculateMetrics = (history: { date: string, normalized_nav: number }[], startStr: string) => {
    const filtered = history
        .filter(p => p.date >= startStr)
        .sort((a, b) => a.date.localeCompare(b.date));

    if (filtered.length < 2) return null;

    const initial = filtered[0].normalized_nav;
    const latest = filtered[filtered.length - 1].normalized_nav;

    // Guard against division by zero or very small initial values
    if (!initial || initial <= 0.01) return null;

    const totalReturn = ((latest / initial) - 1) * 100;

    // Skip if return is unrealistically high (data issue)
    if (!isFinite(totalReturn) || Math.abs(totalReturn) > 1000) return null;

    // Calculate periodic returns for volatility
    const returns: number[] = [];
    for (let i = 1; i < filtered.length; i++) {
        const prev = filtered[i - 1].normalized_nav;
        const curr = filtered[i].normalized_nav;
        if (prev > 0) {
            returns.push(Math.log(curr / prev));
        }
    }

    if (returns.length < 2) return { return: totalReturn, volatility: 0, sharpe: 0 };

    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (returns.length - 1);
    // Annualize volatility (assuming weekly data if intervals are apart, but let's approximate based on data points/year)
    // Most funds have weekly NAV reports in Vietnam. Let's assume 52 weeks/year if we have enough points.
    // Better: detect frequency or just use a standard multiplier.
    const annFactor = 52;
    const volatility = Math.sqrt(variance * annFactor) * 100;

    const riskFreeRate = 4; // 4% annual
    const sharpe = (totalReturn - riskFreeRate) / (volatility || 1);

    return { return: totalReturn, volatility, sharpe };
};

export const RiskReturnScatterPlot: React.FC<RiskReturnScatterPlotProps> = ({
    funds,
    benchmark,
    startYear,
}) => {
    const startStr = `${startYear}-01-01`;

    const chartData = useMemo(() => {
        return funds
            .map(fund => {
                const metrics = calculateMetrics(fund.nav_history, startStr);
                if (!metrics) return null;
                return {
                    symbol: fund.symbol,
                    name: fund.name,
                    x: metrics.volatility,
                    y: metrics.return,
                    sharpe: metrics.sharpe,
                    color: getColor(metrics.sharpe),
                };
            })
            .filter((d): d is NonNullable<typeof d> => d !== null)
            // Filter out extreme outliers (likely data corruption) - allow wide range for valid data
            // Returns up to 1000% possible for long timeframes, volatility rarely exceeds 200%
            .filter(d => Math.abs(d.y) <= 1000 && d.x >= 0 && d.x <= 300);
    }, [funds, startStr]);

    const benchmarkPoint = useMemo(() => {
        if (!benchmark) return null;
        const metrics = calculateMetrics(benchmark.nav_history, startStr);
        if (!metrics) return null;

        // Skip benchmark if it has extreme values
        if (Math.abs(metrics.return) > 1000 || metrics.volatility < 0 || metrics.volatility > 300) return null;

        return {
            symbol: benchmark.symbol,
            name: benchmark.name,
            x: metrics.volatility,
            y: metrics.return,
            sharpe: metrics.sharpe,
        };
    }, [benchmark, startStr]);

    // Calculate line from origin through benchmark (Capital Market Line approximation)
    const riskFreeRate = 4; // 4% risk-free rate
    const cmlSlope = (benchmarkPoint && benchmarkPoint.x! > 0)
        ? (benchmarkPoint.y - riskFreeRate) / benchmarkPoint.x!
        : 0.5;

    // ... rest of Tooltip and Guards ...
    const CustomTooltip = ({ active, payload }: any) => {
        if (active && payload && payload.length) {
            const data = payload[0].payload;
            return (
                <div className="bg-base-100 border border-base-300 p-3 rounded-lg shadow-lg">
                    <p className="text-sm font-semibold">{data.name}</p>
                    <p className="text-xs text-primary">Return: {data.y?.toFixed(1)}%</p>
                    <p className="text-xs text-secondary">Volatility (Ann.): {data.x?.toFixed(1)}%</p>
                    <p className="text-xs" style={{ color: data.color }}>
                        Sharpe: {data.sharpe?.toFixed(2) || 'N/A'}
                    </p>
                </div>
            );
        }
        return null;
    };

    if (chartData.length === 0) {
        return (
            <div className="flex items-center justify-center h-full text-base-content/50">
                No risk/return data available
            </div>
        );
    }

    // Determine axis bounds for line clipping
    const xDataMax = Math.max(...chartData.map(d => d.x), benchmarkPoint?.x || 0);
    const yDataMax = Math.max(...chartData.map(d => d.y), benchmarkPoint?.y || 0);

    const xLimit = Math.min(xDataMax * 1.1, 100);
    const yLimit = Math.max(yDataMax * 1.1, 10);

    // Clip the line so it doesn't go to infinity or break SVG rendering
    // We want the line to end at either the xLimit or the yLimit
    let lineEndX = xLimit;
    let lineEndY = riskFreeRate + (cmlSlope * lineEndX);

    if (lineEndY > yLimit) {
        lineEndY = yLimit;
        lineEndX = (lineEndY - riskFreeRate) / (cmlSlope || 0.1);
    }

    // Final sanity check for coordinates
    const safeLineEndX = isFinite(lineEndX) ? Math.max(0, lineEndX) : 50;
    const safeLineEndY = isFinite(lineEndY) ? lineEndY : 50;

    return (
        <div className="w-full">
            <ResponsiveContainer width="100%" height={680} minWidth={0}>
                <ScatterChart margin={{ top: 20, right: 30, left: 20, bottom: 50 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} />
                    <XAxis
                        type="number"
                        dataKey="x"
                        name="Volatility"
                        domain={[0, (dataMax: number) => Math.max(dataMax * 1.1, 10)]}
                        tick={{ fontSize: 11 }}
                        stroke="currentColor"
                        opacity={0.5}
                        tickFormatter={(value: number) => value.toFixed(0)}
                        label={{ value: 'Annualized Volatility (Risk %)', position: 'bottom', style: { fontSize: 11 } }}
                    />
                    <YAxis
                        type="number"
                        dataKey="y"
                        name="Return"
                        domain={[(dataMin: number) => Math.min(dataMin * 1.1, -5), (dataMax: number) => Math.max(dataMax * 1.1, 10)]}
                        tick={{ fontSize: 11 }}
                        stroke="currentColor"
                        opacity={0.5}
                        tickFormatter={(value: number) => value.toFixed(0)}
                        label={{ value: 'Total Return %', angle: -90, position: 'insideLeft', style: { fontSize: 11 } }}
                    />
                    <ZAxis type="number" range={[80, 80]} />
                    <Tooltip content={<CustomTooltip />} isAnimationActive={false} />

                    {/* Capital Market Line - funds above are "alpha" generators */}
                    <ReferenceLine
                        segment={[
                            { x: 0, y: riskFreeRate },
                            { x: safeLineEndX, y: safeLineEndY }
                        ]}
                        stroke="#fbbf24"
                        strokeDasharray="5 5"
                        strokeWidth={2}
                    />

                    {/* Horizontal line at 0% return */}
                    <ReferenceLine y={0} stroke="#6b7280" strokeOpacity={0.5} />

                    {/* Fund scatter points */}
                    <Scatter name="Funds" data={chartData}>
                        {chartData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                    </Scatter>

                    {/* Benchmark point - larger and distinct */}
                    {benchmarkPoint && (
                        <Scatter
                            name={benchmarkPoint.name}
                            data={[benchmarkPoint]}
                            shape="star"
                            fill="#fbbf24"
                        />
                    )}
                </ScatterChart>
            </ResponsiveContainer>

            {/* Legend */}
            <div className="flex flex-wrap justify-center gap-4 mt-4 mb-2 text-xs">
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded-full bg-green-500"></div>
                    <span>Sharpe ≥ 1 (Excellent)</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                    <span>Sharpe ≥ 0.5 (Good)</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded-full bg-amber-500"></div>
                    <span>Sharpe ≥ 0 (Neutral)</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded-full bg-red-500"></div>
                    <span>Sharpe &lt; 0 (Poor)</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className="w-3 h-0.5 bg-yellow-400"></div>
                    <span>Market Line</span>
                </div>
            </div>
            <p className="text-[10px] text-center text-base-content/40 mt-1">
                * Based on data starting from {startYear}. Total return is not annualized. Volatility is annualized.
            </p>
        </div>
    );
};

export default RiskReturnScatterPlot;
