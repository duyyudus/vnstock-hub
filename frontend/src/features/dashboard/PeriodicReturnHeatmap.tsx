import React, { useMemo } from 'react';
import type { FundPerformanceMetrics } from '../../api/stockApi';

interface PeriodicReturnHeatmapProps {
    funds: FundPerformanceMetrics[];
    benchmark: FundPerformanceMetrics | null;
    startYear?: number;
}

// Color scale from red (negative) to green (positive)
const getHeatmapColor = (value: number | undefined): string => {
    if (value === undefined || value === null) return '#374151'; // gray-700

    // Scale: -30% = deep red, 0% = neutral, +50% = deep green
    const normalized = Math.max(-30, Math.min(50, value));

    if (normalized < 0) {
        // Red scale: -30 to 0 maps to deep red to light red
        const intensity = 1 - (normalized / -30);
        const r = 239;
        const g = Math.round(68 + intensity * 100);
        const b = Math.round(68 + intensity * 100);
        return `rgb(${r}, ${g}, ${b})`;
    } else {
        // Green scale: 0 to 50 maps to light green to deep green
        const intensity = normalized / 50;
        const r = Math.round(220 - intensity * 180);
        const g = Math.round(220 - intensity * 80);
        const b = Math.round(220 - intensity * 180);
        return `rgb(${r}, ${g}, ${b})`;
    }
};

const getTextColor = (value: number | undefined): string => {
    if (value === undefined || value === null) return '#9ca3af';
    if (Math.abs(value) > 20) return '#ffffff';
    return '#1f2937';
};

export const PeriodicReturnHeatmap: React.FC<PeriodicReturnHeatmapProps> = ({
    funds,
    benchmark,
    startYear,
}) => {
    // Get all available years
    const years = useMemo(() => {
        const allYears = new Set<string>();
        funds.forEach(fund => {
            Object.keys(fund.yearly_returns).forEach(year => allYears.add(year));
        });
        if (benchmark) {
            Object.keys(benchmark.yearly_returns).forEach(year => allYears.add(year));
        }

        const sortedYears = Array.from(allYears).sort();

        if (startYear) {
            return sortedYears.filter(year => parseInt(year) >= startYear);
        }

        return sortedYears;
    }, [funds, benchmark, startYear]);

    // Sort funds by average return
    const sortedFunds = useMemo(() => {
        return [...funds].sort((a, b) => {
            const aAvg = Object.values(a.yearly_returns).reduce((sum, v) => sum + v, 0) /
                (Object.values(a.yearly_returns).length || 1);
            const bAvg = Object.values(b.yearly_returns).reduce((sum, v) => sum + v, 0) /
                (Object.values(b.yearly_returns).length || 1);
            return bAvg - aAvg;
        });
    }, [funds]);

    if (sortedFunds.length === 0 || years.length === 0) {
        return (
            <div className="flex items-center justify-center h-full text-base-content/50">
                No yearly return data available
            </div>
        );
    }

    return (
        <div className="w-full overflow-x-auto">
            <table className="w-full text-xs">
                <thead className="sticky top-0 bg-base-100 z-10">
                    <tr>
                        <th className="text-left py-1 px-3 font-semibold border-b border-base-300 min-w-[140px]">
                            Fund
                        </th>
                        {years.map(year => (
                            <th
                                key={year}
                                className="text-center py-1 px-2 font-semibold border-b border-base-300 min-w-[60px]"
                            >
                                {year}
                            </th>
                        ))}
                        <th className="text-center py-1 px-2 font-semibold border-b border-base-300 min-w-[70px]">
                            Avg
                        </th>
                    </tr>
                </thead>
                <tbody>
                    {/* Benchmark row first */}
                    {benchmark && (
                        <tr className="border-b-2 border-yellow-400/50">
                            <td className="py-1 px-3 font-semibold flex items-center gap-1">
                                ⭐ {benchmark.name}
                            </td>
                            {years.map(year => {
                                const value = benchmark.yearly_returns[year];
                                return (
                                    <td
                                        key={year}
                                        className="text-center py-1 px-1"
                                        style={{
                                            backgroundColor: getHeatmapColor(value),
                                            color: getTextColor(value),
                                        }}
                                    >
                                        {value !== undefined ? `${value > 0 ? '+' : ''}${value.toFixed(1)}%` : '-'}
                                    </td>
                                );
                            })}
                            <td className="text-center py-1 px-1 font-semibold bg-base-200">
                                {(() => {
                                    const values = Object.values(benchmark.yearly_returns);
                                    const avg = values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
                                    return avg !== null ? `${avg > 0 ? '+' : ''}${avg.toFixed(1)}%` : '-';
                                })()}
                            </td>
                        </tr>
                    )}

                    {/* Fund rows */}
                    {sortedFunds.map((fund, idx) => {
                        const values = Object.values(fund.yearly_returns);
                        const avg = values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;

                        return (
                            <tr
                                key={fund.symbol}
                                className={`border-b border-base-300/50 ${idx % 2 === 0 ? 'bg-base-100' : 'bg-base-200/30'}`}
                            >
                                <td className="py-0.5 px-3 truncate max-w-[140px]" title={fund.name}>
                                    {fund.symbol}
                                </td>
                                {years.map(year => {
                                    const value = fund.yearly_returns[year];
                                    return (
                                        <td
                                            key={year}
                                            className="text-center py-0.5 px-1"
                                            style={{
                                                backgroundColor: getHeatmapColor(value),
                                                color: getTextColor(value),
                                            }}
                                        >
                                            {value !== undefined ? `${value > 0 ? '+' : ''}${value.toFixed(1)}%` : '-'}
                                        </td>
                                    );
                                })}
                                <td
                                    className="text-center py-0.5 px-1 font-medium"
                                    style={{
                                        backgroundColor: getHeatmapColor(avg ?? undefined),
                                        color: getTextColor(avg ?? undefined),
                                    }}
                                >
                                    {avg !== null ? `${avg > 0 ? '+' : ''}${avg.toFixed(1)}%` : '-'}
                                </td>
                            </tr>
                        );
                    })}
                </tbody>
            </table>

            {/* Legend */}
            <div className="flex justify-center items-center gap-2 mt-4 mb-2 text-xs">
                <span className="text-base-content/50">Returns:</span>
                <div className="flex items-center">
                    <div className="w-8 h-4" style={{ backgroundColor: getHeatmapColor(-30) }}></div>
                    <span className="ml-1">-30%</span>
                </div>
                <div className="flex items-center">
                    <div className="w-8 h-4" style={{ backgroundColor: getHeatmapColor(-10) }}></div>
                    <span className="ml-1">-10%</span>
                </div>
                <div className="flex items-center">
                    <div className="w-8 h-4" style={{ backgroundColor: getHeatmapColor(0) }}></div>
                    <span className="ml-1">0%</span>
                </div>
                <div className="flex items-center">
                    <div className="w-8 h-4" style={{ backgroundColor: getHeatmapColor(20) }}></div>
                    <span className="ml-1">+20%</span>
                </div>
                <div className="flex items-center">
                    <div className="w-8 h-4" style={{ backgroundColor: getHeatmapColor(50) }}></div>
                    <span className="ml-1">+50%</span>
                </div>
            </div>
        </div>
    );
};

export default PeriodicReturnHeatmap;
