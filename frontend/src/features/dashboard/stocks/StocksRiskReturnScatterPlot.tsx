import React, { useEffect, useMemo, useState } from 'react';
import {
    ScatterChart,
    Scatter,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    ReferenceLine,
    Cell,
    ZAxis,
    LabelList,
} from 'recharts';
import { stockApi } from '../../../api/stockApi';
import type { Stock, StocksWeeklyPricesResponse, WeeklyPricePoint } from '../../../api/stockApi';
import type { DateRange } from './utils/dateRange';

interface StocksRiskReturnScatterPlotProps {
    stocks: Stock[];
    dateRange: DateRange;
}

type Benchmark = 'VNINDEX' | 'VN30';

interface RiskMetrics {
    return: number;
    volatility: number;
    sharpe: number;
}

interface ScatterPoint {
    ticker: string;
    name: string;
    x: number;
    y: number;
    sharpe: number;
    color: string;
}

interface TooltipPayloadEntry {
    payload: ScatterPoint;
}

const RISK_FREE_RATE = 4;

const getColor = (sharpe: number): string => {
    if (sharpe >= 1) return '#10b981';
    if (sharpe >= 0.5) return '#3b82f6';
    if (sharpe >= 0) return '#f59e0b';
    return '#ef4444';
};

const calculateMetrics = (history: WeeklyPricePoint[], startStr: string, endStr: string): RiskMetrics | null => {
    const filtered = history
        .filter((point) => point.date >= startStr && point.date <= endStr)
        .sort((a, b) => a.date.localeCompare(b.date));

    if (filtered.length < 2) return null;

    const initial = filtered[0].close;
    const latest = filtered[filtered.length - 1].close;

    if (!initial || initial <= 0) return null;

    const totalReturn = ((latest / initial) - 1) * 100;
    if (!Number.isFinite(totalReturn) || Math.abs(totalReturn) > 1000) return null;

    const logReturns: number[] = [];
    for (let i = 1; i < filtered.length; i++) {
        const prev = filtered[i - 1].close;
        const curr = filtered[i].close;
        if (prev > 0 && curr > 0) {
            const value = Math.log(curr / prev);
            if (Number.isFinite(value)) {
                logReturns.push(value);
            }
        }
    }

    if (logReturns.length < 2) {
        const sharpe = (totalReturn - RISK_FREE_RATE) / 1;
        return { return: totalReturn, volatility: 0, sharpe };
    }

    const mean = logReturns.reduce((sum, value) => sum + value, 0) / logReturns.length;
    const variance = logReturns.reduce((sum, value) => sum + Math.pow(value - mean, 2), 0) / (logReturns.length - 1);
    const volatility = Math.sqrt(variance * 52) * 100;
    if (!Number.isFinite(volatility) || volatility < 0 || volatility > 300) return null;

    const sharpe = (totalReturn - RISK_FREE_RATE) / (volatility || 1);
    return { return: totalReturn, volatility, sharpe };
};

export const StocksRiskReturnScatterPlot: React.FC<StocksRiskReturnScatterPlotProps> = ({ stocks, dateRange }) => {
    const [benchmark, setBenchmark] = useState<Benchmark>('VNINDEX');
    const [priceData, setPriceData] = useState<StocksWeeklyPricesResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [isSyncing, setIsSyncing] = useState(false);

    const symbols = useMemo(() => stocks.map((stock) => stock.ticker), [stocks]);
    const startStr = dateRange.startDate;
    const endStr = dateRange.endDate;

    useEffect(() => {
        const fetchData = async () => {
            if (symbols.length === 0) {
                setPriceData(null);
                setError(null);
                setIsSyncing(false);
                return;
            }

            setLoading(true);
            setError(null);

            try {
                const response = await stockApi.getStocksWeeklyPrices(symbols, startStr, endStr, true);
                setPriceData(response);
                setIsSyncing(response.is_syncing || false);
            } catch (err) {
                console.error('Error fetching weekly prices for risk/return chart:', err);
                setError('Failed to load risk/return data. Please try again.');
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, [endStr, startStr, symbols]);

    useEffect(() => {
        if (!isSyncing || symbols.length === 0) return;

        const pollForFreshData = async () => {
            try {
                const response = await stockApi.getStocksWeeklyPrices(symbols, startStr, endStr, true);
                setPriceData(response);
                setIsSyncing(response.is_syncing || false);
            } catch (err) {
                console.error('Error polling weekly prices for risk/return chart:', err);
            }
        };

        const interval = setInterval(pollForFreshData, 5000);
        return () => clearInterval(interval);
    }, [endStr, isSyncing, startStr, symbols]);

    const chartData = useMemo(() => {
        if (!priceData?.stocks?.length) return [];

        return priceData.stocks
            .map((stock): ScatterPoint | null => {
                const metrics = calculateMetrics(stock.prices, startStr, endStr);
                if (!metrics) return null;

                return {
                    ticker: stock.ticker,
                    name: stock.company_name || stock.ticker,
                    x: metrics.volatility,
                    y: metrics.return,
                    sharpe: metrics.sharpe,
                    color: getColor(metrics.sharpe),
                };
            })
            .filter((entry): entry is ScatterPoint => entry !== null)
            .filter((entry) => Number.isFinite(entry.x) && Number.isFinite(entry.y));
    }, [endStr, priceData, startStr]);

    const benchmarkPoint = useMemo(() => {
        const benchmarkHistory = priceData?.benchmarks?.[benchmark];
        if (!benchmarkHistory?.length) return null;

        const metrics = calculateMetrics(benchmarkHistory, startStr, endStr);
        if (!metrics) return null;

        return {
            ticker: benchmark,
            name: benchmark === 'VNINDEX' ? 'VN-Index' : 'VN30',
            x: metrics.volatility,
            y: metrics.return,
            sharpe: metrics.sharpe,
        };
    }, [benchmark, endStr, priceData, startStr]);

    const cmlSlope = benchmarkPoint && benchmarkPoint.x > 0
        ? (benchmarkPoint.y - RISK_FREE_RATE) / benchmarkPoint.x
        : 0.5;

    const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: TooltipPayloadEntry[] }) => {
        if (active && payload && payload.length > 0) {
            const data = payload[0].payload;
            return (
                <div className="bg-base-100 border border-base-300 p-3 rounded-lg shadow-lg">
                    <p className="text-sm font-semibold">{data.ticker}</p>
                    <p className="text-xs text-base-content/70">{data.name}</p>
                    <p className="text-xs text-primary">Return: {data.y.toFixed(1)}%</p>
                    <p className="text-xs text-secondary">Volatility (Ann.): {data.x.toFixed(1)}%</p>
                    <p className="text-xs" style={{ color: data.color }}>
                        Sharpe: {data.sharpe.toFixed(2)}
                    </p>
                </div>
            );
        }
        return null;
    };

    const xDataMax = chartData.length > 0 ? Math.max(...chartData.map((point) => point.x), benchmarkPoint?.x || 0) : 0;
    const yDataMax = chartData.length > 0 ? Math.max(...chartData.map((point) => point.y), benchmarkPoint?.y || 0) : 0;

    const xLimit = Math.min(xDataMax * 1.1, 100);
    const yLimit = Math.max(yDataMax * 1.1, 10);

    let lineEndX = xLimit;
    let lineEndY = RISK_FREE_RATE + (cmlSlope * lineEndX);

    if (lineEndY > yLimit) {
        lineEndY = yLimit;
        lineEndX = (lineEndY - RISK_FREE_RATE) / (cmlSlope || 0.1);
    }

    const safeLineEndX = Number.isFinite(lineEndX) ? Math.max(0, lineEndX) : 50;
    const safeLineEndY = Number.isFinite(lineEndY) ? lineEndY : 50;

    return (
        <div className="w-full h-full flex flex-col space-y-4">
            <div className="flex flex-wrap items-center gap-4 border-b border-base-300 pb-2">
                <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-base-content/70">Vs:</span>
                    <button
                        className={`btn btn-sm ${benchmark === 'VN30' ? 'btn-secondary' : 'btn-accent'}`}
                        onClick={() => setBenchmark((prev) => prev === 'VNINDEX' ? 'VN30' : 'VNINDEX')}
                        title={`Click to switch to ${benchmark === 'VN30' ? 'VN-Index' : 'VN30'}`}
                    >
                        {benchmark === 'VNINDEX' ? 'VN-Index' : 'VN30'}
                    </button>
                </div>

                {isSyncing && (
                    <div className="ml-auto flex items-center gap-1 text-xs text-warning">
                        <span className="loading loading-spinner loading-xs"></span>
                        Syncing...
                    </div>
                )}
            </div>

            {loading ? (
                <div className="flex flex-col items-center justify-center h-96 gap-4">
                    <span className="loading loading-spinner loading-lg text-primary"></span>
                    <p className="text-base-content/70">Loading risk/return data...</p>
                </div>
            ) : error ? (
                <div className="flex flex-col items-center justify-center h-96 gap-4">
                    <div className="alert alert-error max-w-md">
                        <svg xmlns="http://www.w3.org/2000/svg" className="stroke-current shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <span>{error}</span>
                    </div>
                </div>
            ) : chartData.length === 0 ? (
                <div className="flex items-center justify-center h-96 text-base-content/50">
                    No risk/return data available
                </div>
            ) : (
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

                            <ReferenceLine
                                segment={[
                                    { x: 0, y: RISK_FREE_RATE },
                                    { x: safeLineEndX, y: safeLineEndY },
                                ]}
                                stroke="#fbbf24"
                                strokeDasharray="5 5"
                                strokeWidth={2}
                            />

                            <ReferenceLine y={0} stroke="#6b7280" strokeOpacity={0.5} />

                            <Scatter name="Stocks" data={chartData}>
                                {chartData.map((entry, index) => (
                                    <Cell key={`cell-${entry.ticker}-${index}`} fill={entry.color} />
                                ))}
                                <LabelList
                                    dataKey="ticker"
                                    position="right"
                                    offset={6}
                                    fontSize={10}
                                    fill="currentColor"
                                />
                            </Scatter>

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
                        * Based on stock data from {dateRange.startDate} to {dateRange.endDate}. Total return is not annualized. Volatility is annualized.
                    </p>
                </div>
            )}
        </div>
    );
};

export default StocksRiskReturnScatterPlot;
