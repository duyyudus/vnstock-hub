import React, { useMemo } from 'react';
import {
    Bar,
    BarChart,
    CartesianGrid,
    LabelList,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from 'recharts';
import type { Stock } from '../../../../api/stockApi';
import type { BarShapeProps } from 'recharts';
import { getForeignHolding } from '../foreignHolding';

interface ForeignHoldingChartProps {
    stocks: Stock[];
}

interface ForeignHoldingChartItem {
    ticker: string;
    companyName: string;
    valueBilVnd: number;
    roomRatio: number;
}

interface ForeignHoldingTooltipEntry {
    payload: ForeignHoldingChartItem;
}

interface ForeignHoldingTooltipProps {
    active?: boolean;
    payload?: ForeignHoldingTooltipEntry[];
}

const ROW_HEIGHT = 34;
const MIN_CHART_HEIGHT = 320;

const formatBilVnd = (value: number): string => new Intl.NumberFormat('en-US', {
    maximumFractionDigits: 2,
}).format(value);

const formatAxisValue = (value: number): string => formatBilVnd(value);

const formatRoomPercent = (value: number): string => new Intl.NumberFormat('en-US', {
    style: 'percent',
    maximumFractionDigits: 1,
}).format(value);

const formatBarLabel = (value: number | string | boolean | null | undefined): string => {
    const numericValue = typeof value === 'number' ? value : Number(value);
    return Number.isFinite(numericValue) ? `${formatBilVnd(numericValue)} B` : '';
};

const ForeignHoldingTooltip: React.FC<ForeignHoldingTooltipProps> = ({ active, payload }) => {
    if (!active || !payload?.length) {
        return null;
    }

    const item = payload[0].payload;
    return (
        <div className="rounded-lg border border-base-300 bg-base-100 p-3 shadow-lg">
            <p className="text-sm font-semibold">{item.ticker}</p>
            <p className="mt-1 max-w-64 whitespace-normal text-xs text-base-content/70">
                {item.companyName}
            </p>
            <p className="mt-1 text-xs text-primary">
                Foreign holding: {formatBilVnd(item.valueBilVnd)} B VND
            </p>
            <p className="mt-1 text-xs text-base-content/70">
                Room: {formatRoomPercent(item.roomRatio)}
            </p>
        </div>
    );
};

const ForeignHoldingBar = ({ x, y, width, height, payload }: BarShapeProps) => {
    const item = payload as ForeignHoldingChartItem;
    const roomRatio = Math.min(Math.max(item.roomRatio, 0), 1);
    const centerY = y + height / 2;

    return (
        <g>
            <rect x={x} y={y} width={width} height={height} rx={4} ry={4} fill="#10b981" />
            <line
                x1={x}
                y1={centerY}
                x2={x + width * roomRatio}
                y2={centerY}
                stroke="#065f46"
                strokeWidth={3}
                strokeLinecap="round"
                pointerEvents="none"
            />
        </g>
    );
};

export const ForeignHoldingChart: React.FC<ForeignHoldingChartProps> = ({ stocks }) => {
    const data = useMemo<ForeignHoldingChartItem[]>(() => stocks
        .map((stock) => {
            const holding = getForeignHolding(stock);
            if (!holding) {
                return null;
            }
            return {
                ticker: stock.ticker.toUpperCase(),
                companyName: stock.company_name || stock.ticker.toUpperCase(),
                valueBilVnd: holding.valueBilVnd,
                roomRatio: stock.current_room! / stock.total_room!,
            };
        })
        .filter((item): item is ForeignHoldingChartItem => item !== null)
        .sort((a, b) => {
            if (b.valueBilVnd !== a.valueBilVnd) {
                return b.valueBilVnd - a.valueBilVnd;
            }
            return a.ticker.localeCompare(b.ticker);
        }), [stocks]);

    if (data.length === 0) {
        return (
            <div className="flex h-80 items-center justify-center text-base-content/60">
                No valid foreign holding data available.
            </div>
        );
    }

    const chartHeight = Math.max(MIN_CHART_HEIGHT, data.length * ROW_HEIGHT);

    return (
        <div style={{ height: `${chartHeight}px` }}>
            <ResponsiveContainer width="100%" height="100%">
                <BarChart
                    data={data}
                    layout="vertical"
                    margin={{ top: 10, right: 100, left: 8, bottom: 24 }}
                >
                    <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} />
                    <XAxis
                        type="number"
                        tickFormatter={formatAxisValue}
                        tick={{ fontSize: 12 }}
                        stroke="currentColor"
                        opacity={0.5}
                        label={{ value: 'B VND', position: 'insideBottom', offset: -16 }}
                    />
                    <YAxis
                        type="category"
                        dataKey="ticker"
                        tick={{ fontSize: 12 }}
                        stroke="currentColor"
                        opacity={0.5}
                        width={48}
                    />
                    <Tooltip content={<ForeignHoldingTooltip />} isAnimationActive={false} />
                    <Bar
                        dataKey="valueBilVnd"
                        name="Foreign holding (B VND)"
                        shape={ForeignHoldingBar}
                        maxBarSize={18}
                    >
                        <LabelList
                            dataKey="valueBilVnd"
                            position="right"
                            formatter={formatBarLabel}
                            fill="currentColor"
                            fontSize={12}
                        />
                    </Bar>
                </BarChart>
            </ResponsiveContainer>
        </div>
    );
};
