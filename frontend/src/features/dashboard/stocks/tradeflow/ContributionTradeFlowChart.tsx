import React, { useMemo, useState } from 'react';
import {
    Bar,
    BarChart,
    CartesianGrid,
    LabelList,
    ReferenceLine,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from 'recharts';
import type { Stock, StocksVolumeSeriesResponse } from '../../../../api/stockApi';
import type { DateRange } from '../dateRange';

type FlowType = 'foreign' | 'proprietary';

interface ContributionTradeFlowChartProps {
    stocks: Stock[];
    flowResponse: StocksVolumeSeriesResponse | null;
    dateRange: DateRange;
}

interface ContributionDatum {
    ticker: string;
    companyName: string;
    netFlow: number;
    positiveFlow: number | null;
    negativeFlow: number | null;
    basketContributionPct: number | null;
    absolutePressureSharePct: number;
}

interface TooltipPayloadEntry {
    payload?: ContributionDatum;
}

interface ContributionTooltipProps {
    active?: boolean;
    payload?: TooltipPayloadEntry[];
}

interface ContributionChartPanelProps {
    title: string;
    modeLabel: string;
    data: ContributionDatum[];
}

const ROW_HEIGHT = 34;
const MIN_CHART_HEIGHT = 360;
const POSITIVE_COLOR = '#059669';
const NEGATIVE_COLOR = '#e11d48';

const roundToTwo = (value: number): number => Math.round(value * 100) / 100;

const formatBilVnd = (value: number): string => {
    const absolute = Math.abs(value);
    if (absolute >= 1000) return `${value < 0 ? '-' : ''}${(absolute / 1000).toFixed(1)}K`;
    if (absolute >= 100) return `${value < 0 ? '-' : ''}${absolute.toFixed(0)}`;
    if (absolute >= 10) return `${value < 0 ? '-' : ''}${absolute.toFixed(1)}`;
    return `${value < 0 ? '-' : ''}${absolute.toFixed(2)}`;
};

const formatSignedBilVnd = (value: number): string => {
    const sign = value >= 0 ? '+' : '-';
    return `${sign}${formatBilVnd(Math.abs(value))} Bil VND`;
};

const formatAxisValue = (value: number): string => formatBilVnd(value);

const formatLabelValue = (value: number | string | boolean | undefined | null): string => {
    const numericValue = typeof value === 'number' ? value : Number(value);
    if (!Number.isFinite(numericValue)) {
        return '';
    }
    return formatBilVnd(numericValue);
};

const formatPercent = (value: number | null): string => {
    if (value === null || !Number.isFinite(value)) {
        return 'N/A';
    }
    return `${value >= 0 ? '+' : ''}${value.toFixed(1)}%`;
};

const getSignedValueClassName = (value: number | null): string => {
    if (value === null) {
        return 'font-medium text-base-content/70';
    }
    return value >= 0 ? 'font-medium text-success' : 'font-medium text-error';
};

const getDomain = (data: ContributionDatum[]): [number, number] => {
    const maxAbsValue = data.reduce((maxValue, point) => Math.max(maxValue, Math.abs(point.netFlow)), 0);
    if (maxAbsValue === 0) {
        return [-1, 1];
    }
    const paddedValue = Math.max(maxAbsValue * 1.12, 1);
    return [-roundToTwo(paddedValue), roundToTwo(paddedValue)];
};

const getTicks = (domain: [number, number]): number[] => {
    const maxValue = Math.max(Math.abs(domain[0]), Math.abs(domain[1]));
    return [
        -roundToTwo(maxValue),
        -roundToTwo(maxValue / 2),
        0,
        roundToTwo(maxValue / 2),
        roundToTwo(maxValue),
    ];
};

const getStockMetaMap = (stocks: Stock[]): Map<string, Stock> => {
    return new Map(stocks.map((stock) => [stock.ticker.toUpperCase(), stock]));
};

const buildContributionRows = (
    rawRows: Array<{ ticker: string; companyName: string; netFlow: number }>,
): ContributionDatum[] => {
    const basketNetFlow = rawRows.reduce((total, row) => total + row.netFlow, 0);
    const totalAbsoluteFlow = rawRows.reduce((total, row) => total + Math.abs(row.netFlow), 0);

    return rawRows
        .map((row) => {
            const netFlow = roundToTwo(row.netFlow);
            return {
                ticker: row.ticker,
                companyName: row.companyName,
                netFlow,
                positiveFlow: netFlow >= 0 ? netFlow : null,
                negativeFlow: netFlow < 0 ? netFlow : null,
                basketContributionPct: basketNetFlow !== 0 ? (row.netFlow / basketNetFlow) * 100 : null,
                absolutePressureSharePct: totalAbsoluteFlow > 0 ? (Math.abs(row.netFlow) / totalAbsoluteFlow) * 100 : 0,
            };
        })
        .sort((a, b) => {
            const absoluteDelta = Math.abs(b.netFlow) - Math.abs(a.netFlow);
            if (absoluteDelta !== 0) {
                return absoluteDelta;
            }
            return a.ticker.localeCompare(b.ticker);
        });
};

const buildHistoricalRows = (
    flowResponse: StocksVolumeSeriesResponse | null,
    stocks: Stock[],
    activeFlow: FlowType,
): ContributionDatum[] => {
    if (!flowResponse) {
        return [];
    }

    const stockMetaMap = getStockMetaMap(stocks);
    const valueKey = activeFlow === 'foreign' ? 'foreign_net_value' : 'prop_net_value';
    const rawRows = flowResponse.stocks.flatMap((stockSeries) => {
        let total = 0;
        let hasValue = false;

        stockSeries.data.forEach((point) => {
            const value = point[valueKey];
            if (value !== null) {
                total += value;
                hasValue = true;
            }
        });

        if (!hasValue) {
            return [];
        }

        const ticker = stockSeries.ticker.toUpperCase();
        const stockMeta = stockMetaMap.get(ticker);
        return [{
            ticker,
            companyName: stockSeries.company_name || stockMeta?.company_name || ticker,
            netFlow: total,
        }];
    });

    return buildContributionRows(rawRows);
};

const buildTodayRows = (stocks: Stock[]): ContributionDatum[] => {
    const rawRows = stocks.flatMap((stock) => {
        if (stock.foreign_buy_value === null || stock.foreign_sell_value === null) {
            return [];
        }

        const ticker = stock.ticker.toUpperCase();
        return [{
            ticker,
            companyName: stock.company_name || ticker,
            netFlow: stock.foreign_buy_value - stock.foreign_sell_value,
        }];
    });

    return buildContributionRows(rawRows);
};

const ContributionTooltip: React.FC<ContributionTooltipProps> = ({ active, payload }) => {
    const datum = payload?.[0]?.payload;
    if (!active || !datum) {
        return null;
    }

    return (
        <div className="rounded-lg border border-base-300 bg-base-100 p-3 shadow-lg">
            <p className="text-sm font-semibold">{datum.ticker}</p>
            <p className="mt-1 max-w-60 whitespace-normal text-xs text-base-content/70">{datum.companyName}</p>
            <div className="mt-3 space-y-1 text-xs">
                <p>
                    Net flow:{' '}
                    <span className={datum.netFlow >= 0 ? 'font-medium text-success' : 'font-medium text-error'}>
                        {formatSignedBilVnd(datum.netFlow)}
                    </span>
                </p>
                <p>
                    Basket net contribution:{' '}
                    <span className="font-medium">{formatPercent(datum.basketContributionPct)}</span>
                </p>
                <p>
                    Absolute pressure share:{' '}
                    <span className="font-medium">{datum.absolutePressureSharePct.toFixed(1)}%</span>
                </p>
            </div>
        </div>
    );
};

const ContributionChartPanel: React.FC<ContributionChartPanelProps> = ({ title, modeLabel, data }) => {
    const domain = useMemo(() => getDomain(data), [data]);
    const ticks = useMemo(() => getTicks(domain), [domain]);
    const basketNetFlow = useMemo(
        () => data.reduce((total, point) => total + point.netFlow, 0),
        [data],
    );
    const chartHeight = Math.max(MIN_CHART_HEIGHT, data.length * ROW_HEIGHT + 82);
    const basketNetValue = data.length > 0 ? roundToTwo(basketNetFlow) : null;

    return (
        <div className="min-w-0 space-y-3 rounded-md border border-base-300 p-3">
            <div className="flex flex-col gap-1 border-b border-base-300 pb-2">
                <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="text-sm font-semibold text-base-content">{title}</div>
                    <div className="text-xs text-base-content/60">{modeLabel}</div>
                </div>
                <div className="text-xs text-base-content/70">
                    Basket net:{' '}
                    <span className={getSignedValueClassName(basketNetValue)}>
                        {basketNetValue !== null ? formatSignedBilVnd(basketNetValue) : 'N/A'}
                    </span>
                </div>
            </div>

            {data.length === 0 ? (
                <div className="flex h-64 items-center justify-center text-sm text-base-content/50">
                    No {title.toLowerCase()} contribution data available for this mode.
                </div>
            ) : (
                <div className="w-full overflow-x-auto">
                    <div className="min-w-[520px] w-full" style={{ height: `${chartHeight}px` }}>
                        <ResponsiveContainer width="100%" height="100%" debounce={50}>
                            <BarChart
                                data={data}
                                layout="vertical"
                                margin={{ top: 16, right: 12, left: 0, bottom: 34 }}
                                barCategoryGap="18%"
                            >
                                <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} />
                                <XAxis
                                    type="number"
                                    domain={domain}
                                    ticks={ticks}
                                    tickFormatter={formatAxisValue}
                                    tick={{ fontSize: 11 }}
                                    stroke="currentColor"
                                    opacity={0.55}
                                    label={{ value: 'Net Flow (Bil VND)', position: 'insideBottom', offset: -18, style: { fontSize: 11 } }}
                                />
                                <YAxis
                                    type="category"
                                    dataKey="ticker"
                                    tick={{ fontSize: 12, fontWeight: 600 }}
                                    stroke="currentColor"
                                    opacity={0.75}
                                    width={58}
                                    tickMargin={8}
                                    interval={0}
                                />
                                <Tooltip content={<ContributionTooltip />} isAnimationActive={false} />
                                <ReferenceLine x={0} stroke="currentColor" strokeOpacity={0.38} />
                                <Bar
                                    dataKey="negativeFlow"
                                    fill={NEGATIVE_COLOR}
                                    maxBarSize={22}
                                    radius={[4, 0, 0, 4]}
                                    isAnimationActive={false}
                                >
                                    <LabelList
                                        dataKey="negativeFlow"
                                        position="left"
                                        formatter={formatLabelValue}
                                        fill={NEGATIVE_COLOR}
                                        fontSize={11}
                                    />
                                </Bar>
                                <Bar
                                    dataKey="positiveFlow"
                                    fill={POSITIVE_COLOR}
                                    maxBarSize={22}
                                    radius={[0, 4, 4, 0]}
                                    isAnimationActive={false}
                                >
                                    <LabelList
                                        dataKey="positiveFlow"
                                        position="right"
                                        formatter={formatLabelValue}
                                        fill={POSITIVE_COLOR}
                                        fontSize={11}
                                    />
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            )}
        </div>
    );
};

export const ContributionTradeFlowChart: React.FC<ContributionTradeFlowChartProps> = ({
    stocks,
    flowResponse,
    dateRange,
}) => {
    const [useToday, setUseToday] = useState(false);

    const foreignChartData = useMemo(() => {
        if (useToday) {
            return buildTodayRows(stocks);
        }
        return buildHistoricalRows(flowResponse, stocks, 'foreign');
    }, [flowResponse, stocks, useToday]);
    const proprietaryChartData = useMemo(
        () => buildHistoricalRows(flowResponse, stocks, 'proprietary'),
        [flowResponse, stocks],
    );
    const rangeModeLabel = `${dateRange.startDate} to ${dateRange.endDate}`;
    const foreignModeLabel = useToday ? 'current session' : rangeModeLabel;

    return (
        <section className="space-y-3">
            <div className="flex flex-col gap-3 border-b border-base-300 pb-3 lg:flex-row lg:items-start lg:justify-between">
                <div>
                    <div className="text-sm font-semibold text-base-content">Capital flow contribution</div>
                    <div className="mt-1 text-xs text-base-content/60">
                        All selected stocks by absolute net buy/sell value
                    </div>
                    <div className="mt-2 flex flex-wrap items-center gap-3 text-xs text-base-content/70">
                        <span className="inline-flex items-center gap-1.5">
                            <span className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: POSITIVE_COLOR }} />
                            Net buy
                        </span>
                        <span className="inline-flex items-center gap-1.5">
                            <span className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: NEGATIVE_COLOR }} />
                            Net sell
                        </span>
                    </div>
                </div>

                <div className="flex flex-wrap items-center gap-3">
                    <label
                        className="flex items-center gap-2 text-xs text-base-content/70"
                        title="Use current trading session data for the foreign chart only"
                    >
                        <span>Today foreign</span>
                        <input
                            type="checkbox"
                            className="toggle toggle-sm"
                            checked={useToday}
                            onChange={(event) => setUseToday(event.target.checked)}
                        />
                    </label>
                </div>
            </div>

            <div className="grid gap-4 xl:grid-cols-2">
                <ContributionChartPanel
                    title="Foreign"
                    modeLabel={foreignModeLabel}
                    data={foreignChartData}
                />
                <ContributionChartPanel
                    title="Proprietary"
                    modeLabel={rangeModeLabel}
                    data={proprietaryChartData}
                />
            </div>
        </section>
    );
};

export default ContributionTradeFlowChart;
