import React, { useMemo, useState } from 'react';
import type { Stock, StocksVolumeSeriesResponse } from '../../../../api/stockApi';
import type { DateRange } from '../dateRange';

interface SectorRotationHeatmapProps {
    stocks: Stock[];
    flowResponse: StocksVolumeSeriesResponse | null;
    dateRange: DateRange;
}

interface SectorRotationRow {
    industry: string;
    stockCount: number;
    tradedValue: number | null;
    foreignNetFlow: number | null;
    propNetFlow: number | null;
}

interface MutableSectorRow {
    industry: string;
    stockCount: number;
    tradedValueSum: number;
    hasTradedValue: boolean;
    foreignNetFlowSum: number;
    hasForeignNetFlow: boolean;
    propNetFlowSum: number;
    hasPropNetFlow: boolean;
}

const UNCLASSIFIED_INDUSTRY = 'Unclassified';

const roundToTwo = (value: number): number => Math.round(value * 100) / 100;

const normalizeIndustry = (industry: string | null | undefined): string => {
    const normalized = industry?.trim();
    return normalized || UNCLASSIFIED_INDUSTRY;
};

const formatBilVnd = (value: number): string => {
    const absolute = Math.abs(value);
    if (absolute >= 1000) return `${value < 0 ? '-' : ''}${(absolute / 1000).toFixed(1)}K`;
    if (absolute >= 100) return `${value < 0 ? '-' : ''}${absolute.toFixed(0)}`;
    if (absolute >= 10) return `${value < 0 ? '-' : ''}${absolute.toFixed(1)}`;
    return `${value < 0 ? '-' : ''}${absolute.toFixed(2)}`;
};

const formatValueBilVnd = (value: number | null): string => {
    if (value === null || !Number.isFinite(value)) {
        return 'N/A';
    }
    return `${formatBilVnd(value)} Bil VND`;
};

const formatSignedBilVnd = (value: number | null): string => {
    if (value === null || !Number.isFinite(value)) {
        return 'N/A';
    }
    const sign = value >= 0 ? '+' : '-';
    return `${sign}${formatBilVnd(Math.abs(value))} Bil VND`;
};

const getStockMetaMap = (stocks: Stock[]): Map<string, Stock> => {
    return new Map(stocks.map((stock) => [stock.ticker.toUpperCase(), stock]));
};

const getOrCreateSectorRow = (rowsByIndustry: Map<string, MutableSectorRow>, industry: string): MutableSectorRow => {
    const existingRow = rowsByIndustry.get(industry);
    if (existingRow) {
        return existingRow;
    }

    const row: MutableSectorRow = {
        industry,
        stockCount: 0,
        tradedValueSum: 0,
        hasTradedValue: false,
        foreignNetFlowSum: 0,
        hasForeignNetFlow: false,
        propNetFlowSum: 0,
        hasPropNetFlow: false,
    };
    rowsByIndustry.set(industry, row);
    return row;
};

const toSectorRows = (rowsByIndustry: Map<string, MutableSectorRow>): SectorRotationRow[] => {
    return Array.from(rowsByIndustry.values())
        .flatMap((row) => {
            if (!row.hasTradedValue && !row.hasForeignNetFlow && !row.hasPropNetFlow) {
                return [];
            }

            return [{
                industry: row.industry,
                stockCount: row.stockCount,
                tradedValue: row.hasTradedValue ? roundToTwo(row.tradedValueSum) : null,
                foreignNetFlow: row.hasForeignNetFlow ? roundToTwo(row.foreignNetFlowSum) : null,
                propNetFlow: row.hasPropNetFlow ? roundToTwo(row.propNetFlowSum) : null,
            }];
        })
        .sort((a, b) => {
            const tradedValueDelta = (b.tradedValue ?? -1) - (a.tradedValue ?? -1);
            if (tradedValueDelta !== 0) {
                return tradedValueDelta;
            }
            return a.industry.localeCompare(b.industry);
        });
};

const buildHistoricalRows = (
    stocks: Stock[],
    flowResponse: StocksVolumeSeriesResponse | null,
): SectorRotationRow[] => {
    if (!flowResponse) {
        return [];
    }

    const stockMetaMap = getStockMetaMap(stocks);
    const rowsByIndustry = new Map<string, MutableSectorRow>();

    flowResponse.stocks.forEach((stockSeries) => {
        const ticker = stockSeries.ticker.toUpperCase();
        const stockMeta = stockMetaMap.get(ticker);
        const row = getOrCreateSectorRow(rowsByIndustry, normalizeIndustry(stockMeta?.industry));
        row.stockCount += 1;

        stockSeries.data.forEach((point) => {
            if (point.value !== null) {
                row.tradedValueSum += point.value;
                row.hasTradedValue = true;
            }
            if (point.foreign_net_value !== null) {
                row.foreignNetFlowSum += point.foreign_net_value;
                row.hasForeignNetFlow = true;
            }
            if (point.prop_net_value !== null) {
                row.propNetFlowSum += point.prop_net_value;
                row.hasPropNetFlow = true;
            }
        });
    });

    return toSectorRows(rowsByIndustry);
};

const buildTodayRows = (stocks: Stock[]): SectorRotationRow[] => {
    const rowsByIndustry = new Map<string, MutableSectorRow>();

    stocks.forEach((stock) => {
        const row = getOrCreateSectorRow(rowsByIndustry, normalizeIndustry(stock.industry));
        row.stockCount += 1;

        if (stock.accumulated_value !== null) {
            row.tradedValueSum += stock.accumulated_value;
            row.hasTradedValue = true;
        }
        if (stock.foreign_buy_value !== null && stock.foreign_sell_value !== null) {
            row.foreignNetFlowSum += stock.foreign_buy_value - stock.foreign_sell_value;
            row.hasForeignNetFlow = true;
        }
    });

    return toSectorRows(rowsByIndustry);
};

const getFlowCellClassName = (value: number | null): string => {
    const baseClassName = 'rounded px-3 py-2 text-right font-semibold';
    if (value === null) {
        return `${baseClassName} bg-base-200/60 text-base-content/45`;
    }
    if (value > 0) {
        return `${baseClassName} bg-success/10 text-success`;
    }
    if (value < 0) {
        return `${baseClassName} bg-error/10 text-error`;
    }
    return `${baseClassName} bg-base-200/70 text-base-content/70`;
};

const getTradedCellStyle = (value: number | null, maxValue: number): React.CSSProperties | undefined => {
    if (value === null || maxValue <= 0) {
        return undefined;
    }
    const intensity = 0.06 + Math.min(value / maxValue, 1) * 0.16;
    return { backgroundColor: `rgba(100, 116, 139, ${intensity})` };
};

export const SectorRotationHeatmap: React.FC<SectorRotationHeatmapProps> = ({
    stocks,
    flowResponse,
    dateRange,
}) => {
    const [useToday, setUseToday] = useState(true);

    const rows = useMemo(() => {
        if (useToday) {
            return buildTodayRows(stocks);
        }
        return buildHistoricalRows(stocks, flowResponse);
    }, [flowResponse, stocks, useToday]);

    const maxTradedValue = useMemo(
        () => rows.reduce((maxValue, row) => Math.max(maxValue, row.tradedValue ?? 0), 0),
        [rows],
    );

    const totalTradedValue = useMemo(
        () => rows.reduce((total, row) => total + (row.tradedValue ?? 0), 0),
        [rows],
    );
    const foreignNetTotal = useMemo(
        () => rows.reduce((total, row) => total + (row.foreignNetFlow ?? 0), 0),
        [rows],
    );
    const propNetTotal = useMemo(() => {
        if (useToday) {
            return null;
        }
        return rows.reduce((total, row) => total + (row.propNetFlow ?? 0), 0);
    }, [rows, useToday]);

    const modeLabel = useToday ? 'current session' : `${dateRange.startDate} to ${dateRange.endDate}`;

    return (
        <section className="space-y-3">
            <div className="flex flex-col gap-3 border-b border-base-300 pb-3 lg:flex-row lg:items-start lg:justify-between">
                <div>
                    <div className="text-sm font-semibold text-base-content">Sector rotation heatmap</div>
                    <div className="mt-1 text-xs text-base-content/60">
                        Industry-level traded value and net flow, {modeLabel}
                    </div>
                    <div className="mt-2 flex flex-wrap items-center gap-3 text-xs text-base-content/70">
                        <span>
                            Traded:{' '}
                            <span className="font-medium text-base-content">{formatValueBilVnd(roundToTwo(totalTradedValue))}</span>
                        </span>
                        <span>
                            Foreign net:{' '}
                            <span className={foreignNetTotal >= 0 ? 'font-medium text-success' : 'font-medium text-error'}>
                                {formatSignedBilVnd(roundToTwo(foreignNetTotal))}
                            </span>
                        </span>
                        <span>
                            Proprietary net:{' '}
                            <span className={propNetTotal === null ? 'font-medium text-base-content/50' : propNetTotal >= 0 ? 'font-medium text-success' : 'font-medium text-error'}>
                                {propNetTotal !== null ? formatSignedBilVnd(roundToTwo(propNetTotal)) : 'N/A'}
                            </span>
                        </span>
                    </div>
                </div>

                <label
                    className="flex items-center gap-2 text-xs text-base-content/70"
                    title="Use current trading session traded value and foreign flow data"
                >
                    <span>Today</span>
                    <input
                        type="checkbox"
                        className="toggle toggle-sm"
                        checked={useToday}
                        onChange={(event) => setUseToday(event.target.checked)}
                    />
                </label>
            </div>

            {rows.length === 0 ? (
                <div className="flex h-44 items-center justify-center text-sm text-base-content/50">
                    No sector rotation data available for this mode.
                </div>
            ) : (
                <div className="w-full overflow-x-auto">
                    <table className="table table-sm min-w-[760px]">
                        <thead>
                            <tr className="text-xs uppercase text-base-content/55">
                                <th className="w-[34%]">Sector / Industry</th>
                                <th className="text-right">Total traded value</th>
                                <th className="text-right">Net foreign flow</th>
                                <th className="text-right">Net proprietary flow</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows.map((row) => (
                                <tr key={row.industry} className="border-base-300">
                                    <td>
                                        <div className="font-semibold text-base-content">{row.industry}</div>
                                        <div className="mt-1 text-xs text-base-content/50">
                                            {row.stockCount} {row.stockCount === 1 ? 'stock' : 'stocks'}
                                        </div>
                                    </td>
                                    <td>
                                        <div
                                            className="rounded px-3 py-2 text-right font-semibold text-base-content"
                                            style={getTradedCellStyle(row.tradedValue, maxTradedValue)}
                                        >
                                            {formatValueBilVnd(row.tradedValue)}
                                        </div>
                                    </td>
                                    <td>
                                        <div className={getFlowCellClassName(row.foreignNetFlow)}>
                                            {formatSignedBilVnd(row.foreignNetFlow)}
                                        </div>
                                    </td>
                                    <td>
                                        <div className={getFlowCellClassName(row.propNetFlow)}>
                                            {formatSignedBilVnd(row.propNetFlow)}
                                        </div>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </section>
    );
};

export default SectorRotationHeatmap;
