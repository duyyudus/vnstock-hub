import React from 'react';
import type { FundOverviewResponse, FundOverviewStock } from '../../../api/stockApi';

interface FundOverviewProps {
    data: FundOverviewResponse | null;
    loading?: boolean;
    error?: string | null;
}

const formatPercent = (value: number) => `${value.toFixed(1)}%`;

const formatDate = (value: string | null) => {
    if (!value) return 'N/A';
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) {
        return value;
    }
    return parsed.toLocaleDateString();
};

const FundRow: React.FC<{ stock: FundOverviewStock }> = ({ stock }) => {
    const topFunds = stock.funds.slice(0, 8);

    return (
        <tr>
            <td className="w-32 align-top">
                <span className="font-semibold text-base-content">{stock.ticker}</span>
            </td>
            <td className="min-w-0 align-top">
                {stock.company_name ? (
                    <span className="block max-w-md truncate text-sm text-base-content/70" title={stock.company_name}>
                        {stock.company_name}
                    </span>
                ) : (
                    <span className="text-base-content/40">-</span>
                )}
            </td>
            <td className="w-28 whitespace-nowrap text-right align-top font-semibold text-primary">
                {formatPercent(stock.total_allocation)}
            </td>
            <td className="w-24 whitespace-nowrap text-right align-top text-base-content/70">
                {stock.fund_count}
            </td>
            <td className="min-w-[18rem] align-top">
                <div className="flex flex-wrap justify-end gap-1.5">
                    {topFunds.map((fund) => (
                        <span
                            key={`${stock.ticker}-${fund.symbol}`}
                            className="tooltip tooltip-left rounded-full border border-base-content/30 px-2 py-0.5 text-xs text-base-content/80"
                            data-tip={fund.name || fund.symbol}
                        >
                            {fund.symbol} {formatPercent(fund.allocation)}
                        </span>
                    ))}

                    {stock.funds.length > topFunds.length && (
                        <span className="rounded-full bg-base-200 px-2 py-0.5 text-xs text-base-content/70">
                            +{stock.funds.length - topFunds.length}
                        </span>
                    )}
                </div>
            </td>
        </tr>
    );
};

const SectorRows: React.FC<{
    sector: FundOverviewResponse['sectors'][number];
}> = ({ sector }) => {
    return (
        <React.Fragment>
            <tr className="bg-base-200/70">
                <td colSpan={5} className="py-3">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                        <div>
                            <span className="font-semibold text-base-content">{sector.sector}</span>
                            <span className="ml-2 text-xs text-base-content/50">
                                {sector.stock_count} {sector.stock_count === 1 ? 'stock' : 'stocks'}
                            </span>
                        </div>
                        <div className="text-sm">
                            <span className="font-semibold text-primary">{formatPercent(sector.total_allocation)}</span>
                            <span className="ml-1 text-xs text-base-content/50">sector total</span>
                        </div>
                    </div>
                </td>
            </tr>
            {sector.stocks.map((stock) => (
                <FundRow key={`${sector.sector}-${stock.ticker}`} stock={stock} />
            ))}
        </React.Fragment>
    );
};

const FundOverviewTable: React.FC<{ data: FundOverviewResponse }> = ({ data }) => {
    return (
        <div className="overflow-x-auto overflow-y-hidden rounded-lg border border-base-300 bg-base-100 shadow-md">
            <table className="table table-sm">
                <thead>
                    <tr>
                        <th className="w-32">Ticker</th>
                        <th>Company</th>
                        <th className="w-28 text-right">Allocation</th>
                        <th className="w-24 text-right">Funds</th>
                        <th className="min-w-[18rem] text-right">Top funds</th>
                    </tr>
                </thead>
                <tbody>
                    {data.sectors.map((sector) => (
                        <SectorRows key={sector.sector} sector={sector} />
                    ))}
                </tbody>
            </table>
        </div>
    );
};

export const FundOverview: React.FC<FundOverviewProps> = ({ data, loading = false, error = null }) => {
    return (
        <div className="w-full space-y-4">
            <div className="flex items-center justify-between border-b border-base-300 pb-2">
                <h2 className="text-xl font-bold">Overview</h2>
            </div>

            {loading && !data ? (
                <div className="flex h-64 flex-col items-center justify-center rounded-lg border border-base-300 bg-base-100 shadow-md">
                    <span className="loading loading-spinner loading-lg text-primary"></span>
                    <p className="mt-4 text-base-content/70">Loading fund holdings overview...</p>
                </div>
            ) : error ? (
                <div className="alert alert-error shadow-lg">
                    <span>{error}</span>
                </div>
            ) : !data || data.sectors.length === 0 ? (
                <div className="rounded-lg border border-base-300 bg-base-100 p-6 text-center text-base-content/60 shadow-md">
                    No cached stock holding data available
                </div>
            ) : (
                <>
                    <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                        <div className="rounded-lg border border-base-300 bg-base-100 p-4 shadow-md">
                            <div className="text-xs uppercase text-base-content/50">Funds with holdings</div>
                            <div className="mt-1 text-2xl font-bold">{data.fund_count}</div>
                        </div>
                        <div className="rounded-lg border border-base-300 bg-base-100 p-4 shadow-md">
                            <div className="text-xs uppercase text-base-content/50">Stocks included</div>
                            <div className="mt-1 text-2xl font-bold">{data.stock_count}</div>
                        </div>
                        <div className="rounded-lg border border-base-300 bg-base-100 p-4 shadow-md">
                            <div className="text-xs uppercase text-base-content/50">Latest holding date</div>
                            <div className="mt-1 text-2xl font-bold">{formatDate(data.last_updated)}</div>
                        </div>
                    </div>

                    <FundOverviewTable data={data} />
                </>
            )}
        </div>
    );
};

export default FundOverview;
