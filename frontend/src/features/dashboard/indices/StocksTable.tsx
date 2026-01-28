import React, { useState, useMemo } from 'react';
import type { Stock } from '../../../api/stockApi';

interface StocksTableProps {
    /** List of stocks to display */
    stocks: Stock[];
}

type SortKey = keyof Stock;
type SortDirection = 'asc' | 'desc';

interface SortConfig {
    key: SortKey;
    direction: SortDirection;
}

export const StocksTable: React.FC<StocksTableProps> = ({ stocks }) => {
    const [sortConfig, setSortConfig] = useState<SortConfig>({
        key: 'market_cap',
        direction: 'desc'
    });
    const [isCompanyCollapsed, setIsCompanyCollapsed] = useState(true);

    // Formatters
    const formatPrice = (price: number): string => {
        return new Intl.NumberFormat('en-US').format(price);
    };

    const formatMarketCap = (marketCap: number): string => {
        return new Intl.NumberFormat('en-US', {
            minimumFractionDigits: 0,
            maximumFractionDigits: 0,
        }).format(marketCap);
    };

    const formatCharterCapital = (charterCapital: number): string => {
        return new Intl.NumberFormat('en-US', {
            minimumFractionDigits: 0,
            maximumFractionDigits: 0,
        }).format(charterCapital);
    };

    const formatPE = (pe: number | null): string => {
        if (pe === null) return '-';
        return new Intl.NumberFormat('en-US', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
        }).format(pe);
    };

    const formatAccumulatedValue = (value: number | null): string => {
        if (value === null) return '-';
        return new Intl.NumberFormat('en-US', {
            minimumFractionDigits: 0,
            maximumFractionDigits: 0,
        }).format(value);
    };

    const formatPriceChange = (change: number | null): { text: string; className: string } => {
        if (change === null) return { text: '-', className: 'text-base-content/50' };
        const prefix = change > 0 ? '+' : '';
        const formattedValue = new Intl.NumberFormat('en-US', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
        }).format(change);
        const text = `${prefix}${formattedValue}%`;
        const className = change > 0 ? 'text-success' : change < 0 ? 'text-error' : 'text-base-content';
        return { text, className };
    };

    const handleSort = (key: SortKey) => {
        let direction: SortDirection = 'asc';
        if (sortConfig.key === key && sortConfig.direction === 'asc') {
            direction = 'desc';
        } else if (sortConfig.key === key && sortConfig.direction === 'desc') {
            direction = 'asc';
        } else {
            direction = 'desc';
        }
        setSortConfig({ key, direction });
    };

    const sortedStocks = useMemo(() => {
        const sortableStocks = [...stocks];
        if (sortConfig.key) {
            sortableStocks.sort((a, b) => {
                const aValue = a[sortConfig.key];
                const bValue = b[sortConfig.key];

                if (aValue === null) return 1;
                if (bValue === null) return -1;

                if (aValue < bValue) {
                    return sortConfig.direction === 'asc' ? -1 : 1;
                }
                if (aValue > bValue) {
                    return sortConfig.direction === 'asc' ? 1 : -1;
                }
                return 0;
            });
        }
        return sortableStocks;
    }, [stocks, sortConfig]);

    const renderSortIcon = (key: SortKey) => {
        if (sortConfig.key !== key) {
            return (
                <svg className="w-3 h-3 ml-1 opacity-20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4" />
                </svg>
            );
        }
        return sortConfig.direction === 'asc' ? (
            <svg className="w-3 h-3 ml-1 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 15l7-7 7 7" />
            </svg>
        ) : (
            <svg className="w-3 h-3 ml-1 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
            </svg>
        );
    };

    return (
        <div className="overflow-x-auto rounded-xl">
            <table className="table table-zebra table-sm">
                <thead className="bg-base-200">
                    <tr>
                        <th className="text-base-content font-bold">#</th>
                        <th
                            className="text-base-content font-bold cursor-pointer hover:bg-base-300 transition-colors"
                            onClick={() => setIsCompanyCollapsed(!isCompanyCollapsed)}
                        >
                            <div className="flex items-center">
                                {isCompanyCollapsed ? (
                                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7" />
                                    </svg>
                                ) : (
                                    <>
                                        <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
                                        </svg>
                                        Company
                                    </>
                                )}
                            </div>
                        </th>
                        <th
                            className="text-base-content font-bold cursor-pointer hover:bg-base-300 transition-colors"
                            onClick={() => handleSort('ticker')}
                        >
                            <div className="flex items-center">
                                Ticker
                                {renderSortIcon('ticker')}
                            </div>
                        </th>
                        <th
                            className="text-base-content font-bold text-right cursor-pointer hover:bg-base-300 transition-colors"
                            onClick={() => handleSort('price')}
                        >
                            <div className="flex items-center justify-end">
                                <div className="text-right">
                                    Price<br />(VND)
                                </div>
                                {renderSortIcon('price')}
                            </div>
                        </th>
                        <th
                            className="text-base-content font-bold text-right cursor-pointer hover:bg-base-300 transition-colors"
                            onClick={() => handleSort('market_cap')}
                        >
                            <div className="flex items-center justify-end">
                                <div className="text-right">
                                    Market Cap<br />(B VND)
                                </div>
                                {renderSortIcon('market_cap')}
                            </div>
                        </th>
                        <th
                            className="text-base-content font-bold text-right cursor-pointer hover:bg-base-300 transition-colors"
                            onClick={() => handleSort('charter_capital')}
                        >
                            <div className="flex items-center justify-end">
                                <div className="text-right">
                                    Charter Cap<br />(B VND)
                                </div>
                                {renderSortIcon('charter_capital')}
                            </div>
                        </th>
                        <th
                            className="text-base-content font-bold text-right cursor-pointer hover:bg-base-300 transition-colors"
                            onClick={() => handleSort('pe_ratio')}
                        >
                            <div className="flex items-center justify-end">
                                P/E
                                {renderSortIcon('pe_ratio')}
                            </div>
                        </th>
                        <th
                            className="text-base-content font-bold text-right cursor-pointer hover:bg-base-300 transition-colors"
                            onClick={() => handleSort('accumulated_value')}
                        >
                            <div className="flex items-center justify-end">
                                <div className="text-right">
                                    Vol<br />(B VND)
                                </div>
                                {renderSortIcon('accumulated_value')}
                            </div>
                        </th>
                        <th
                            className="text-base-content font-bold text-right cursor-pointer hover:bg-base-300 transition-colors"
                            onClick={() => handleSort('price_change_24h')}
                        >
                            <div className="flex items-center justify-end">
                                24h
                                {renderSortIcon('price_change_24h')}
                            </div>
                        </th>
                        <th
                            className="text-base-content font-bold text-right cursor-pointer hover:bg-base-300 transition-colors"
                            onClick={() => handleSort('price_change_1w')}
                        >
                            <div className="flex items-center justify-end">
                                1W
                                {renderSortIcon('price_change_1w')}
                            </div>
                        </th>
                        <th
                            className="text-base-content font-bold text-right cursor-pointer hover:bg-base-300 transition-colors"
                            onClick={() => handleSort('price_change_1m')}
                        >
                            <div className="flex items-center justify-end">
                                1M
                                {renderSortIcon('price_change_1m')}
                            </div>
                        </th>
                        <th
                            className="text-base-content font-bold text-right cursor-pointer hover:bg-base-300 transition-colors"
                            onClick={() => handleSort('price_change_1y')}
                        >
                            <div className="flex items-center justify-end">
                                1Y
                                {renderSortIcon('price_change_1y')}
                            </div>
                        </th>
                    </tr>
                </thead>

                <tbody>
                    {sortedStocks.length === 0 ? (
                        <tr>
                            <td colSpan={12} className="text-center py-8 text-base-content/60 italic">
                                No stocks found
                            </td>
                        </tr>
                    ) : (
                        sortedStocks.map((stock, index) => {
                            const change24h = formatPriceChange(stock.price_change_24h);
                            const change1w = formatPriceChange(stock.price_change_1w);
                            const change1m = formatPriceChange(stock.price_change_1m);
                            const change1y = formatPriceChange(stock.price_change_1y);
                            const fullNameWithExchange = stock.exchange 
                                ? `${stock.exchange} - ${stock.company_name}`
                                : stock.company_name;

                            return (
                                <tr key={stock.ticker} className="hover">
                                    <td className="text-base-content/60">{index + 1}</td>
                                    <td
                                        className={`${isCompanyCollapsed ? 'w-0 p-0 overflow-hidden opacity-0' : 'whitespace-nowrap'} transition-all duration-200`}
                                        title={isCompanyCollapsed ? "" : fullNameWithExchange}
                                    >
                                        {!isCompanyCollapsed && stock.company_name}
                                    </td>
                                    <td>
                                        <div className="tooltip tooltip-right" data-tip={fullNameWithExchange}>
                                            <button
                                                className="font-bold text-primary uppercase cursor-pointer hover:underline focus:outline-none"
                                                onClick={() => (window as any).onTickerClick?.(stock.ticker, stock.company_name)}
                                                title={`View financial details for ${stock.ticker}`}
                                            >
                                                {stock.ticker.slice(0, 3)}
                                            </button>
                                        </div>
                                    </td>
                                    <td className="text-right font-mono text-base-content">
                                        {formatPrice(stock.price)}
                                    </td>
                                    <td className="text-right font-mono text-base-content">
                                        {formatMarketCap(stock.market_cap)}
                                    </td>
                                    <td className="text-right font-mono text-base-content">
                                        {formatCharterCapital(stock.charter_capital)}
                                    </td>
                                    <td className="text-right font-mono text-base-content">
                                        {formatPE(stock.pe_ratio)}
                                    </td>
                                    <td className="text-right font-mono text-base-content">
                                        <button
                                            className="cursor-pointer hover:text-primary hover:underline focus:outline-none"
                                            onClick={() => (window as any).onVolumeClick?.(stock.ticker, stock.company_name)}
                                            title={`View 30-day volume chart for ${stock.ticker}`}
                                        >
                                            {formatAccumulatedValue(stock.accumulated_value)}
                                        </button>
                                    </td>
                                    <td className={`text-right font-mono ${change24h.className}`}>
                                        {change24h.text}
                                    </td>
                                    <td className={`text-right font-mono ${change1w.className}`}>
                                        {change1w.text}
                                    </td>
                                    <td className={`text-right font-mono ${change1m.className}`}>
                                        {change1m.text}
                                    </td>
                                    <td className={`text-right font-mono ${change1y.className}`}>
                                        {change1y.text}
                                    </td>
                                </tr>
                            );
                        })
                    )}
                </tbody>
            </table>
        </div>
    );
};

export default StocksTable;