import React, { useEffect, useState, useMemo } from 'react';
import { stockApi } from '../../../api/stockApi';
import type { Stock } from '../../../api/stockApi';
import type { IndustryInfo } from '../../../api/stockApi';
import { IndexSelector } from './IndexSelector';
import { IndustrySelector } from './IndustrySelector';
import type { IndexConfig } from './indexConfig';

interface StocksTableProps {
    /** List of available indices */
    indices: IndexConfig[];
}

/**
 * Generic stocks table component.
 * Displays stocks for the selected index or industry with price, market cap, and additional metrics.
 */

type SortKey = keyof Stock;
type SortDirection = 'asc' | 'desc';

interface SortConfig {
    key: SortKey;
    direction: SortDirection;
}

export const StocksTable: React.FC<StocksTableProps> = ({
    indices
}) => {
    // Default to VN30 if available, otherwise first index
    const [selectedIndex, setSelectedIndex] = useState<IndexConfig | null>(() => {
        if (indices.length === 0) return null;
        return indices.find(idx => idx.id === 'VN30') || indices[0];
    });
    const [stocks, setStocks] = useState<Stock[]>([]);
    const [industries, setIndustries] = useState<IndustryInfo[]>([]);
    const [selectedIndustryName, setSelectedIndustryName] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [sortConfig, setSortConfig] = useState<SortConfig>({
        key: 'market_cap',
        direction: 'desc'
    });
    const [searchQuery, setSearchQuery] = useState('');
    const [isCompanyCollapsed, setIsCompanyCollapsed] = useState(true);


    // Update selected index if indices prop changes and we don't have a selection yet
    useEffect(() => {
        if (indices.length > 0 && !selectedIndex) {
            const defaultIndex = indices.find(idx => idx.id === 'VN30') || indices[0];
            setSelectedIndex(defaultIndex);
        }
    }, [indices, selectedIndex]);

    // Fetch industries on mount
    useEffect(() => {
        const fetchIndustries = async () => {
            try {
                const response = await stockApi.getIndustries();
                setIndustries(response.industries);
            } catch (err) {
                console.error('Failed to fetch industries:', err);
            }
        };
        fetchIndustries();
    }, []);

    useEffect(() => {
        const fetchData = async () => {
            try {
                setLoading(true);
                setError(null);

                if (selectedIndustryName) {
                    const response = await stockApi.getIndustryStocks(selectedIndustryName);
                    setStocks(response.stocks);
                } else if (selectedIndex) {
                    const response = await stockApi.getIndexStocks(selectedIndex.apiEndpoint);
                    setStocks(response.stocks);
                }
            } catch (err) {
                const label = selectedIndustryName || (selectedIndex ? selectedIndex.label : 'stocks');
                setError(`Failed to fetch ${label} stocks data. Please try again.`);
                console.error(`Error fetching ${label} data:`, err);
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, [selectedIndex, selectedIndustryName]);

    const handleIndexChange = (newIndex: IndexConfig) => {
        setSelectedIndex(newIndex);
        // Clear industry filter when changing index explicitly? 
        // Or keep them separate. User asked for "Place this drop-down next to existing index selector".
        // Let's clear industry when explicit index is chosen, and vice versa?
        // Actually, if I select an industry, it should filter the current view.
        // If I then select another index, I probably want to see that index.
        setSelectedIndustryName(null);
    };

    const handleIndustryChange = (industryName: string | null) => {
        setSelectedIndustryName(industryName);
    };

    // Format price to VND
    const formatPrice = (price: number): string => {
        return new Intl.NumberFormat('en-US').format(price);
    };

    // Format market cap (in billion VND)
    const formatMarketCap = (marketCap: number): string => {
        return new Intl.NumberFormat('en-US', {
            minimumFractionDigits: 0,
            maximumFractionDigits: 0,
        }).format(marketCap);
    };

    // Format charter capital (in billion VND)
    const formatCharterCapital = (charterCapital: number): string => {
        return new Intl.NumberFormat('en-US', {
            minimumFractionDigits: 0,
            maximumFractionDigits: 0,
        }).format(charterCapital);
    };

    // Format P/E ratio
    const formatPE = (pe: number | null): string => {
        if (pe === null) return '-';
        return new Intl.NumberFormat('en-US', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
        }).format(pe);
    };

    // Format accumulated value (in billion VND)
    const formatAccumulatedValue = (value: number | null): string => {
        if (value === null) return '-';
        return new Intl.NumberFormat('en-US', {
            minimumFractionDigits: 0,
            maximumFractionDigits: 0,
        }).format(value);
    };

    // Format price change percentage with color
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
            // Toggle back to asc or keep desc? Usually it's asc -> desc -> asc
            direction = 'asc';
        } else {
            // New key, default to desc for numeric, asc for strings? 
            // Most financial tables default to 'desc' for value-based columns.
            direction = 'desc';
        }
        setSortConfig({ key, direction });
    };

    const filteredStocks = useMemo(() => {
        if (!searchQuery.trim()) return stocks;
        const query = searchQuery.toLowerCase().trim();
        return stocks.filter(stock =>
            stock.ticker.toLowerCase().includes(query)
        );
    }, [stocks, searchQuery]);

    const sortedStocks = useMemo(() => {
        const sortableStocks = [...filteredStocks];
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
    }, [filteredStocks, sortConfig]);

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


    if (!selectedIndex) {
        return <div>No indices available.</div>;
    }

    if (loading) {
        return (
            <div className="flex flex-col gap-4">
                {/* Header with selector */}
                <div className="flex items-center justify-between">
                    <div>
                        <h2 className="text-2xl font-bold text-base-content">
                            {selectedIndustryName || selectedIndex.title}
                        </h2>
                        <p className="text-base-content/60 text-sm">
                            {selectedIndustryName ? `Top stocks in ${selectedIndustryName}` : selectedIndex.description}
                        </p>
                    </div>
                    <div className="flex gap-2">
                        <div className="relative">
                            <input
                                type="text"
                                placeholder="Search tickers..."
                                className="input input-sm input-bordered w-32 md:w-64 pl-8"
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                            />
                            <svg
                                className="w-4 h-4 absolute left-2.5 top-1/2 -translate-y-1/2 text-base-content/40"
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                            >
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                            </svg>
                        </div>
                        <IndustrySelector
                            industries={industries}
                            selectedIndustryName={selectedIndustryName}
                            onIndustryChange={handleIndustryChange}
                        />
                        <IndexSelector
                            indices={indices}
                            selectedIndex={selectedIndex}
                            onIndexChange={handleIndexChange}
                        />
                    </div>
                </div>
                {/* Loading spinner */}
                <div className="flex flex-col items-center justify-center h-64 gap-4">
                    <span className="loading loading-spinner loading-lg text-primary"></span>
                    <p className="text-base-content/70">Loading {selectedIndustryName || selectedIndex.label} stocks...</p>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="flex flex-col gap-4">
                {/* Header with selector */}
                <div className="flex items-center justify-between">
                    <div>
                        <h2 className="text-2xl font-bold text-base-content">
                            {selectedIndustryName || selectedIndex.title}
                        </h2>
                        <p className="text-base-content/60 text-sm">
                            {selectedIndustryName ? `Top stocks in ${selectedIndustryName}` : selectedIndex.description}
                        </p>
                    </div>
                    <div className="flex gap-2">
                        <div className="relative">
                            <input
                                type="text"
                                placeholder="Search tickers..."
                                className="input input-sm input-bordered w-32 md:w-64 pl-8"
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                            />
                            <svg
                                className="w-4 h-4 absolute left-2.5 top-1/2 -translate-y-1/2 text-base-content/40"
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                            >
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                            </svg>
                        </div>
                        <IndustrySelector
                            industries={industries}
                            selectedIndustryName={selectedIndustryName}
                            onIndustryChange={handleIndustryChange}
                        />
                        <IndexSelector
                            indices={indices}
                            selectedIndex={selectedIndex}
                            onIndexChange={handleIndexChange}
                        />
                    </div>
                </div>
                {/* Error alert */}
                <div className="alert alert-error">
                    <svg
                        xmlns="http://www.w3.org/2000/svg"
                        className="stroke-current shrink-0 h-6 w-6"
                        fill="none"
                        viewBox="0 0 24 24"
                    >
                        <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth="1"
                            d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"
                        />
                    </svg>
                    <span>{error}</span>
                    <button
                        className="btn btn-sm btn-ghost"
                        onClick={() => window.location.reload()}
                    >
                        Retry
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="flex flex-col gap-4">
            {/* Header with selector */}
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-2xl font-bold text-base-content">
                        {selectedIndustryName || selectedIndex.title}
                    </h2>
                    <p className="text-base-content/60 text-sm">
                        {selectedIndustryName ? `Top stocks in ${selectedIndustryName}` : selectedIndex.description}
                    </p>
                </div>
                <div className="flex gap-2">
                    <div className="relative">
                        <input
                            type="text"
                            placeholder="Search tickers..."
                            className="input input-sm input-bordered w-32 md:w-64 pl-8"
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                        />
                        <svg
                            className="w-4 h-4 absolute left-2.5 top-1/2 -translate-y-1/2 text-base-content/40"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                        >
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                        </svg>
                    </div>
                    <IndustrySelector
                        industries={industries}
                        selectedIndustryName={selectedIndustryName}
                        onIndustryChange={handleIndustryChange}
                    />
                    <IndexSelector
                        indices={indices}
                        selectedIndex={selectedIndex}
                        onIndexChange={handleIndexChange}
                    />
                </div>
            </div>

            {/* Table */}
            <div className="overflow-x-auto rounded-xl border border-base-300 bg-base-100">
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
                                    No stocks found matching "{searchQuery}"
                                </td>
                            </tr>
                        ) : (
                            sortedStocks.map((stock, index) => {
                                const change24h = formatPriceChange(stock.price_change_24h);
                                const change1w = formatPriceChange(stock.price_change_1w);
                                const change1m = formatPriceChange(stock.price_change_1m);
                                const change1y = formatPriceChange(stock.price_change_1y);

                                return (
                                    <tr key={stock.ticker} className="hover">
                                        <td className="text-base-content/60">{index + 1}</td>
                                        <td
                                            className={`${isCompanyCollapsed ? 'w-0 p-0 overflow-hidden opacity-0' : 'whitespace-nowrap'} transition-all duration-200`}
                                            title={isCompanyCollapsed ? "" : stock.company_name}
                                        >
                                            {!isCompanyCollapsed && stock.company_name}
                                        </td>
                                        <td>
                                            <div className="tooltip tooltip-right" data-tip={stock.company_name}>
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
        </div>
    );
};

export default StocksTable;
