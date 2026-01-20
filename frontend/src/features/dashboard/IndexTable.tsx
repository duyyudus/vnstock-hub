import React, { useEffect, useState } from 'react';
import { stockApi } from '../../api/stockApi';
import type { Stock } from '../../api/stockApi';
import type { IndustryInfo } from '../../api/stockApi';
import { IndexSelector } from './IndexSelector';
import { IndustrySelector } from './IndustrySelector';
import type { IndexConfig } from './indexConfig';

interface IndexTableProps {
    /** List of available indices */
    indices: IndexConfig[];
}

/**
 * Generic index stocks table component.
 * Displays stocks for the selected index with price, market cap, and additional metrics.
 */
export const IndexTable: React.FC<IndexTableProps> = ({
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
                            <th className="text-base-content font-bold">Ticker</th>
                            <th className="text-base-content font-bold text-right">
                                Price (VND)
                            </th>
                            <th className="text-base-content font-bold text-right">
                                Market Cap (B VND)
                            </th>
                            <th className="text-base-content font-bold text-right">
                                Charter Cap (B VND)
                            </th>
                            <th className="text-base-content font-bold text-right">
                                P/E
                            </th>
                            <th className="text-base-content font-bold text-right">
                                24h
                            </th>
                            <th className="text-base-content font-bold text-right">
                                1W
                            </th>
                            <th className="text-base-content font-bold text-right">
                                1M
                            </th>
                            <th className="text-base-content font-bold text-right">
                                1Y
                            </th>
                        </tr>
                    </thead>
                    <tbody>
                        {stocks.map((stock, index) => {
                            const change24h = formatPriceChange(stock.price_change_24h);
                            const change1w = formatPriceChange(stock.price_change_1w);
                            const change1m = formatPriceChange(stock.price_change_1m);
                            const change1y = formatPriceChange(stock.price_change_1y);

                            return (
                                <tr key={stock.ticker} className="hover">
                                    <td className="text-base-content/60">{index + 1}</td>
                                    <td>
                                        <div className="tooltip tooltip-right" data-tip={stock.company_name}>
                                            <span className="font-bold text-base-content uppercase cursor-help">
                                                {stock.ticker.slice(0, 3)}
                                            </span>
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
                        })}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default IndexTable;
