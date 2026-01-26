import React, { useEffect, useState, useMemo } from 'react';
import { stockApi } from '../../../api/stockApi';
import type { Stock, IndustryInfo } from '../../../api/stockApi';
import { IndexSelector } from './IndexSelector';
import { IndustrySelector } from './IndustrySelector';
import { StocksGrowthChart } from './StocksGrowthChart';
import { StocksTable } from './StocksTable';
import type { IndexConfig } from './indexConfig';

interface IndicesTabProps {
    /** List of available indices */
    indices: IndexConfig[];
}

type ViewMode = 'table' | 'growth';

/**
 * Indices Tab - Main container for Index/Industry stock views.
 * Manages state for selection, fetching, and view switching.
 */
export const IndicesTab: React.FC<IndicesTabProps> = ({ indices }) => {
    // --- Selection State ---
    // Default to VN30 if available, otherwise first index
    const [selectedIndex, setSelectedIndex] = useState<IndexConfig | null>(() => {
        if (indices.length === 0) return null;
        return indices.find(idx => idx.id === 'VN30') || indices[0];
    });
    const [selectedIndustryName, setSelectedIndustryName] = useState<string | null>(null);

    // --- Data State ---
    const [stocks, setStocks] = useState<Stock[]>([]);
    const [industries, setIndustries] = useState<IndustryInfo[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // --- View State ---
    const [viewMode, setViewMode] = useState<ViewMode>('table');
    const [searchQuery, setSearchQuery] = useState('');

    // --- Effects ---

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

    // Fetch Stocks Data
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
            } catch (err: any) {
                const label = selectedIndustryName || (selectedIndex ? selectedIndex.label : 'stocks');

                // If it's a rate limit error (429) or if we can check global status
                try {
                    const syncStatus = await stockApi.getSyncStatus();
                    if (syncStatus.is_rate_limited) {
                        setError(`Market data source is currently busy. Retrying automatically...`);

                        // Set up a one-time retry after a delay
                        setTimeout(() => fetchData(), 30000);
                        return;
                    }
                } catch (e) {
                    // Ignore sync status fetch error
                }

                setError(`Failed to fetch ${label} stocks data. Please try again.`);
                console.error(`Error fetching ${label} data:`, err);
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, [selectedIndex, selectedIndustryName]);

    // --- Handlers ---

    const handleIndexChange = (newIndex: IndexConfig) => {
        setSelectedIndex(newIndex);
        setSelectedIndustryName(null); // Clear industry when index selected
    };

    const handleIndustryChange = (industryName: string | null) => {
        setSelectedIndustryName(industryName);
    };

    // --- Filtering ---

    const filteredStocks = useMemo(() => {
        if (!searchQuery.trim()) return stocks;
        const query = searchQuery.toLowerCase().trim();
        return stocks.filter(stock =>
            stock.ticker.toLowerCase().includes(query)
        );
    }, [stocks, searchQuery]);

    // --- Render ---

    if (!selectedIndex) {
        return <div>No indices available.</div>;
    }

    return (
        <div className="space-y-6 p-4">
            {/* Header Section */}
            <div className="flex flex-col gap-4">
                <div className="flex items-center justify-between flex-wrap gap-2">
                    <div>
                        <h2 className="text-2xl font-bold text-base-content">
                            {selectedIndustryName || selectedIndex.title}
                        </h2>
                    </div>

                    {/* Toolbar */}
                    <div className="flex flex-wrap gap-2 items-center">
                        {/* View Mode Toggle */}
                        <div className="join">
                            <button
                                className={`join-item btn btn-sm ${viewMode === 'table' ? 'btn-primary' : 'btn-ghost'}`}
                                onClick={() => setViewMode('table')}
                            >
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                                </svg>
                                Price Table
                            </button>
                            <button
                                className={`join-item btn btn-sm ${viewMode === 'growth' ? 'btn-primary' : 'btn-ghost'}`}
                                onClick={() => setViewMode('growth')}
                            >
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
                                </svg>
                                Growth Chart
                            </button>
                        </div>

                        {/* Search & Selectors */}
                        <div className="relative">
                            <input
                                type="text"
                                placeholder="Search tickers..."
                                className="input input-sm input-bordered w-32 md:w-48 pl-8"
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
            </div>

            {/* Content Section */}
            {loading ? (
                <div className="flex flex-col items-center justify-center h-64 gap-4 card bg-base-100 shadow-md border border-base-300">
                    <span className="loading loading-spinner loading-lg text-primary"></span>
                    <p className="text-base-content/70">Loading {selectedIndustryName || selectedIndex.label} stocks...</p>
                </div>
            ) : error ? (
                <div className="alert alert-error shadow-lg">
                    <svg xmlns="http://www.w3.org/2000/svg" className="stroke-current shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <span>{error}</span>
                    <button className="btn btn-sm btn-ghost" onClick={() => window.location.reload()}>Retry</button>
                </div>
            ) : (
                <div className="card bg-base-100 shadow-md border border-base-300">
                    <div className="card-body p-4">
                        {viewMode === 'growth' ? (
                            <StocksGrowthChart
                                stocks={filteredStocks}
                            />
                        ) : (
                            <StocksTable stocks={filteredStocks} />
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};

export default IndicesTab;
