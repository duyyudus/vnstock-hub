import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { stockApi } from '../../../api/stockApi';
import type { Stock, IndustryInfo, BookmarkGroup } from '../../../api/stockApi';
import { IndexSelector } from '../indices/IndexSelector';
import { IndustrySelector } from '../indices/IndustrySelector';
import { BookmarkSelector } from '../indices/BookmarkSelector';
import type { IndexConfig } from '../indices/indexConfig';
import { deriveIndexIndustryScope } from '../indices/indexIndustryScope';
import { useAuthUser } from '../../auth/useAuthUser';
import { FinancialHealthScreener } from './FinancialHealthScreener';
import { ValuationScreener } from './ValuationScreener';
import { MarginTrendScreener } from './MarginTrendScreener';

interface ScreenersTabProps {
    indices: IndexConfig[];
}

export const ScreenersTab: React.FC<ScreenersTabProps> = ({ indices }) => {
    const user = useAuthUser();
    const [activeScreener, setActiveScreener] = useState<'valuation' | 'margin-trend' | 'financial-health'>('valuation');

    const [selectedIndexId, setSelectedIndexId] = useState<string | null>(() => {
        if (indices.length === 0) return null;
        const defaultIndex = indices.find((idx) => idx.id === 'VN100')
            || indices.find((idx) => idx.id === 'VN30')
            || indices[0];
        return defaultIndex.id;
    });
    const [selectedIndustryName, setSelectedIndustryName] = useState<string | null>(null);
    const [selectedBookmarkGroupId, setSelectedBookmarkGroupId] = useState<number | null>(null);
    const [searchQuery, setSearchQuery] = useState('');

    const [benchmarkStocks, setBenchmarkStocks] = useState<Stock[]>([]);
    const [benchmarkIndexId, setBenchmarkIndexId] = useState<string | null>(null);
    const [displayStocks, setDisplayStocks] = useState<Stock[]>([]);
    const [industries, setIndustries] = useState<IndustryInfo[]>([]);
    const [bookmarkGroups, setBookmarkGroups] = useState<BookmarkGroup[]>([]);
    const [portfolioTickers, setPortfolioTickers] = useState<string[]>([]);

    const [sourceLoading, setSourceLoading] = useState(false);
    const [benchmarkLoading, setBenchmarkLoading] = useState(false);
    const [bookmarkLoading, setBookmarkLoading] = useState(false);
    const [sourceError, setSourceError] = useState<string | null>(null);

    const selectedIndex = useMemo(() => {
        if (indices.length === 0) return null;
        if (!selectedIndexId) {
            return indices.find((idx) => idx.id === 'VN100')
                || indices.find((idx) => idx.id === 'VN30')
                || indices[0];
        }
        return indices.find((idx) => idx.id === selectedIndexId) || indices[0];
    }, [indices, selectedIndexId]);

    const selectedBookmarkGroup = useMemo(() => {
        if (!selectedBookmarkGroupId) return null;
        return bookmarkGroups.find((group) => group.id === selectedBookmarkGroupId) || null;
    }, [bookmarkGroups, selectedBookmarkGroupId]);

    const {
        selectorIndustries,
        allowedIndustryNames,
    } = useMemo(
        () => deriveIndexIndustryScope(benchmarkStocks, industries),
        [benchmarkStocks, industries],
    );

    const industriesForSelector = useMemo(() => {
        if (selectedBookmarkGroupId || !selectedIndex) {
            return industries;
        }
        return selectorIndustries;
    }, [industries, selectedBookmarkGroupId, selectedIndex, selectorIndustries]);

    const refreshBookmarkGroups = useCallback(async () => {
        if (!user) {
            setBookmarkGroups([]);
            setBookmarkLoading(false);
            return;
        }
        try {
            setBookmarkLoading(true);
            const response = await stockApi.getBookmarkGroups();
            setBookmarkGroups(response.groups);
        } catch (error) {
            console.error('Failed to fetch bookmark groups for screeners:', error);
        } finally {
            setBookmarkLoading(false);
        }
    }, [user]);

    useEffect(() => {
        const fetchIndustries = async () => {
            try {
                const response = await stockApi.getIndustries();
                setIndustries(response.industries);
            } catch (error) {
                console.error('Failed to fetch industries for screeners:', error);
                setIndustries([]);
            }
        };
        fetchIndustries();
    }, []);

    useEffect(() => {
        if (!user) {
            setBookmarkGroups([]);
            setBookmarkLoading(false);
            setSelectedBookmarkGroupId(null);
            return;
        }
        refreshBookmarkGroups();
    }, [user, refreshBookmarkGroups]);

    useEffect(() => {
        if (!user) {
            setPortfolioTickers([]);
            return;
        }
        let isMounted = true;
        stockApi.getPortfolioPositions()
            .then((response) => {
                if (!isMounted) return;
                const uniqueTickers = Array.from(
                    new Set(response.positions.map((position) => position.ticker.toUpperCase())),
                );
                setPortfolioTickers(uniqueTickers);
            })
            .catch(() => {
                if (!isMounted) return;
                setPortfolioTickers([]);
            });

        return () => {
            isMounted = false;
        };
    }, [user]);

    useEffect(() => {
        if (selectedBookmarkGroupId && !bookmarkGroups.some((group) => group.id === selectedBookmarkGroupId)) {
            setSelectedBookmarkGroupId(null);
        }
    }, [bookmarkGroups, selectedBookmarkGroupId]);

    useEffect(() => {
        if (!selectedIndustryName || !selectedIndex || selectedBookmarkGroupId || benchmarkLoading || benchmarkIndexId !== selectedIndex.id) {
            return;
        }
        if (!allowedIndustryNames.has(selectedIndustryName)) {
            setSelectedIndustryName(null);
        }
    }, [
        allowedIndustryNames,
        benchmarkIndexId,
        benchmarkLoading,
        selectedBookmarkGroupId,
        selectedIndex,
        selectedIndustryName,
    ]);

    useEffect(() => {
        const fetchBenchmarkStocks = async () => {
            if (!selectedIndex) {
                setBenchmarkStocks([]);
                setBenchmarkIndexId(null);
                return;
            }
            try {
                setBenchmarkLoading(true);
                const response = await stockApi.getIndexStocks(selectedIndex.apiEndpoint);
                setBenchmarkStocks(response.stocks);
                setBenchmarkIndexId(selectedIndex.id);
            } catch (error) {
                console.error(`Failed to fetch benchmark stocks for ${selectedIndex.label}:`, error);
                setBenchmarkStocks([]);
                setBenchmarkIndexId(selectedIndex.id);
            } finally {
                setBenchmarkLoading(false);
            }
        };
        fetchBenchmarkStocks();
    }, [selectedIndex]);

    useEffect(() => {
        const fetchDisplayStocks = async () => {
            if (!selectedIndex && !selectedBookmarkGroupId) {
                setDisplayStocks([]);
                return;
            }
            try {
                setSourceLoading(true);
                setSourceError(null);

                if (selectedBookmarkGroupId) {
                    const response = await stockApi.getBookmarkGroupStocks(selectedBookmarkGroupId);
                    setDisplayStocks(response.stocks);
                    return;
                }

                if (selectedIndustryName && selectedIndex) {
                    const industryResponse = await stockApi.getIndustryStocks(selectedIndustryName);
                    const industryTickers = new Set(industryResponse.stocks.map((stock) => stock.ticker.toUpperCase()));
                    let indexStocks = benchmarkStocks;
                    if (benchmarkIndexId !== selectedIndex.id) {
                        const indexResponse = await stockApi.getIndexStocks(selectedIndex.apiEndpoint);
                        indexStocks = indexResponse.stocks;
                    }
                    setDisplayStocks(indexStocks.filter((stock) => industryTickers.has(stock.ticker.toUpperCase())));
                    return;
                }

                if (selectedIndustryName) {
                    const response = await stockApi.getIndustryStocks(selectedIndustryName);
                    setDisplayStocks(response.stocks);
                    return;
                }

                if (selectedIndex) {
                    if (benchmarkIndexId === selectedIndex.id) {
                        setDisplayStocks(benchmarkStocks);
                    } else {
                        const response = await stockApi.getIndexStocks(selectedIndex.apiEndpoint);
                        setDisplayStocks(response.stocks);
                    }
                    return;
                }

                setDisplayStocks([]);
            } catch (error) {
                const label = selectedBookmarkGroupId
                    ? 'bookmark group'
                    : (selectedIndustryName && selectedIndex
                        ? `${selectedIndex.label} + ${selectedIndustryName}`
                        : (selectedIndustryName || (selectedIndex ? selectedIndex.label : 'stocks')));
                setSourceError(`Failed to fetch ${label} stocks data. Please try again.`);
                console.error(`Error fetching screener source data for ${label}:`, error);
            } finally {
                setSourceLoading(false);
            }
        };
        fetchDisplayStocks();
    }, [
        benchmarkIndexId,
        benchmarkStocks,
        selectedBookmarkGroupId,
        selectedIndex,
        selectedIndustryName,
    ]);

    const handleIndexChange = (index: IndexConfig) => {
        setSelectedIndexId(index.id);
    };

    const handleIndustryChange = (industryName: string | null) => {
        setSelectedIndustryName(industryName);
        setSelectedBookmarkGroupId(null);
    };

    const handleBookmarkGroupChange = (groupId: number | null) => {
        setSelectedBookmarkGroupId(groupId);
        if (groupId) {
            setSelectedIndustryName(null);
        }
    };

    if (!selectedIndex && !selectedBookmarkGroupId) {
        return (
            <div className="flex items-center justify-center h-64">
                <p className="text-base-content/60">No indices available for screeners.</p>
            </div>
        );
    }

    const scopeLabel = selectedBookmarkGroup?.name || selectedIndustryName || selectedIndex?.title || 'Screeners';

    return (
        <div className="space-y-6 p-4">
            <div className="flex flex-col gap-4">
                <h2 className="text-2xl font-bold text-base-content">{scopeLabel}</h2>

                <div className="flex flex-wrap gap-2 items-center">
                    <div className="relative">
                        <input
                            type="text"
                            placeholder="Search tickers..."
                            className="input input-sm input-bordered w-32 md:w-48 pl-8"
                            value={searchQuery}
                            onChange={(event) => setSearchQuery(event.target.value)}
                        />
                        <svg
                            className="w-4 h-4 absolute left-2.5 top-1/2 -translate-y-1/2 text-base-content/40"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                        >
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth="2"
                                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                            />
                        </svg>
                    </div>

                    {user ? (
                        <BookmarkSelector
                            groups={bookmarkGroups}
                            selectedGroupId={selectedBookmarkGroupId}
                            onGroupChange={handleBookmarkGroupChange}
                            disabled={bookmarkLoading}
                        />
                    ) : null}

                    <IndustrySelector
                        industries={industriesForSelector}
                        selectedIndustryName={selectedIndustryName}
                        onIndustryChange={handleIndustryChange}
                    />

                    {selectedIndex ? (
                        <IndexSelector
                            indices={indices}
                            selectedIndex={selectedIndex}
                            onIndexChange={handleIndexChange}
                        />
                    ) : null}
                </div>

                <div className="overflow-x-auto">
                    <div className="join min-w-max">
                        <button
                            className={`join-item btn btn-sm ${activeScreener === 'valuation' ? 'btn-primary' : 'btn-outline'}`}
                            onClick={() => setActiveScreener('valuation')}
                        >
                            Valuation
                        </button>
                        <button
                            className={`join-item btn btn-sm ${activeScreener === 'margin-trend' ? 'btn-primary' : 'btn-outline'}`}
                            onClick={() => setActiveScreener('margin-trend')}
                        >
                            Margin Trend
                        </button>
                        <button
                            className={`join-item btn btn-sm ${activeScreener === 'financial-health' ? 'btn-primary' : 'btn-outline'}`}
                            onClick={() => setActiveScreener('financial-health')}
                        >
                            Financial Health
                        </button>
                    </div>
                </div>

            </div>

            {activeScreener === 'valuation' ? (
                <ValuationScreener
                    benchmarkStocks={benchmarkStocks}
                    displayStocks={displayStocks}
                    portfolioTickers={portfolioTickers}
                    benchmarkLabel={selectedIndex?.label || 'N/A'}
                    sourceLoading={sourceLoading}
                    benchmarkLoading={benchmarkLoading}
                    sourceError={sourceError}
                    searchQuery={searchQuery}
                />
            ) : activeScreener === 'margin-trend' ? (
                <MarginTrendScreener
                    benchmarkStocks={benchmarkStocks}
                    displayStocks={displayStocks}
                    industries={industries}
                    portfolioTickers={portfolioTickers}
                    benchmarkLabel={selectedIndex?.label || 'N/A'}
                    sourceLoading={sourceLoading}
                    benchmarkLoading={benchmarkLoading}
                    sourceError={sourceError}
                    searchQuery={searchQuery}
                />
            ) : (
                <FinancialHealthScreener
                    benchmarkStocks={benchmarkStocks}
                    displayStocks={displayStocks}
                    industries={industries}
                    portfolioTickers={portfolioTickers}
                    benchmarkLabel={selectedIndex?.label || 'N/A'}
                    sourceLoading={sourceLoading}
                    benchmarkLoading={benchmarkLoading}
                    sourceError={sourceError}
                    searchQuery={searchQuery}
                />
            )}
        </div>
    );
};

export default ScreenersTab;
