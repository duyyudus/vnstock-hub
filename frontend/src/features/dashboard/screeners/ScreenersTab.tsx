import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { stockApi } from '../../../api/stockApi';
import type { Stock, IndustryInfo, BookmarkGroup } from '../../../api/stockApi';
import { IndexSelector } from '../indices/IndexSelector';
import { IndustrySelector } from '../indices/IndustrySelector';
import { BookmarkSelector } from '../indices/BookmarkSelector';
import type { IndexConfig } from '../indices/indexConfig';
import { deriveIndexIndustryScope } from '../indices/indexIndustryScope';
import { useAuthUser } from '../../auth/useAuthUser';

interface ScreenersTabProps {
    /** List of available indices */
    indices: IndexConfig[];
}

type ScreenerViewMode = 'value' | 'growth' | 'momentum' | 'quality' | 'volatility';

/**
 * Screeners Tab - layout scaffold for future stock screener implementations.
 */
export const ScreenersTab: React.FC<ScreenersTabProps> = ({ indices }) => {
    const user = useAuthUser();

    const [selectedIndexId, setSelectedIndexId] = useState<string | null>(() => {
        if (indices.length === 0) return null;
        const defaultIndex = indices.find((idx) => idx.id === 'VN30') || indices[0];
        return defaultIndex.id;
    });
    const [selectedIndustryName, setSelectedIndustryName] = useState<string | null>(null);
    const [selectedBookmarkGroupId, setSelectedBookmarkGroupId] = useState<number | null>(null);
    const [searchQuery, setSearchQuery] = useState('');
    const [viewMode, setViewMode] = useState<ScreenerViewMode>('value');

    const [stocks, setStocks] = useState<Stock[]>([]);
    const [indexUniverseStocks, setIndexUniverseStocks] = useState<Stock[]>([]);
    const [indexUniverseIndexId, setIndexUniverseIndexId] = useState<string | null>(null);
    const [industries, setIndustries] = useState<IndustryInfo[]>([]);
    const [bookmarkGroups, setBookmarkGroups] = useState<BookmarkGroup[]>([]);
    const [bookmarkLoading, setBookmarkLoading] = useState(false);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const selectedIndex = useMemo(() => {
        if (indices.length === 0) {
            return null;
        }
        if (!selectedIndexId) {
            return indices.find((idx) => idx.id === 'VN30') || indices[0];
        }
        return indices.find((idx) => idx.id === selectedIndexId) || indices[0];
    }, [indices, selectedIndexId]);

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
        } catch (err) {
            console.error('Failed to fetch bookmark groups for screeners:', err);
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
            setSelectedBookmarkGroupId(null);
            setBookmarkLoading(false);
            return;
        }
        refreshBookmarkGroups();
    }, [user, refreshBookmarkGroups]);

    useEffect(() => {
        if (selectedBookmarkGroupId && !bookmarkGroups.some((group) => group.id === selectedBookmarkGroupId)) {
            setSelectedBookmarkGroupId(null);
        }
    }, [bookmarkGroups, selectedBookmarkGroupId]);

    const {
        selectorIndustries,
        allowedIndustryNames,
    } = useMemo(
        () => deriveIndexIndustryScope(indexUniverseStocks, industries),
        [indexUniverseStocks, industries]
    );
    const industriesForSelector = useMemo(() => {
        if (selectedBookmarkGroupId || !selectedIndex) {
            return industries;
        }
        return selectorIndustries;
    }, [industries, selectedBookmarkGroupId, selectedIndex, selectorIndustries]);

    useEffect(() => {
        if (
            !selectedIndustryName ||
            !selectedIndex ||
            selectedBookmarkGroupId ||
            loading ||
            indexUniverseIndexId !== selectedIndex.id
        ) {
            return;
        }
        if (!allowedIndustryNames.has(selectedIndustryName)) {
            setSelectedIndustryName(null);
        }
    }, [allowedIndustryNames, indexUniverseIndexId, loading, selectedBookmarkGroupId, selectedIndex, selectedIndustryName]);

    useEffect(() => {
        const fetchData = async () => {
            try {
                setLoading(true);
                setError(null);

                if (selectedBookmarkGroupId) {
                    const response = await stockApi.getBookmarkGroupStocks(selectedBookmarkGroupId);
                    setStocks(response.stocks);
                    setIndexUniverseStocks([]);
                    setIndexUniverseIndexId(null);
                } else if (selectedIndex && selectedIndustryName) {
                    const [indexResponse, industryResponse] = await Promise.all([
                        stockApi.getIndexStocks(selectedIndex.apiEndpoint),
                        stockApi.getIndustryStocks(selectedIndustryName)
                    ]);

                    setIndexUniverseStocks(indexResponse.stocks);
                    setIndexUniverseIndexId(selectedIndex.id);

                    const industryTickers = new Set(industryResponse.stocks.map((stock) => stock.ticker));
                    const intersectedStocks = indexResponse.stocks.filter((stock) =>
                        industryTickers.has(stock.ticker)
                    );
                    setStocks(intersectedStocks);
                } else if (selectedIndustryName) {
                    const response = await stockApi.getIndustryStocks(selectedIndustryName);
                    setStocks(response.stocks);
                    setIndexUniverseStocks([]);
                    setIndexUniverseIndexId(null);
                } else if (selectedIndex) {
                    const response = await stockApi.getIndexStocks(selectedIndex.apiEndpoint);
                    setStocks(response.stocks);
                    setIndexUniverseStocks(response.stocks);
                    setIndexUniverseIndexId(selectedIndex.id);
                } else {
                    setStocks([]);
                    setIndexUniverseStocks([]);
                    setIndexUniverseIndexId(null);
                }
            } catch (err) {
                const label = selectedBookmarkGroupId
                    ? 'bookmark group'
                    : (selectedIndustryName && selectedIndex
                        ? `${selectedIndex.label} + ${selectedIndustryName}`
                        : (selectedIndustryName || (selectedIndex ? selectedIndex.label : 'stocks')));
                setError(`Failed to fetch ${label} stocks data. Please try again.`);
                console.error(`Error fetching screener source data for ${label}:`, err);
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, [selectedBookmarkGroupId, selectedIndex, selectedIndustryName]);

    const selectedBookmarkGroup = useMemo(() => {
        if (!selectedBookmarkGroupId) return null;
        return bookmarkGroups.find((group) => group.id === selectedBookmarkGroupId) || null;
    }, [bookmarkGroups, selectedBookmarkGroupId]);

    const filteredStocks = useMemo(() => {
        if (!searchQuery.trim()) return stocks;
        const query = searchQuery.toLowerCase().trim();
        return stocks.filter((stock) =>
            stock.ticker.toLowerCase().includes(query)
        );
    }, [searchQuery, stocks]);

    const handleIndexChange = (index: IndexConfig) => {
        setSelectedIndexId(index.id);
        setIndexUniverseStocks([]);
        setIndexUniverseIndexId(null);
        setSelectedBookmarkGroupId(null);
    };

    const handleIndustryChange = (industryName: string | null) => {
        setSelectedIndustryName(industryName);
        setSelectedBookmarkGroupId(null);
    };

    const handleBookmarkGroupChange = (groupId: number | null) => {
        setSelectedBookmarkGroupId(groupId);
        if (groupId) {
            setIndexUniverseStocks([]);
            setIndexUniverseIndexId(null);
            setSelectedIndustryName(null);
            setSelectedIndexId(null);
        }
    };

    const activeModeLabel = useMemo(() => {
        switch (viewMode) {
            case 'value':
                return 'Value';
            case 'growth':
                return 'Growth';
            case 'momentum':
                return 'Momentum';
            case 'quality':
                return 'Quality';
            case 'volatility':
                return 'Volatility';
            default:
                return 'Screener';
        }
    }, [viewMode]);

    if (!selectedIndex && !selectedBookmarkGroupId) {
        return (
            <div className="flex items-center justify-center h-64">
                <p className="text-base-content/60">No indices available for screeners.</p>
            </div>
        );
    }

    return (
        <div className="space-y-6 p-4">
            <div className="flex flex-col gap-4">
                <div>
                    <h2 className="text-2xl font-bold text-base-content">
                        {selectedBookmarkGroup?.name || selectedIndustryName || selectedIndex?.title || 'Screeners'}
                    </h2>
                </div>

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
                            className={`join-item btn btn-sm ${viewMode === 'value' ? 'btn-primary' : 'btn-ghost'}`}
                            onClick={() => setViewMode('value')}
                        >
                            Value
                        </button>
                        <button
                            className={`join-item btn btn-sm ${viewMode === 'growth' ? 'btn-primary' : 'btn-ghost'}`}
                            onClick={() => setViewMode('growth')}
                        >
                            Growth
                        </button>
                        <button
                            className={`join-item btn btn-sm ${viewMode === 'momentum' ? 'btn-primary' : 'btn-ghost'}`}
                            onClick={() => setViewMode('momentum')}
                        >
                            Momentum
                        </button>
                        <button
                            className={`join-item btn btn-sm ${viewMode === 'quality' ? 'btn-primary' : 'btn-ghost'}`}
                            onClick={() => setViewMode('quality')}
                        >
                            Quality
                        </button>
                        <button
                            className={`join-item btn btn-sm ${viewMode === 'volatility' ? 'btn-primary' : 'btn-ghost'}`}
                            onClick={() => setViewMode('volatility')}
                        >
                            Volatility
                        </button>
                    </div>
                </div>
            </div>

            <div className="card bg-base-100 shadow-md border border-base-300">
                <div className="card-body p-4">
                    <h3 className="text-lg font-semibold">{activeModeLabel} Screener (Placeholder)</h3>
                    <p className="text-sm text-base-content/70 mt-1">
                        This is a layout scaffold only. Screener rules, ranked results, and data visualizations will be added next.
                    </p>
                    {loading ? (
                        <div className="flex flex-col items-center justify-center h-40 gap-3">
                            <span className="loading loading-spinner loading-md text-primary"></span>
                            <p className="text-sm text-base-content/70">
                                Loading {selectedBookmarkGroup?.name || selectedIndustryName || selectedIndex?.label || 'stocks'}...
                            </p>
                        </div>
                    ) : error ? (
                        <div className="alert alert-error mt-4">
                            <span>{error}</span>
                        </div>
                    ) : (
                        <div className="mt-4 grid gap-2 text-sm">
                            <div>
                                <span className="font-medium text-base-content/70">Selected Bookmark Group:</span>{' '}
                                <span>{selectedBookmarkGroup?.name || 'None'}</span>
                            </div>
                            <div>
                                <span className="font-medium text-base-content/70">Selected Index:</span>{' '}
                                <span>{selectedIndex?.label || 'None'}</span>
                            </div>
                            <div>
                                <span className="font-medium text-base-content/70">Selected Industry:</span>{' '}
                                <span>{selectedIndustryName || 'All Industries'}</span>
                            </div>
                            <div>
                                <span className="font-medium text-base-content/70">Search Query:</span>{' '}
                                <span>{searchQuery.trim() || 'None'}</span>
                            </div>
                            <div>
                                <span className="font-medium text-base-content/70">Matched Stocks:</span>{' '}
                                <span>{filteredStocks.length}</span>
                            </div>
                            <div className="flex flex-wrap gap-2 pt-2">
                                {filteredStocks.slice(0, 24).map((stock) => (
                                    <span key={stock.ticker} className="badge badge-outline badge-sm">
                                        {stock.ticker}
                                    </span>
                                ))}
                                {filteredStocks.length > 24 ? (
                                    <span className="text-base-content/60">+{filteredStocks.length - 24} more</span>
                                ) : null}
                                {filteredStocks.length === 0 ? (
                                    <span className="text-base-content/60">No stocks match current filters.</span>
                                ) : null}
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default ScreenersTab;
