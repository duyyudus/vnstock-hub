import React, { useEffect, useState, useMemo, useCallback, useRef } from 'react';
import { stockApi } from '../../../api/stockApi';
import type { Stock, IndustryInfo, BookmarkGroup } from '../../../api/stockApi';
import { IndexSelector } from './IndexSelector';
import { IndustrySelector } from './IndustrySelector';
import { BookmarkSelector } from './BookmarkSelector';
import { IndustryHoldingChart } from '../components/IndustryHoldingChart';
import { StocksGrowthChart } from './StocksGrowthChart';
import { StocksComparisonChart } from './StocksComparisonChart';
import { StocksRiskReturnScatterPlot } from './StocksRiskReturnScatterPlot';
import { StocksTable } from './StocksTable';
import type { IndexConfig } from './indexConfig';
import { useAuthUser } from '../../auth/useAuthUser';
import {
    resolveCompanyExportCategory,
    resolveFinanceExportCategory,
} from '../../../utils/exportCsv';
import {
    COMPANY_EXPORT_DEFINITIONS,
    FINANCE_EXPORT_DEFINITIONS,
    runTickerExportDefinitions,
} from './stockExport';

interface IndicesTabProps {
    /** List of available indices */
    indices: IndexConfig[];
}

type ViewMode = 'table' | 'growth' | 'comparison' | 'risk_return';

interface ExportNotice {
    kind: 'success' | 'warning';
    message: string;
}

/**
 * Indices Tab - Main container for Index/Industry stock views.
 * Manages state for selection, fetching, and view switching.
 */
export const IndicesTab: React.FC<IndicesTabProps> = ({ indices }) => {
    const user = useAuthUser();

    // --- Selection State ---
    // Default to VN30 if available, otherwise first index
    const [selectedIndex, setSelectedIndex] = useState<IndexConfig | null>(() => {
        if (indices.length === 0) return null;
        return indices.find(idx => idx.id === 'VN30') || indices[0];
    });
    const [selectedIndustryName, setSelectedIndustryName] = useState<string | null>(null);
    const [selectedBookmarkGroupId, setSelectedBookmarkGroupId] = useState<number | null>(null);
    const [bookmarkRefreshKey, setBookmarkRefreshKey] = useState(0);

    // --- Data State ---
    const [stocks, setStocks] = useState<Stock[]>([]);
    const [industries, setIndustries] = useState<IndustryInfo[]>([]);
    const [bookmarkGroups, setBookmarkGroups] = useState<BookmarkGroup[]>([]);
    const [bookmarkLoading, setBookmarkLoading] = useState(false);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [portfolioTickers, setPortfolioTickers] = useState<string[]>([]);

    // --- View State ---
    const [viewMode, setViewMode] = useState<ViewMode>('table');
    const [searchQuery, setSearchQuery] = useState('');
    const indexDetailsDialogRef = useRef<HTMLDialogElement>(null);
    const batchExportDialogRef = useRef<HTMLDialogElement>(null);
    const [batchExportSelections, setBatchExportSelections] = useState<Record<string, boolean>>({});
    const [batchExporting, setBatchExporting] = useState(false);
    const [batchExportNotice, setBatchExportNotice] = useState<ExportNotice | null>(null);

    const isIndexContextActive = !selectedIndustryName && !selectedBookmarkGroupId;

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
            console.error('Failed to fetch bookmark groups:', err);
        } finally {
            setBookmarkLoading(false);
        }
    }, [user]);

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
        if (!user) {
            setPortfolioTickers([]);
            return;
        }
        let isMounted = true;
        stockApi.getPortfolioPositions()
            .then((response) => {
                if (!isMounted) return;
                const uniqueTickers = Array.from(
                    new Set(response.positions.map((position) => position.ticker.toUpperCase()))
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
        if (selectedBookmarkGroupId && !bookmarkGroups.some(group => group.id === selectedBookmarkGroupId)) {
            setSelectedBookmarkGroupId(null);
        }
    }, [bookmarkGroups, selectedBookmarkGroupId]);

    useEffect(() => {
        if (!isIndexContextActive) {
            indexDetailsDialogRef.current?.close();
        }
    }, [isIndexContextActive]);

    useEffect(() => {
        if (!batchExportNotice) {
            return;
        }
        const timeoutId = window.setTimeout(() => {
            setBatchExportNotice(null);
        }, 3000);
        return () => {
            window.clearTimeout(timeoutId);
        };
    }, [batchExportNotice]);

    // Fetch Stocks Data
    useEffect(() => {
        const fetchData = async () => {
            try {
                setLoading(true);
                setError(null);

                if (selectedBookmarkGroupId) {
                    const response = await stockApi.getBookmarkGroupStocks(selectedBookmarkGroupId);
                    setStocks(response.stocks);
                } else if (selectedIndustryName) {
                    const response = await stockApi.getIndustryStocks(selectedIndustryName);
                    setStocks(response.stocks);
                } else if (selectedIndex) {
                    const response = await stockApi.getIndexStocks(selectedIndex.apiEndpoint);
                    setStocks(response.stocks);
                }
            } catch (err: unknown) {
                const label = selectedBookmarkGroupId
                    ? 'bookmark group'
                    : (selectedIndustryName || (selectedIndex ? selectedIndex.label : 'stocks'));

                // If it's a rate limit error (429) or if we can check global status
                try {
                    const syncStatus = await stockApi.getSyncStatus();
                    if (syncStatus.is_rate_limited) {
                        setError(`Market data source is currently busy. Retrying automatically...`);

                        // Set up a one-time retry after a delay
                        setTimeout(() => fetchData(), 30000);
                        return;
                    }
                } catch {
                    // Ignore sync status fetch error
                }

                setError(`Failed to fetch ${label} stocks data. Please try again.`);
                console.error(`Error fetching ${label} data:`, err);
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, [selectedIndex, selectedIndustryName, selectedBookmarkGroupId, bookmarkRefreshKey]);

    // --- Handlers ---

    const handleIndexChange = (newIndex: IndexConfig) => {
        setSelectedIndex(newIndex);
        setSelectedIndustryName(null); // Clear industry when index selected
        setSelectedBookmarkGroupId(null);
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

    const handleBookmarksUpdated = async (groupId?: number) => {
        await refreshBookmarkGroups();
        if (selectedBookmarkGroupId && (!groupId || groupId === selectedBookmarkGroupId)) {
            setBookmarkRefreshKey((prev) => prev + 1);
        }
    };

    const openIndexDetailsModal = () => {
        indexDetailsDialogRef.current?.showModal();
    };

    const closeIndexDetailsModal = () => {
        indexDetailsDialogRef.current?.close();
    };

    // --- Filtering ---

    const filteredStocks = useMemo(() => {
        if (!searchQuery.trim()) return stocks;
        const query = searchQuery.toLowerCase().trim();
        return stocks.filter(stock =>
            stock.ticker.toLowerCase().includes(query)
        );
    }, [stocks, searchQuery]);

    const selectedBatchExportStocks = useMemo(() => {
        return filteredStocks.filter((stock) => batchExportSelections[stock.ticker.toUpperCase()]);
    }, [filteredStocks, batchExportSelections]);

    const openBatchExportModal = () => {
        if (filteredStocks.length === 0 || batchExporting) {
            return;
        }
        const nextSelections: Record<string, boolean> = {};
        filteredStocks.forEach((stock) => {
            nextSelections[stock.ticker.toUpperCase()] = true;
        });
        setBatchExportSelections(nextSelections);
        batchExportDialogRef.current?.showModal();
    };

    const closeBatchExportModal = () => {
        if (batchExporting) {
            return;
        }
        batchExportDialogRef.current?.close();
    };

    const handleBatchExportBackdropClick = (event: React.MouseEvent<HTMLButtonElement>) => {
        if (batchExporting) {
            event.preventDefault();
            return;
        }
        closeBatchExportModal();
    };

    const handleBatchExportDialogCancel = (event: React.SyntheticEvent<HTMLDialogElement, Event>) => {
        if (batchExporting) {
            event.preventDefault();
            return;
        }
        closeBatchExportModal();
    };

    const handleToggleBatchExportTicker = (ticker: string) => {
        const key = ticker.toUpperCase();
        setBatchExportSelections((previous) => ({
            ...previous,
            [key]: !previous[key],
        }));
    };

    const handleSelectAllBatchExportStocks = () => {
        const nextSelections: Record<string, boolean> = {};
        filteredStocks.forEach((stock) => {
            nextSelections[stock.ticker.toUpperCase()] = true;
        });
        setBatchExportSelections(nextSelections);
    };

    const handleDeselectAllBatchExportStocks = () => {
        const nextSelections: Record<string, boolean> = {};
        filteredStocks.forEach((stock) => {
            nextSelections[stock.ticker.toUpperCase()] = false;
        });
        setBatchExportSelections(nextSelections);
    };

    const handleBatchExport = async () => {
        if (batchExporting || selectedBatchExportStocks.length === 0) {
            return;
        }

        setBatchExporting(true);
        setBatchExportNotice(null);

        const hasCustomFolderPreference = Boolean(user?.download_folder?.trim());
        const companyCategory = resolveCompanyExportCategory(user?.company_export_category);
        const financeCategory = resolveFinanceExportCategory(user?.finance_export_category);

        let companyTotal = 0;
        let financeTotal = 0;
        let successCount = 0;
        let failedCount = 0;
        let browserFallbackCount = 0;

        try {
            for (const stock of selectedBatchExportStocks) {
                const companyResult = await runTickerExportDefinitions({
                    ticker: stock.ticker,
                    datasetName: 'company',
                    exportDefinitions: COMPANY_EXPORT_DEFINITIONS,
                    category: companyCategory,
                    user,
                });
                companyTotal += companyResult.total;
                successCount += companyResult.successCount;
                failedCount += companyResult.failedCount;
                browserFallbackCount += companyResult.browserFallbackCount;

                const financeResult = await runTickerExportDefinitions({
                    ticker: stock.ticker,
                    datasetName: 'finance',
                    exportDefinitions: FINANCE_EXPORT_DEFINITIONS,
                    category: financeCategory,
                    user,
                });
                financeTotal += financeResult.total;
                successCount += financeResult.successCount;
                failedCount += financeResult.failedCount;
                browserFallbackCount += financeResult.browserFallbackCount;
            }

            const totalFiles = companyTotal + financeTotal;
            if (failedCount === 0) {
                if (hasCustomFolderPreference && browserFallbackCount > 0) {
                    setBatchExportNotice({
                        kind: 'warning',
                        message: `Exported ${totalFiles}/${totalFiles} files for ${selectedBatchExportStocks.length} stocks. ${browserFallbackCount} saved to browser default location.`,
                    });
                } else if (hasCustomFolderPreference) {
                    setBatchExportNotice({
                        kind: 'success',
                        message: `Exported ${totalFiles} files for ${selectedBatchExportStocks.length} stocks to <TICKER>/${companyCategory} and <TICKER>/${financeCategory}.`,
                    });
                } else {
                    setBatchExportNotice({
                        kind: 'success',
                        message: `Exported ${totalFiles} files for ${selectedBatchExportStocks.length} stocks using browser default location.`,
                    });
                }
            } else {
                const fallbackMessage = hasCustomFolderPreference && browserFallbackCount > 0
                    ? ` ${browserFallbackCount} saved to browser default location.`
                    : '';
                setBatchExportNotice({
                    kind: 'warning',
                    message: `Exported ${successCount}/${totalFiles} files for ${selectedBatchExportStocks.length} stocks. ${failedCount} failed.${fallbackMessage}`,
                });
            }
            batchExportDialogRef.current?.close();
        } finally {
            setBatchExporting(false);
        }
    };

    const selectedBookmarkGroup = useMemo(() => {
        if (!selectedBookmarkGroupId) return null;
        return bookmarkGroups.find(group => group.id === selectedBookmarkGroupId) || null;
    }, [bookmarkGroups, selectedBookmarkGroupId]);

    const industryAllocation = useMemo(() => {
        const industryMap = new Map<string, number>();
        let totalMarketCap = 0;

        stocks.forEach((stock) => {
            const marketCap = Number(stock.market_cap);
            if (!Number.isFinite(marketCap) || marketCap <= 0) {
                return;
            }

            const industryName = stock.industry?.trim() || 'Other';
            industryMap.set(industryName, (industryMap.get(industryName) || 0) + marketCap);
            totalMarketCap += marketCap;
        });

        if (totalMarketCap <= 0) {
            return [];
        }

        return Array.from(industryMap.entries())
            .map(([industry, marketCap]) => ({
                industry,
                allocation: (marketCap / totalMarketCap) * 100,
            }))
            .sort((a, b) => b.allocation - a.allocation);
    }, [stocks]);

    // --- Render ---

    if (!selectedIndex) {
        return <div>No indices available.</div>;
    }

    return (
        <div className="space-y-6 p-4">
            {/* Header Section */}
            <div className="flex flex-col gap-4">
                {/* Row 1: Title */}
                <div>
                    <h2 className="text-2xl font-bold text-base-content">
                        {selectedBookmarkGroup?.name || selectedIndustryName || selectedIndex.title}
                    </h2>
                </div>

                {/* Row 2: Search & Selectors */}
                <div className="flex flex-wrap gap-2 items-center">
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
                    {user ? (
                        <BookmarkSelector
                            groups={bookmarkGroups}
                            selectedGroupId={selectedBookmarkGroupId}
                            onGroupChange={handleBookmarkGroupChange}
                            disabled={bookmarkLoading}
                        />
                    ) : null}
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
                    {isIndexContextActive ? (
                        <button
                            type="button"
                            className="btn btn-sm btn-outline"
                            onClick={openIndexDetailsModal}
                        >
                            Index Details
                        </button>
                    ) : null}
                    <button
                        type="button"
                        className="btn btn-sm btn-primary ml-auto"
                        onClick={openBatchExportModal}
                        disabled={filteredStocks.length === 0 || loading || batchExporting}
                    >
                        {batchExporting ? 'Exporting...' : 'Batch Export'}
                    </button>
                </div>

                {batchExportNotice ? (
                    <div className={`alert text-sm ${batchExportNotice.kind === 'warning' ? 'alert-warning' : 'alert-success'}`}>
                        <span>{batchExportNotice.message}</span>
                    </div>
                ) : null}

                {/* Row 3: View Mode Toggle */}
                <div className="overflow-x-auto">
                    <div className="join min-w-max">
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
                        <button
                            className={`join-item btn btn-sm ${viewMode === 'comparison' ? 'btn-primary' : 'btn-ghost'}`}
                            onClick={() => setViewMode('comparison')}
                        >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                            </svg>
                            Compare
                        </button>
                        <button
                            className={`join-item btn btn-sm ${viewMode === 'risk_return' ? 'btn-primary' : 'btn-ghost'}`}
                            onClick={() => setViewMode('risk_return')}
                        >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8c-2.2 0-4-1.8-4-4H6a6 6 0 0012 0h-2c0 2.2-1.8 4-4 4zm0 8c2.2 0 4 1.8 4 4h2a6 6 0 00-12 0h2c0-2.2 1.8-4 4-4zm-8-4v-2h16v2H4z" />
                            </svg>
                            Risk/Return
                        </button>
                    </div>
                </div>
            </div>

            {/* Content Section */}
            {loading ? (
                <div className="flex flex-col items-center justify-center h-64 gap-4 card bg-base-100 shadow-md border border-base-300">
                    <span className="loading loading-spinner loading-lg text-primary"></span>
                    <p className="text-base-content/70">Loading {selectedBookmarkGroup?.name || selectedIndustryName || selectedIndex.label} stocks...</p>
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
                        ) : viewMode === 'comparison' ? (
                            <StocksComparisonChart
                                stocks={filteredStocks}
                            />
                        ) : viewMode === 'risk_return' ? (
                            <StocksRiskReturnScatterPlot
                                stocks={filteredStocks}
                            />
                        ) : (
                            <StocksTable
                                stocks={filteredStocks}
                                bookmarkGroups={bookmarkGroups}
                                portfolioTickers={portfolioTickers}
                                onBookmarksUpdated={handleBookmarksUpdated}
                            />
                        )}
                    </div>
                </div>
            )}

            <dialog
                ref={batchExportDialogRef}
                className="modal"
                onCancel={handleBatchExportDialogCancel}
            >
                <div className="modal-box max-w-2xl">
                    <h3 className="font-bold text-lg">Batch export stocks</h3>
                    <p className="text-sm text-base-content/70 mt-1">
                        Select stocks to export both company and finance CSV data.
                    </p>

                    <div className="mt-4 flex items-center justify-between gap-3">
                        <span className="text-sm text-base-content/70">
                            {selectedBatchExportStocks.length} selected of {filteredStocks.length}
                        </span>
                        <div className="flex items-center gap-2">
                            <button
                                type="button"
                                className="btn btn-xs btn-ghost"
                                onClick={handleSelectAllBatchExportStocks}
                                disabled={batchExporting || filteredStocks.length === 0}
                            >
                                Select all
                            </button>
                            <button
                                type="button"
                                className="btn btn-xs btn-ghost"
                                onClick={handleDeselectAllBatchExportStocks}
                                disabled={batchExporting || filteredStocks.length === 0}
                            >
                                Deselect all
                            </button>
                        </div>
                    </div>

                    <div className="mt-3 max-h-80 overflow-y-auto rounded-lg border border-base-300">
                        {filteredStocks.length === 0 ? (
                            <div className="p-4 text-sm text-base-content/60">
                                No stocks available for export in the current filter.
                            </div>
                        ) : (
                            <div className="divide-y divide-base-300">
                                {filteredStocks.map((stock) => {
                                    const key = stock.ticker.toUpperCase();
                                    const checked = Boolean(batchExportSelections[key]);
                                    return (
                                        <label
                                            key={stock.ticker}
                                            className="flex cursor-pointer items-center gap-3 px-3 py-2 hover:bg-base-200/60"
                                        >
                                            <input
                                                type="checkbox"
                                                className="checkbox checkbox-sm"
                                                checked={checked}
                                                onChange={() => handleToggleBatchExportTicker(stock.ticker)}
                                                disabled={batchExporting}
                                            />
                                            <span className="w-16 font-semibold uppercase">{stock.ticker}</span>
                                            <span className="truncate text-sm text-base-content/70">
                                                {stock.company_name || '-'}
                                            </span>
                                        </label>
                                    );
                                })}
                            </div>
                        )}
                    </div>

                    <div className="modal-action">
                        <button
                            type="button"
                            className="btn btn-ghost"
                            onClick={closeBatchExportModal}
                            disabled={batchExporting}
                        >
                            Close
                        </button>
                        <button
                            type="button"
                            className="btn btn-primary"
                            onClick={() => void handleBatchExport()}
                            disabled={batchExporting || selectedBatchExportStocks.length === 0}
                        >
                            {batchExporting ? 'Exporting...' : 'Export selected'}
                        </button>
                    </div>
                </div>
                <form method="dialog" className="modal-backdrop">
                    <button onClick={handleBatchExportBackdropClick}>close</button>
                </form>
            </dialog>

            <dialog ref={indexDetailsDialogRef} className="modal">
                <div className="modal-box max-w-3xl">
                    <h3 className="font-bold text-lg">{selectedIndex.label} index details</h3>
                    <p className="text-sm text-base-content/70 mt-1">
                        Industry allocation by market cap
                    </p>

                    <div className="h-80 mt-4">
                        <IndustryHoldingChart data={industryAllocation} />
                    </div>

                    <div className="modal-action">
                        <button type="button" className="btn btn-ghost" onClick={closeIndexDetailsModal}>
                            Close
                        </button>
                    </div>
                </div>
                <form method="dialog" className="modal-backdrop">
                    <button onClick={closeIndexDetailsModal}>close</button>
                </form>
            </dialog>
        </div>
    );
};

export default IndicesTab;
