import React, { useState, useMemo, useRef } from 'react';
import { stockApi } from '../../../api/stockApi';
import type { Stock, BookmarkGroup } from '../../../api/stockApi';
import { useAuthUser } from '../../auth/useAuthUser';

interface StocksTableProps {
    /** List of stocks to display */
    stocks: Stock[];
    /** Bookmark groups for the logged-in user */
    bookmarkGroups?: BookmarkGroup[];
    /** Notify parent to refresh bookmark data */
    onBookmarksUpdated?: (groupId?: number) => void;
}

type SortKey = keyof Stock;
type SortDirection = 'asc' | 'desc';

interface SortConfig {
    key: SortKey;
    direction: SortDirection;
}

export const StocksTable: React.FC<StocksTableProps> = ({
    stocks,
    bookmarkGroups = [],
    onBookmarksUpdated,
}) => {
    const [sortConfig, setSortConfig] = useState<SortConfig>({
        key: 'market_cap',
        direction: 'desc'
    });
    const [isCompanyCollapsed, setIsCompanyCollapsed] = useState(true);
    const user = useAuthUser();
    const isLoggedIn = Boolean(user);
    const dialogRef = useRef<HTMLDialogElement>(null);
    const [activeStock, setActiveStock] = useState<Stock | null>(null);
    const [newGroupName, setNewGroupName] = useState('');
    const [bookmarkError, setBookmarkError] = useState('');
    const [bookmarkLoading, setBookmarkLoading] = useState(false);
    const [editingGroupId, setEditingGroupId] = useState<number | null>(null);
    const [editingGroupName, setEditingGroupName] = useState('');
    const groupTickerMap = useMemo(() => {
        const map = new Map<number, Set<string>>();
        bookmarkGroups.forEach((group) => {
            const normalized = group.tickers.map((ticker) => ticker.toUpperCase());
            map.set(group.id, new Set(normalized));
        });
        return map;
    }, [bookmarkGroups]);

    const bookmarkedTickers = useMemo(() => {
        const tickers = new Set<string>();
        bookmarkGroups.forEach((group) => {
            group.tickers.forEach((ticker) => tickers.add(ticker.toUpperCase()));
        });
        return tickers;
    }, [bookmarkGroups]);

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

    const getErrorMessage = (error: unknown) => {
        if (typeof error === 'object' && error && 'response' in error) {
            const response = (error as { response?: { data?: { detail?: string } } }).response;
            if (response?.data?.detail) {
                return response.data.detail;
            }
        }
        if (error instanceof Error) {
            return error.message;
        }
        return 'Unable to update bookmarks.';
    };

    const openBookmarkDialog = (stock: Stock) => {
        setActiveStock(stock);
        setBookmarkError('');
        setNewGroupName('');
        dialogRef.current?.showModal();
    };

    const closeBookmarkDialog = () => {
        dialogRef.current?.close();
        setActiveStock(null);
        setBookmarkError('');
        setNewGroupName('');
        setEditingGroupId(null);
        setEditingGroupName('');
    };

    const handleToggleGroup = async (groupId: number) => {
        if (!activeStock || bookmarkLoading) {
            return;
        }
        setBookmarkLoading(true);
        setBookmarkError('');
        try {
            const ticker = activeStock.ticker.toUpperCase();
            const isMember = groupTickerMap.get(groupId)?.has(ticker.toUpperCase()) ?? false;
            if (isMember) {
                await stockApi.removeBookmarkStock(groupId, ticker);
            } else {
                await stockApi.addBookmarkStock(groupId, ticker);
            }
            onBookmarksUpdated?.(groupId);
        } catch (err) {
            setBookmarkError(getErrorMessage(err));
        } finally {
            setBookmarkLoading(false);
        }
    };

    const handleCreateGroup = async (event: React.FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        if (!activeStock || bookmarkLoading) {
            return;
        }
        const name = newGroupName.trim();
        if (!name) {
            setBookmarkError('Please provide a group name.');
            return;
        }
        setBookmarkLoading(true);
        setBookmarkError('');
        try {
            const group = await stockApi.createBookmarkGroup({ name });
            await stockApi.addBookmarkStock(group.id, activeStock.ticker.toUpperCase());
            setNewGroupName('');
            onBookmarksUpdated?.(group.id);
        } catch (err) {
            setBookmarkError(getErrorMessage(err));
        } finally {
            setBookmarkLoading(false);
        }
    };

    const startEditGroup = (group: BookmarkGroup) => {
        setEditingGroupId(group.id);
        setEditingGroupName(group.name);
        setBookmarkError('');
    };

    const cancelEditGroup = () => {
        setEditingGroupId(null);
        setEditingGroupName('');
    };

    const handleRenameGroup = async (groupId: number) => {
        if (bookmarkLoading) {
            return;
        }
        const name = editingGroupName.trim();
        if (!name) {
            setBookmarkError('Please provide a group name.');
            return;
        }
        setBookmarkLoading(true);
        setBookmarkError('');
        try {
            await stockApi.updateBookmarkGroup(groupId, { name });
            onBookmarksUpdated?.(groupId);
            setEditingGroupId(null);
            setEditingGroupName('');
        } catch (err) {
            setBookmarkError(getErrorMessage(err));
        } finally {
            setBookmarkLoading(false);
        }
    };

    const handleDeleteGroup = async (groupId: number, groupName: string) => {
        if (bookmarkLoading) {
            return;
        }
        const confirmed = window.confirm(`Delete the \"${groupName}\" group? This cannot be undone.`);
        if (!confirmed) {
            return;
        }
        setBookmarkLoading(true);
        setBookmarkError('');
        try {
            await stockApi.deleteBookmarkGroup(groupId);
            onBookmarksUpdated?.(groupId);
            if (editingGroupId === groupId) {
                cancelEditGroup();
            }
        } catch (err) {
            setBookmarkError(getErrorMessage(err));
        } finally {
            setBookmarkLoading(false);
        }
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

    const totalColumns = isLoggedIn ? 13 : 12;

    return (
        <div className="overflow-x-auto rounded-xl">
            <table className="table table-zebra table-sm">
                <thead className="bg-base-200">
                    <tr>
                        <th className="text-base-content font-bold">#</th>
                        {isLoggedIn ? (
                            <th className="text-base-content font-bold text-center w-12">
                                Fav
                            </th>
                        ) : null}
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
                            className="text-base-content font-bold cursor-pointer hover:bg-base-300 transition-colors w-16"
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
                            <td colSpan={totalColumns} className="text-center py-8 text-base-content/60 italic">
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
                            const isBookmarked = isLoggedIn && bookmarkedTickers.has(stock.ticker.toUpperCase());

                            return (
                                <tr key={stock.ticker} className="hover">
                                    <td className="text-base-content/60">{index + 1}</td>
                                    {isLoggedIn ? (
                                        <td className="text-center w-12">
                                            <button
                                                className={`btn btn-ghost btn-xs ${isBookmarked ? 'text-warning' : ''}`}
                                                onClick={() => openBookmarkDialog(stock)}
                                                title={isBookmarked ? 'Manage bookmark groups' : 'Add to bookmark group'}
                                            >
                                                <svg className="w-4 h-4" fill={isBookmarked ? 'currentColor' : 'none'} stroke="currentColor" viewBox="0 0 24 24">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l2.036 6.27h6.588c.969 0 1.371 1.24.588 1.81l-5.333 3.874 2.036 6.27c.3.921-.755 1.688-1.539 1.118L12 17.77l-5.327 3.869c-.784.57-1.838-.197-1.539-1.118l2.036-6.27-5.333-3.874c-.783-.57-.38-1.81.588-1.81h6.588l2.036-6.27z" />
                                                </svg>
                                            </button>
                                        </td>
                                    ) : null}
                                    <td
                                        className={`${isCompanyCollapsed ? 'w-0 p-0 overflow-hidden opacity-0' : 'whitespace-nowrap'} transition-all duration-200`}
                                        title={isCompanyCollapsed ? "" : fullNameWithExchange}
                                    >
                                        {!isCompanyCollapsed && stock.company_name}
                                    </td>
                                    <td className="w-16">
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

            {isLoggedIn ? (
                <dialog ref={dialogRef} className="modal">
                    <div className="modal-box">
                        <h3 className="font-bold text-lg">
                            {activeStock ? `Manage ${activeStock.ticker} bookmarks` : 'Manage bookmarks'}
                        </h3>
                        <p className="text-sm text-base-content/70 mt-1">
                            Add or remove this stock from your favorite groups.
                        </p>

                        <div className="mt-4 space-y-3">
                            {bookmarkGroups.length === 0 ? (
                                <div className="text-sm text-base-content/60">
                                    No groups yet. Create one below to get started.
                                </div>
                            ) : (
                                <div className="space-y-2">
                                    {bookmarkGroups.map((group) => {
                                        const isMember = activeStock
                                            ? groupTickerMap.get(group.id)?.has(activeStock.ticker.toUpperCase()) ?? false
                                            : false;
                                        const isEditing = editingGroupId === group.id;
                                        return (
                                            <div key={group.id} className="flex items-center justify-between gap-3">
                                                <div className="flex-1">
                                                    {isEditing ? (
                                                        <div className="flex items-center gap-2">
                                                            <input
                                                                type="text"
                                                                className="input input-bordered input-sm w-full"
                                                                value={editingGroupName}
                                                                onChange={(event) => setEditingGroupName(event.target.value)}
                                                                onKeyDown={(event) => {
                                                                    if (event.key === 'Enter') {
                                                                        event.preventDefault();
                                                                        handleRenameGroup(group.id);
                                                                    }
                                                                }}
                                                                maxLength={120}
                                                                disabled={bookmarkLoading}
                                                            />
                                                            <button
                                                                type="button"
                                                                className="btn btn-xs btn-primary"
                                                                onClick={() => handleRenameGroup(group.id)}
                                                                disabled={bookmarkLoading}
                                                            >
                                                                Save
                                                            </button>
                                                            <button
                                                                type="button"
                                                                className="btn btn-xs btn-ghost"
                                                                onClick={cancelEditGroup}
                                                                disabled={bookmarkLoading}
                                                            >
                                                                Cancel
                                                            </button>
                                                        </div>
                                                    ) : (
                                                        <>
                                                            <div className="font-medium">{group.name}</div>
                                                            <div className="text-xs text-base-content/60">
                                                                {group.tickers.length} stocks
                                                            </div>
                                                        </>
                                                    )}
                                                </div>
                                                {!isEditing ? (
                                                    <div className="flex items-center gap-2">
                                                        <button
                                                            type="button"
                                                            className={`btn btn-xs ${isMember ? 'btn-outline btn-error' : 'btn-primary'}`}
                                                            onClick={() => handleToggleGroup(group.id)}
                                                            disabled={bookmarkLoading || !activeStock}
                                                        >
                                                            {isMember ? 'Remove' : 'Add'}
                                                        </button>
                                                        <button
                                                            type="button"
                                                            className="btn btn-xs btn-ghost"
                                                            onClick={() => startEditGroup(group)}
                                                            disabled={bookmarkLoading}
                                                            title="Rename group"
                                                        >
                                                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15.232 5.232l3.536 3.536M9 11l6.293-6.293a1 1 0 011.414 0l2.586 2.586a1 1 0 010 1.414L13 15l-4 1 1-4z" />
                                                            </svg>
                                                        </button>
                                                        <button
                                                            type="button"
                                                            className="btn btn-xs btn-ghost text-error"
                                                            onClick={() => handleDeleteGroup(group.id, group.name)}
                                                            disabled={bookmarkLoading}
                                                            title="Delete group"
                                                        >
                                                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3m-4 0h14" />
                                                            </svg>
                                                        </button>
                                                    </div>
                                                ) : null}
                                            </div>
                                        );
                                    })}
                                </div>
                            )}
                        </div>

                        <form onSubmit={handleCreateGroup} className="mt-4 space-y-2">
                            <label className="form-control w-full">
                                <div className="label">
                                    <span className="label-text">New group</span>
                                </div>
                                <div className="join w-full">
                                    <input
                                        type="text"
                                        className="input input-bordered join-item w-full"
                                        placeholder="e.g. Long-term picks"
                                        value={newGroupName}
                                        onChange={(event) => setNewGroupName(event.target.value)}
                                        maxLength={120}
                                        disabled={bookmarkLoading}
                                    />
                                    <button className="btn btn-primary join-item" type="submit" disabled={bookmarkLoading}>
                                        Create
                                    </button>
                                </div>
                            </label>
                        </form>

                        {bookmarkError ? (
                            <div className="alert alert-error text-sm mt-3">
                                <span>{bookmarkError}</span>
                            </div>
                        ) : null}

                        <div className="modal-action">
                            <button type="button" className="btn btn-ghost" onClick={closeBookmarkDialog}>
                                Close
                            </button>
                        </div>
                    </div>
                    <form method="dialog" className="modal-backdrop">
                        <button onClick={closeBookmarkDialog}>close</button>
                    </form>
                </dialog>
            ) : null}
        </div>
    );
};

export default StocksTable;
