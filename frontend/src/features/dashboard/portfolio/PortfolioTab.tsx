import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
    stockApi,
    type PortfolioImportBroker,
    type PortfolioImportResponse,
    type PortfolioPosition,
    type Stock
} from '../../../api/stockApi';
import { useAuthUser } from '../../auth/useAuthUser';

interface FormState {
    ticker: string;
    quantity: string;
    averageCost: string;
    purchaseDate: string;
}

type SortKey = 'ticker' | 'quantity' | 'averageCost' | 'purchaseDate' | 'currentPrice' | 'marketValue' | 'pnl';
type SortDirection = 'asc' | 'desc';

const emptyFormState: FormState = {
    ticker: '',
    quantity: '',
    averageCost: '',
    purchaseDate: '',
};

const getErrorMessage = (error: unknown) => {
    if (typeof error === 'object' && error && 'response' in error) {
        const response = (error as { response?: { status?: number; data?: { detail?: string } } }).response;
        if (response?.status === 409) {
            return 'You already have a position for this ticker.';
        }
        if (response?.data?.detail) {
            return response.data.detail;
        }
    }
    if (error instanceof Error) {
        return error.message;
    }
    return 'Unable to complete the request.';
};

export const PortfolioTab: React.FC = () => {
    const user = useAuthUser();
    const [positions, setPositions] = useState<PortfolioPosition[]>([]);
    const [quotes, setQuotes] = useState<Record<string, Stock>>({});
    const [loading, setLoading] = useState(true);
    const [quoteLoading, setQuoteLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [addFormError, setAddFormError] = useState<string | null>(null);
    const [addFormLoading, setAddFormLoading] = useState(false);
    const [addFormState, setAddFormState] = useState<FormState>(emptyFormState);
    const [editingId, setEditingId] = useState<number | null>(null);
    const [editFormError, setEditFormError] = useState<string | null>(null);
    const [editFormLoading, setEditFormLoading] = useState(false);
    const [editFormState, setEditFormState] = useState<FormState>(emptyFormState);
    const importDialogRef = useRef<HTMLDialogElement>(null);
    const [importBrokers, setImportBrokers] = useState<PortfolioImportBroker[]>([]);
    const [importBrokersLoading, setImportBrokersLoading] = useState(false);
    const [importBrokerId, setImportBrokerId] = useState('');
    const [importSheet, setImportSheet] = useState('');
    const [importTopLeft, setImportTopLeft] = useState('');
    const [importBottomRight, setImportBottomRight] = useState('');
    const [importFiles, setImportFiles] = useState<File[]>([]);
    const [importLoading, setImportLoading] = useState(false);
    const [importError, setImportError] = useState<string | null>(null);
    const [importResult, setImportResult] = useState<PortfolioImportResponse | null>(null);
    const [sortKey, setSortKey] = useState<SortKey | null>(null);
    const [sortDirection, setSortDirection] = useState<SortDirection>('asc');

    const formatNumber = useMemo(() => {
        return (value: number, options?: Intl.NumberFormatOptions) =>
            new Intl.NumberFormat('en-US', options).format(value);
    }, []);

    const formatPercent = useMemo(() => {
        return (value: number) => {
            const formatted = new Intl.NumberFormat('en-US', {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2,
            }).format(value);
            const prefix = value > 0 ? '+' : '';
            return `${prefix}${formatted}%`;
        };
    }, []);

    const totalMarketValue = useMemo(() => {
        let total = 0;
        let pricedCount = 0;

        positions.forEach((position) => {
            const quote = quotes[position.ticker.toUpperCase()];
            const price = quote?.price;
            if (typeof price === 'number' && Number.isFinite(price)) {
                total += position.quantity * price;
                pricedCount += 1;
            }
        });

        return { total, pricedCount };
    }, [positions, quotes]);

    const handleSort = (key: SortKey) => {
        const isSameColumn = sortKey === key;
        const nextDirection: SortDirection = isSameColumn
            ? (sortDirection === 'asc' ? 'desc' : 'asc')
            : 'asc';

        setSortKey(key);
        setSortDirection(nextDirection);
    };

    const sortedPositions = useMemo(() => {
        if (!sortKey) {
            return positions;
        }

        const getAverageCost = (position: PortfolioPosition) => {
            if (typeof position.average_cost === 'number' && Number.isFinite(position.average_cost) && position.average_cost > 0) {
                return position.average_cost;
            }
            return null;
        };

        const getSortValue = (position: PortfolioPosition) => {
            const ticker = position.ticker.toUpperCase();
            const quote = quotes[ticker];
            const price = typeof quote?.price === 'number' && Number.isFinite(quote.price) ? quote.price : null;
            const averageCost = getAverageCost(position);
            const costBasis = averageCost !== null ? position.quantity * averageCost : null;
            const marketValue = price !== null ? position.quantity * price : null;

            switch (sortKey) {
                case 'ticker':
                    return ticker;
                case 'quantity':
                    return position.quantity;
                case 'averageCost':
                    return averageCost;
                case 'purchaseDate':
                    return position.purchase_date ? position.purchase_date : null;
                case 'currentPrice':
                    return price;
                case 'marketValue':
                    return marketValue;
                case 'pnl':
                    return price !== null && costBasis !== null ? marketValue! - costBasis : null;
                default:
                    return null;
            }
        };

        const direction = sortDirection === 'asc' ? 1 : -1;

        return [...positions].sort((a, b) => {
            const aValue = getSortValue(a);
            const bValue = getSortValue(b);

            if (aValue === null && bValue === null) return 0;
            if (aValue === null) return 1;
            if (bValue === null) return -1;

            if (typeof aValue === 'string' && typeof bValue === 'string') {
                return aValue.localeCompare(bValue) * direction;
            }

            return (Number(aValue) - Number(bValue)) * direction;
        });
    }, [positions, quotes, sortDirection, sortKey]);

    const renderSortHeader = (label: string, key: SortKey) => {
        const isActive = sortKey === key;

        return (
            <button
                type="button"
                className="flex items-center gap-1 text-left"
                onClick={() => handleSort(key)}
            >
                <span>{label}</span>
                {isActive && (
                    <span className="text-[10px] text-base-content/60">
                        {sortDirection === 'asc' ? '▲' : '▼'}
                    </span>
                )}
            </button>
        );
    };

    const resetAddForm = () => {
        setAddFormState(emptyFormState);
        setAddFormError(null);
    };

    const resetEditForm = () => {
        setEditingId(null);
        setEditFormState(emptyFormState);
        setEditFormError(null);
    };

    const applyBrokerDefaults = (broker: PortfolioImportBroker) => {
        setImportBrokerId(broker.id);
        setImportSheet(broker.sheet ?? '');
        setImportTopLeft(broker.top_left);
        setImportBottomRight(broker.bottom_right);
    };

    const resetImportForm = () => {
        setImportFiles([]);
        setImportError(null);
        setImportResult(null);
    };

    const loadImportBrokers = async () => {
        if (importBrokersLoading) return;
        setImportBrokersLoading(true);
        setImportError(null);
        try {
            const response = await stockApi.getPortfolioImportBrokers();
            setImportBrokers(response);
            if (response.length > 0) {
                applyBrokerDefaults(response[0]);
            }
        } catch (err) {
            setImportError(getErrorMessage(err));
        } finally {
            setImportBrokersLoading(false);
        }
    };

    const openImportDialog = async () => {
        resetImportForm();
        if (importBrokers.length === 0) {
            await loadImportBrokers();
        }
        importDialogRef.current?.showModal();
    };

    const closeImportDialog = () => {
        importDialogRef.current?.close();
        resetImportForm();
    };

    const fetchPositions = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await stockApi.getPortfolioPositions();
            setPositions(response.positions);
        } catch (err) {
            setError(getErrorMessage(err));
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        if (!user) {
            setPositions([]);
            setQuotes({});
            setLoading(false);
            return;
        }
        fetchPositions();
    }, [user]);

    useEffect(() => {
        const tickers = positions.map((position) => position.ticker.toUpperCase());
        if (tickers.length === 0) {
            setQuotes({});
            return;
        }

        let isMounted = true;
        setQuoteLoading(true);
        stockApi.getStockQuotes(tickers)
            .then((response) => {
                if (!isMounted) return;
                const map: Record<string, Stock> = {};
                response.stocks.forEach((stock) => {
                    map[stock.ticker.toUpperCase()] = stock;
                });
                setQuotes(map);
            })
            .catch(() => {
                if (!isMounted) return;
            })
            .finally(() => {
                if (!isMounted) return;
                setQuoteLoading(false);
            });

        return () => {
            isMounted = false;
        };
    }, [positions]);

    const handleAddInputChange = (field: keyof FormState) => (event: React.ChangeEvent<HTMLInputElement>) => {
        setAddFormState((prev) => ({
            ...prev,
            [field]: event.target.value,
        }));
    };

    const handleEditInputChange = (field: keyof FormState) => (event: React.ChangeEvent<HTMLInputElement>) => {
        setEditFormState((prev) => ({
            ...prev,
            [field]: event.target.value,
        }));
    };

    const handleImportBrokerChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
        const value = event.target.value;
        setImportBrokerId(value);
        const broker = importBrokers.find((item) => item.id === value);
        if (broker) {
            applyBrokerDefaults(broker);
        }
    };

    const handleImportSubmit = async () => {
        if (importLoading) return;
        setImportError(null);
        setImportResult(null);

        if (importFiles.length === 0) {
            setImportError('Please select a CSV, XLSX, or image file.');
            return;
        }
        const isImageFile = (candidate: File) => {
            if (candidate.type.startsWith('image/')) return true;
            return /\.(png|jpe?g|webp)$/i.test(candidate.name);
        };
        const isSpreadsheetFile = (candidate: File) => /\.(csv|xlsx)$/i.test(candidate.name);
        if (importFiles.length > 1) {
            const nonImages = importFiles.filter((candidate) => !isImageFile(candidate));
            if (nonImages.length > 0) {
                setImportError('When selecting multiple files, all must be images.');
                return;
            }
        } else if (importFiles.length === 1) {
            const single = importFiles[0];
            if (!isImageFile(single) && !isSpreadsheetFile(single)) {
                setImportError('Please select a CSV, XLSX, or image file.');
                return;
            }
        }
        if (!importBrokerId) {
            setImportError('Please select a broker.');
            return;
        }

        const formData = new FormData();
        importFiles.forEach((upload) => {
            formData.append('file', upload);
        });
        formData.append('broker_id', importBrokerId);
        if (importSheet.trim()) {
            formData.append('sheet', importSheet.trim());
        }
        if (importTopLeft.trim()) {
            formData.append('top_left', importTopLeft.trim());
        }
        if (importBottomRight.trim()) {
            formData.append('bottom_right', importBottomRight.trim());
        }

        setImportLoading(true);
        try {
            const response = await stockApi.importPortfolioPositions(formData);
            setImportResult(response);
            await fetchPositions();
        } catch (err) {
            setImportError(getErrorMessage(err));
        } finally {
            setImportLoading(false);
        }
    };

    const handleAddSubmit = async () => {
        if (addFormLoading) return;

        setAddFormError(null);

        const ticker = addFormState.ticker.trim().toUpperCase();
        const quantity = Number(addFormState.quantity);
        const averageCostValue = addFormState.averageCost.trim();
        const averageCost = averageCostValue ? Number(averageCostValue) : null;
        const purchaseDate = addFormState.purchaseDate.trim();

        if (!ticker) {
            setAddFormError('Ticker is required.');
            return;
        }
        if (!Number.isFinite(quantity) || quantity <= 0) {
            setAddFormError('Quantity must be greater than zero.');
            return;
        }
        if (averageCostValue && (!Number.isFinite(averageCost) || averageCost <= 0)) {
            setAddFormError('Average cost must be greater than zero when provided.');
            return;
        }
        setAddFormLoading(true);
        try {
            await stockApi.createPortfolioPosition({
                ticker,
                quantity,
                average_cost: averageCost,
                purchase_date: purchaseDate || null,
            });
            await fetchPositions();
            resetAddForm();
        } catch (err) {
            setAddFormError(getErrorMessage(err));
        } finally {
            setAddFormLoading(false);
        }
    };

    const startEdit = (position: PortfolioPosition) => {
        setEditingId(position.id);
        setEditFormError(null);
        setEditFormState({
            ticker: position.ticker,
            quantity: String(position.quantity),
            averageCost: position.average_cost !== null ? String(position.average_cost) : '',
            purchaseDate: position.purchase_date ? position.purchase_date.slice(0, 10) : '',
        });
    };

    const handleEditSubmit = async (position: PortfolioPosition) => {
        if (editFormLoading) return;
        setEditFormError(null);

        const quantity = Number(editFormState.quantity);
        const averageCostValue = editFormState.averageCost.trim();
        const averageCost = averageCostValue ? Number(averageCostValue) : null;
        const purchaseDate = editFormState.purchaseDate.trim();

        if (!Number.isFinite(quantity) || quantity <= 0) {
            setEditFormError('Quantity must be greater than zero.');
            return;
        }
        if (averageCostValue && (!Number.isFinite(averageCost) || averageCost <= 0)) {
            setEditFormError('Average cost must be greater than zero when provided.');
            return;
        }

        setEditFormLoading(true);
        try {
            await stockApi.updatePortfolioPosition(position.id, {
                quantity,
                average_cost: averageCost,
                purchase_date: purchaseDate || null,
            });
            await fetchPositions();
            resetEditForm();
        } catch (err) {
            setEditFormError(getErrorMessage(err));
        } finally {
            setEditFormLoading(false);
        }
    };

    const handleEditKeyDown = (
        event: React.KeyboardEvent<HTMLInputElement>,
        position: PortfolioPosition,
    ) => {
        if (event.key === 'Enter') {
            event.preventDefault();
            void handleEditSubmit(position);
            return;
        }
        if (event.key === 'Escape') {
            event.preventDefault();
            resetEditForm();
        }
    };

    const handleAddKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
        if (event.key === 'Enter') {
            event.preventDefault();
            void handleAddSubmit();
            return;
        }
        if (event.key === 'Escape') {
            event.preventDefault();
            resetAddForm();
        }
    };

    const handleDelete = async (position: PortfolioPosition) => {
        if (!window.confirm(`Remove ${position.ticker} from your portfolio?`)) {
            return;
        }
        setError(null);
        try {
            await stockApi.deletePortfolioPosition(position.id);
            if (editingId === position.id) {
                resetEditForm();
            }
            await fetchPositions();
        } catch (err) {
            setError(getErrorMessage(err));
        }
    };

    if (!user) {
        return (
            <div className="p-4">
                <div className="alert alert-info shadow-lg">
                    <span>Sign in to manage your portfolio positions.</span>
                </div>
            </div>
        );
    }

    return (
        <div className="space-y-6 p-4">
            <div className="flex items-center justify-between flex-wrap gap-2">
                <div>
                    <h2 className="text-2xl font-bold text-base-content">Portfolio</h2>
                    <p className="text-sm text-base-content/60">Track your positions and performance.</p>
                </div>
                <div className="text-sm text-base-content/60">
                    {positions.length} position{positions.length === 1 ? '' : 's'}
                </div>
            </div>

            <div className="card bg-base-100 shadow-md border border-base-300">
                <div className="card-body p-4 space-y-4">
                    <div className="flex items-center justify-between flex-wrap gap-2">
                        <div className="flex items-center gap-3">
                            <h3 className="card-title text-base">Positions</h3>
                            {quoteLoading && (
                                <span className="text-xs text-base-content/50 flex items-center gap-2">
                                    <span className="loading loading-spinner loading-xs"></span>
                                    Updating prices...
                                </span>
                            )}
                            <button
                                type="button"
                                className="btn btn-xs btn-outline"
                                onClick={() => openImportDialog()}
                            >
                                Import
                            </button>
                        </div>
                        <div className="text-right">
                            <div className="text-xs text-base-content/60">Total Market Value</div>
                            <div className="text-lg font-semibold text-base-content">
                                {totalMarketValue.pricedCount > 0
                                    ? `${formatNumber(totalMarketValue.total, { maximumFractionDigits: 2 })} VND`
                                    : '--'}
                            </div>
                        </div>
                    </div>

                    {loading ? (
                        <div className="flex flex-col items-center justify-center h-52 gap-3">
                            <span className="loading loading-spinner loading-lg text-primary"></span>
                            <p className="text-base-content/70">Loading portfolio positions...</p>
                        </div>
                    ) : error ? (
                        <div className="alert alert-error text-sm">
                            <span>{error}</span>
                        </div>
                    ) : (
                        <div className="space-y-3">
                            {positions.length === 0 && (
                                <div className="text-sm text-base-content/60">
                                    No positions yet. Add your first ticker to start tracking performance.
                                </div>
                            )}
                            <div className="overflow-x-auto">
                                <table className="table table-zebra table-sm">
                                <thead>
                                    <tr>
                                        <th>{renderSortHeader('Ticker', 'ticker')}</th>
                                        <th>{renderSortHeader('Quantity', 'quantity')}</th>
                                        <th>{renderSortHeader('Avg Cost', 'averageCost')}</th>
                                        <th>{renderSortHeader('Purchase Date', 'purchaseDate')}</th>
                                        <th>{renderSortHeader('Current Price', 'currentPrice')}</th>
                                        <th>{renderSortHeader('Market Value', 'marketValue')}</th>
                                        <th>{renderSortHeader('P&L', 'pnl')}</th>
                                        <th>Actions</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {sortedPositions.map((position) => {
                                        const isEditing = editingId === position.id;
                                        const parsedQuantity = Number(editFormState.quantity);
                                        const parsedAverageCost = Number(editFormState.averageCost);
                                        const quantity = isEditing && Number.isFinite(parsedQuantity) && parsedQuantity > 0
                                            ? parsedQuantity
                                            : position.quantity;
                                        const averageCost = isEditing
                                            ? (editFormState.averageCost.trim() === ''
                                                ? null
                                                : (Number.isFinite(parsedAverageCost) && parsedAverageCost > 0 ? parsedAverageCost : null))
                                            : (typeof position.average_cost === 'number' && Number.isFinite(position.average_cost) && position.average_cost > 0
                                                ? position.average_cost
                                                : null);
                                        const quote = quotes[position.ticker.toUpperCase()];
                                        const price = quote?.price ?? null;
                                        const costBasis = averageCost !== null ? quantity * averageCost : null;
                                        const marketValue = price !== null ? quantity * price : null;
                                        const pnl = price !== null && costBasis !== null ? marketValue! - costBasis : null;
                                        const pnlPercent = pnl !== null && costBasis > 0 ? (pnl / costBasis) * 100 : null;
                                        const pnlClassName = pnl === null
                                            ? 'text-base-content/50'
                                            : pnl > 0
                                                ? 'text-success'
                                                : pnl < 0
                                                    ? 'text-error'
                                                    : 'text-base-content';

                                        return (
                                            <tr key={position.id}>
                                                <td className="font-semibold">{position.ticker}</td>
                                                <td>
                                                    {isEditing ? (
                                                        <input
                                                            type="number"
                                                            min="1"
                                                            step="1"
                                                            className="input input-bordered input-xs w-24"
                                                            value={editFormState.quantity}
                                                            onChange={handleEditInputChange('quantity')}
                                                            onKeyDown={(event) => handleEditKeyDown(event, position)}
                                                        />
                                                    ) : (
                                                        formatNumber(position.quantity, { maximumFractionDigits: 2 })
                                                    )}
                                                </td>
                                                <td>
                                                    {isEditing ? (
                                                        <input
                                                            type="number"
                                                            min="0.01"
                                                            step="0.01"
                                                            className="input input-bordered input-xs w-28"
                                                            value={editFormState.averageCost}
                                                            onChange={handleEditInputChange('averageCost')}
                                                            onKeyDown={(event) => handleEditKeyDown(event, position)}
                                                        />
                                                    ) : (
                                                        position.average_cost !== null
                                                            ? formatNumber(position.average_cost, { maximumFractionDigits: 2 })
                                                            : '--'
                                                    )}
                                                </td>
                                                <td>
                                                    {isEditing ? (
                                                        <input
                                                            type="date"
                                                            className="input input-bordered input-xs"
                                                            value={editFormState.purchaseDate}
                                                            onChange={handleEditInputChange('purchaseDate')}
                                                            onKeyDown={(event) => handleEditKeyDown(event, position)}
                                                        />
                                                    ) : (
                                                        position.purchase_date || '--'
                                                    )}
                                                </td>
                                                <td>{price !== null ? formatNumber(price, { maximumFractionDigits: 2 }) : '--'}</td>
                                                <td>{marketValue !== null ? formatNumber(marketValue, { maximumFractionDigits: 2 }) : '--'}</td>
                                                <td className={pnlClassName}>
                                                    {pnl !== null ? (
                                                        <div className="flex flex-col">
                                                            <span>{formatNumber(pnl, { maximumFractionDigits: 2 })}</span>
                                                            {pnlPercent !== null && (
                                                                <span className="text-xs">
                                                                    {formatPercent(pnlPercent)}
                                                                </span>
                                                            )}
                                                        </div>
                                                    ) : '--'}
                                                </td>
                                                <td>
                                                    {isEditing ? (
                                                        <div className="flex flex-col gap-2">
                                                            <div className="flex items-center gap-2">
                                                                <button
                                                                    type="button"
                                                                    className="btn btn-xs btn-primary"
                                                                    onClick={() => handleEditSubmit(position)}
                                                                    disabled={editFormLoading}
                                                                >
                                                                    {editFormLoading ? 'Saving...' : 'Save'}
                                                                </button>
                                                                <button
                                                                    type="button"
                                                                    className="btn btn-xs btn-ghost"
                                                                    onClick={resetEditForm}
                                                                    disabled={editFormLoading}
                                                                >
                                                                    Cancel
                                                                </button>
                                                            </div>
                                                            {editFormError && (
                                                                <span className="text-xs text-error">{editFormError}</span>
                                                            )}
                                                        </div>
                                                    ) : (
                                                        <div className="flex items-center gap-2">
                                                            <button
                                                                type="button"
                                                                className="btn btn-xs btn-ghost"
                                                                onClick={() => startEdit(position)}
                                                            >
                                                                Edit
                                                            </button>
                                                            <button
                                                                type="button"
                                                                className="btn btn-xs btn-ghost text-error"
                                                                onClick={() => handleDelete(position)}
                                                            >
                                                                Remove
                                                            </button>
                                                        </div>
                                                    )}
                                                </td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                                <tfoot>
                                    <tr>
                                        <td>
                                            <input
                                                type="text"
                                                className="input input-bordered input-xs w-24"
                                                placeholder="Ticker"
                                                value={addFormState.ticker}
                                                onChange={handleAddInputChange('ticker')}
                                                onKeyDown={handleAddKeyDown}
                                            />
                                        </td>
                                        <td>
                                            <input
                                                type="number"
                                                min="1"
                                                step="1"
                                                className="input input-bordered input-xs w-24"
                                                placeholder="Qty"
                                                value={addFormState.quantity}
                                                onChange={handleAddInputChange('quantity')}
                                                onKeyDown={handleAddKeyDown}
                                            />
                                        </td>
                                        <td>
                                            <input
                                                type="number"
                                                min="0.01"
                                                step="0.01"
                                                className="input input-bordered input-xs w-28"
                                                placeholder="Avg"
                                                value={addFormState.averageCost}
                                                onChange={handleAddInputChange('averageCost')}
                                                onKeyDown={handleAddKeyDown}
                                            />
                                        </td>
                                        <td>
                                            <input
                                                type="date"
                                                className="input input-bordered input-xs"
                                                value={addFormState.purchaseDate}
                                                onChange={handleAddInputChange('purchaseDate')}
                                                onKeyDown={handleAddKeyDown}
                                            />
                                        </td>
                                        <td>--</td>
                                        <td>--</td>
                                        <td className="text-base-content/50">--</td>
                                        <td>
                                            <div className="flex flex-col gap-2">
                                                <button
                                                    type="button"
                                                    className="btn btn-xs btn-primary"
                                                    onClick={() => handleAddSubmit()}
                                                    disabled={addFormLoading}
                                                >
                                                    {addFormLoading ? 'Adding...' : 'Add'}
                                                </button>
                                                {addFormError && (
                                                    <span className="text-xs text-error">{addFormError}</span>
                                                )}
                                            </div>
                                        </td>
                                    </tr>
                                </tfoot>
                            </table>
                        </div>
                    </div>
                    )}
                </div>
            </div>

            <dialog ref={importDialogRef} className="modal">
                <div className="modal-box">
                    <h3 className="font-bold text-lg">Import portfolio positions</h3>
                    <p className="text-sm text-base-content/70 mt-1">
                        Upload a broker export or one or more screenshots; crop fields are ignored for images.
                    </p>

                    <div className="mt-4 space-y-4">
                        {importBrokersLoading ? (
                            <div className="flex items-center gap-2 text-sm text-base-content/60">
                                <span className="loading loading-spinner loading-sm"></span>
                                Loading broker profiles...
                            </div>
                        ) : (
                            <>
                                <label className="form-control w-full">
                                    <div className="label">
                                        <span className="label-text">Broker</span>
                                    </div>
                                    <select
                                        className="select select-bordered w-full"
                                        value={importBrokerId}
                                        onChange={handleImportBrokerChange}
                                    >
                                        <option value="" disabled>Select a broker</option>
                                        {importBrokers.map((broker) => (
                                            <option key={broker.id} value={broker.id}>
                                                {broker.name}
                                            </option>
                                        ))}
                                    </select>
                                </label>

                                <label className="form-control w-full">
                                    <div className="label">
                                        <span className="label-text">File</span>
                                    </div>
                                    <input
                                        type="file"
                                        accept=".csv,.xlsx,.png,.jpg,.jpeg,.webp"
                                        multiple
                                        className="file-input file-input-bordered w-full"
                                        onChange={(event) => {
                                            const files = Array.from(event.target.files ?? []);
                                            setImportFiles(files);
                                        }}
                                    />
                                </label>

                                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                                    <label className="form-control w-full">
                                        <div className="label">
                                            <span className="label-text">Sheet</span>
                                        </div>
                                        <input
                                            type="text"
                                            className="input input-bordered w-full"
                                            placeholder="Sheet1"
                                            value={importSheet}
                                            onChange={(event) => setImportSheet(event.target.value)}
                                        />
                                    </label>
                                    <label className="form-control w-full">
                                        <div className="label">
                                            <span className="label-text">Top-left</span>
                                        </div>
                                        <input
                                            type="text"
                                            className="input input-bordered w-full"
                                            placeholder="A9"
                                            value={importTopLeft}
                                            onChange={(event) => setImportTopLeft(event.target.value)}
                                        />
                                    </label>
                                    <label className="form-control w-full">
                                        <div className="label">
                                            <span className="label-text">Bottom-right</span>
                                        </div>
                                        <input
                                            type="text"
                                            className="input input-bordered w-full"
                                            placeholder="E"
                                            value={importBottomRight}
                                            onChange={(event) => setImportBottomRight(event.target.value)}
                                        />
                                    </label>
                                </div>
                                <p className="text-xs text-base-content/60">
                                    Use column only (e.g., <span className="font-mono">E</span>) for open-ended rows.
                                </p>
                            </>
                        )}

                        {importResult ? (
                            <div className="alert alert-success text-sm">
                                <div className="space-y-1">
                                    <div>
                                        Imported {importResult.imported_positions.length} ticker
                                        {importResult.imported_positions.length === 1 ? '' : 's'}.
                                    </div>
                                    <div>
                                        Created {importResult.created_count}, updated {importResult.updated_count}, skipped {importResult.skipped_count}.
                                    </div>
                                </div>
                            </div>
                        ) : null}

                        {importError ? (
                            <div className="alert alert-error text-sm">
                                <span>{importError}</span>
                            </div>
                        ) : null}
                    </div>

                    <div className="modal-action">
                        <button type="button" className="btn btn-ghost" onClick={closeImportDialog}>
                            Close
                        </button>
                        <button
                            type="button"
                            className="btn btn-primary"
                            onClick={() => handleImportSubmit()}
                            disabled={importLoading || importBrokersLoading}
                        >
                            {importLoading ? 'Importing...' : 'Import'}
                        </button>
                    </div>
                </div>
                <form method="dialog" className="modal-backdrop">
                    <button onClick={closeImportDialog}>close</button>
                </form>
            </dialog>
        </div>
    );
};

export default PortfolioTab;
