import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
    stockApi,
    type PortfolioImportBroker,
    type Stock,
    type TradingImportResponse,
    type TradingPosition,
} from '../../../api/stockApi';
import { useAuthUser } from '../../auth/useAuthUser';

interface FormState {
    ticker: string;
    quantity: string;
    averageEntryCost: string;
}

type SortKey =
    | 'ticker'
    | 'quantity'
    | 'averageEntryCost'
    | 'currentPrice'
    | 'marketValue'
    | 'pnl';
type SortDirection = 'asc' | 'desc';

const buildEmptyFormState = (): FormState => ({
    ticker: '',
    quantity: '',
    averageEntryCost: '',
});

const getErrorMessage = (error: unknown) => {
    if (typeof error === 'object' && error && 'response' in error) {
        const response = (error as { response?: { status?: number; data?: { detail?: string } } }).response;
        if (response?.status === 409) {
            return 'You already have this ticker in the selected trading account.';
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

export const TradingTab: React.FC = () => {
    const user = useAuthUser();
    const [positions, setPositions] = useState<TradingPosition[]>([]);
    const [quotes, setQuotes] = useState<Record<string, Stock>>({});
    const [loading, setLoading] = useState(true);
    const [quoteLoading, setQuoteLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [addFormState, setAddFormState] = useState<FormState>(buildEmptyFormState);
    const [addFormLoading, setAddFormLoading] = useState(false);
    const [addFormError, setAddFormError] = useState<string | null>(null);
    const [editingId, setEditingId] = useState<number | null>(null);
    const [editFormState, setEditFormState] = useState<FormState>(buildEmptyFormState);
    const [editFormLoading, setEditFormLoading] = useState(false);
    const [editFormError, setEditFormError] = useState<string | null>(null);
    const [searchQuery, setSearchQuery] = useState('');
    const [accountFilter, setAccountFilter] = useState('all');
    const [sortKey, setSortKey] = useState<SortKey | null>(null);
    const [sortDirection, setSortDirection] = useState<SortDirection>('asc');
    const importDialogRef = useRef<HTMLDialogElement>(null);
    const [importBrokers, setImportBrokers] = useState<PortfolioImportBroker[]>([]);
    const [importBrokersLoading, setImportBrokersLoading] = useState(false);
    const [importBrokerId, setImportBrokerId] = useState('');
    const [importAccountLabel, setImportAccountLabel] = useState('');
    const [importOpenedDate, setImportOpenedDate] = useState('');
    const [importFiles, setImportFiles] = useState<File[]>([]);
    const [importLoading, setImportLoading] = useState(false);
    const [importError, setImportError] = useState<string | null>(null);
    const [importResult, setImportResult] = useState<TradingImportResponse | null>(null);

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

    const formatSignedNumber = useMemo(() => {
        return (value: number, options?: Intl.NumberFormatOptions) => {
            const formatted = formatNumber(Math.abs(value), options);
            const prefix = value > 0 ? '+' : value < 0 ? '-' : '';
            return `${prefix}${formatted}`;
        };
    }, [formatNumber]);

    const renderImportResultSummary = () => {
        if (!importResult) {
            return null;
        }

        return (
            <div className="rounded-xl bg-success px-4 py-3 text-sm text-success-content shadow-sm">
                <div className="space-y-3 w-full">
                    <div className="space-y-1">
                        <div>
                            Imported {importResult.imported_positions.length} ticker
                            {importResult.imported_positions.length === 1 ? '' : 's'} from screenshots.
                        </div>
                        <div>
                            Created {importResult.created_count}, updated {importResult.updated_count}, skipped {importResult.skipped_count}.
                        </div>
                    </div>
                    {importResult.import_outcomes.length > 0 ? (
                        <div className="overflow-x-auto">
                            <table className="table table-xs w-full text-success-content">
                                <thead>
                                    <tr>
                                        <th className="text-success-content/80">Ticker</th>
                                        <th className="text-success-content/80">Status</th>
                                        <th className="text-success-content/80">Qty</th>
                                        <th className="text-success-content/80">Entry</th>
                                        <th className="text-success-content/80">Note</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {importResult.import_outcomes.map((outcome) => {
                                        const statusClassName = outcome.status === 'created'
                                            ? 'badge-success'
                                            : outcome.status === 'updated'
                                                ? 'badge-info'
                                                : 'badge-warning';
                                        return (
                                            <tr key={`${outcome.ticker}-${outcome.status}-${outcome.reason ?? 'none'}`}>
                                                <td className="font-semibold">{outcome.ticker}</td>
                                                <td>
                                                    <span className={`badge badge-sm ${statusClassName}`}>
                                                        {outcome.status}
                                                    </span>
                                                </td>
                                                <td>
                                                    {typeof outcome.quantity === 'number'
                                                        ? formatNumber(outcome.quantity, { maximumFractionDigits: 2 })
                                                        : '--'}
                                                </td>
                                                <td>
                                                    {typeof outcome.average_entry_cost === 'number'
                                                        ? formatNumber(outcome.average_entry_cost, { maximumFractionDigits: 2 })
                                                        : '--'}
                                                </td>
                                                <td className="text-success-content/85">
                                                    {outcome.reason ?? '--'}
                                                </td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        </div>
                    ) : null}
                </div>
            </div>
        );
    };

    const accountOptions = useMemo(() => {
        return Array.from(new Set(positions.map((position) => position.account_label)))
            .sort((left, right) => left.localeCompare(right));
    }, [positions]);

    useEffect(() => {
        if (accountFilter !== 'all' && !accountOptions.includes(accountFilter)) {
            setAccountFilter('all');
        }
    }, [accountFilter, accountOptions]);

    const fetchPositions = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await stockApi.getTradingPositions();
            setPositions(response.positions);
        } catch (err) {
            setError(getErrorMessage(err));
        } finally {
            setLoading(false);
        }
    };

    const loadImportBrokers = async () => {
        if (importBrokersLoading) return;
        setImportBrokersLoading(true);
        setImportError(null);
        try {
            const response = await stockApi.getPortfolioImportBrokers();
            setImportBrokers(response);
            if (response.length > 0) {
                setImportBrokerId(response[0].id);
            }
        } catch (err) {
            setImportError(getErrorMessage(err));
        } finally {
            setImportBrokersLoading(false);
        }
    };

    useEffect(() => {
        if (!user) {
            setPositions([]);
            setQuotes({});
            setLoading(false);
            return;
        }
        void fetchPositions();
    }, [user]);

    useEffect(() => {
        if (!user) {
            setImportBrokers([]);
            setImportBrokerId('');
        }
    }, [user]);

    useEffect(() => {
        const tickers = Array.from(new Set(positions.map((position) => position.ticker.toUpperCase())));
        if (tickers.length === 0) {
            setQuotes({});
            return;
        }

        let isMounted = true;
        setQuoteLoading(true);
        stockApi.getStockQuotes(tickers)
            .then((response) => {
                if (!isMounted) return;
                const nextQuotes = response.stocks.reduce<Record<string, Stock>>((accumulator, stock) => {
                    accumulator[stock.ticker.toUpperCase()] = stock;
                    return accumulator;
                }, {});
                setQuotes(nextQuotes);
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

    const filteredPositions = useMemo(() => {
        const query = searchQuery.trim().toUpperCase();
        return positions.filter((position) => {
            if (accountFilter !== 'all' && position.account_label !== accountFilter) {
                return false;
            }

            if (!query) {
                return true;
            }

            const haystack = [
                position.account_label,
                position.ticker,
            ].join(' ').toUpperCase();

            return haystack.includes(query);
        });
    }, [accountFilter, positions, searchQuery]);

    const handleSort = (key: SortKey) => {
        const isSameColumn = sortKey === key;
        const nextDirection: SortDirection = isSameColumn
            ? (sortDirection === 'asc' ? 'desc' : 'asc')
            : 'asc';

        setSortKey(key);
        setSortDirection(nextDirection);
    };

    const sortedFilteredPositions = useMemo(() => {
        if (!sortKey) {
            return filteredPositions;
        }

        const getSortValue = (position: TradingPosition) => {
            const ticker = position.ticker.toUpperCase();
            const quote = quotes[ticker];
            const currentPrice = typeof quote?.price === 'number' && Number.isFinite(quote.price) ? quote.price : null;
            const marketValue = currentPrice !== null ? position.quantity * currentPrice : null;
            const pnl = currentPrice !== null
                ? position.quantity * (currentPrice - position.average_entry_cost)
                : null;

            switch (sortKey) {
                case 'ticker':
                    return ticker;
                case 'quantity':
                    return position.quantity;
                case 'averageEntryCost':
                    return position.average_entry_cost;
                case 'currentPrice':
                    return currentPrice;
                case 'marketValue':
                    return marketValue;
                case 'pnl':
                    return pnl;
                default:
                    return null;
            }
        };

        const direction = sortDirection === 'asc' ? 1 : -1;

        return [...filteredPositions].sort((left, right) => {
            const leftValue = getSortValue(left);
            const rightValue = getSortValue(right);

            if (leftValue === null && rightValue === null) return 0;
            if (leftValue === null) return 1;
            if (rightValue === null) return -1;

            if (typeof leftValue === 'string' && typeof rightValue === 'string') {
                return leftValue.localeCompare(rightValue) * direction;
            }

            return (Number(leftValue) - Number(rightValue)) * direction;
        });
    }, [filteredPositions, quotes, sortDirection, sortKey]);

    const renderSortHeader = (label: string, key: SortKey) => {
        const isActive = sortKey === key;

        return (
            <button
                type="button"
                className="flex items-center gap-1 text-left"
                onClick={() => handleSort(key)}
            >
                <span>{label}</span>
                {isActive ? (
                    <span className="text-[10px] text-base-content/60">
                        {sortDirection === 'asc' ? '▲' : '▼'}
                    </span>
                ) : null}
            </button>
        );
    };

    const totalMarketValue = useMemo(() => {
        let total = 0;
        let pricedCount = 0;

        filteredPositions.forEach((position) => {
            const price = quotes[position.ticker.toUpperCase()]?.price;
            if (typeof price === 'number' && Number.isFinite(price)) {
                total += position.quantity * price;
                pricedCount += 1;
            }
        });

        return { total, pricedCount };
    }, [filteredPositions, quotes]);

    const totalNetPnl = useMemo(() => {
        let total = 0;
        let pricedCount = 0;

        filteredPositions.forEach((position) => {
            const price = quotes[position.ticker.toUpperCase()]?.price;
            if (typeof price === 'number' && Number.isFinite(price)) {
                total += position.quantity * (price - position.average_entry_cost);
                pricedCount += 1;
            }
        });

        return { total, pricedCount };
    }, [filteredPositions, quotes]);

    const totalNetPnlClassName = totalNetPnl.pricedCount === 0
        ? 'text-base-content/50'
        : totalNetPnl.total > 0
            ? 'text-success'
            : totalNetPnl.total < 0
                ? 'text-error'
                : 'text-base-content';

    const handleAddInputChange = (field: keyof FormState) => (
        event: React.ChangeEvent<HTMLInputElement>,
    ) => {
        setAddFormState((prev) => ({
            ...prev,
            [field]: event.target.value,
        }));
    };

    const handleEditInputChange = (field: keyof FormState) => (
        event: React.ChangeEvent<HTMLInputElement>,
    ) => {
        setEditFormState((prev) => ({
            ...prev,
            [field]: event.target.value,
        }));
    };

    const resetAddForm = () => {
        setAddFormState(buildEmptyFormState());
        setAddFormError(null);
    };

    const resetEditForm = () => {
        setEditingId(null);
        setEditFormState(buildEmptyFormState());
        setEditFormError(null);
    };

    const resetImportForm = () => {
        setImportError(null);
        setImportResult(null);
        setImportFiles([]);
        setImportAccountLabel(accountFilter !== 'all' ? accountFilter : '');
        setImportOpenedDate('');
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

    const handleAddSubmit = async () => {
        if (addFormLoading) return;

        setAddFormError(null);
        const accountLabel = accountFilter !== 'all' ? accountFilter : '';
        const ticker = addFormState.ticker.trim().toUpperCase();
        const quantity = Number(addFormState.quantity);
        const averageEntryCost = Number(addFormState.averageEntryCost);

        if (!accountLabel) {
            setAddFormError('Select a specific account filter before adding a position.');
            return;
        }
        if (!ticker) {
            setAddFormError('Ticker is required.');
            return;
        }
        if (!Number.isFinite(quantity) || quantity <= 0) {
            setAddFormError('Quantity must be greater than zero.');
            return;
        }
        if (!Number.isFinite(averageEntryCost) || averageEntryCost <= 0) {
            setAddFormError('Average entry cost must be greater than zero.');
            return;
        }

        setAddFormLoading(true);
        try {
            await stockApi.createTradingPosition({
                account_label: accountLabel,
                ticker,
                quantity,
                average_entry_cost: averageEntryCost,
            });
            await fetchPositions();
            resetAddForm();
        } catch (err) {
            setAddFormError(getErrorMessage(err));
        } finally {
            setAddFormLoading(false);
        }
    };

    const startEdit = (position: TradingPosition) => {
        setEditingId(position.id);
        setEditFormError(null);
        setEditFormState({
            ticker: position.ticker,
            quantity: String(position.quantity),
            averageEntryCost: String(position.average_entry_cost),
        });
    };

    const handleEditSubmit = async (position: TradingPosition) => {
        if (editFormLoading) return;

        setEditFormError(null);
        const ticker = editFormState.ticker.trim().toUpperCase();
        const quantity = Number(editFormState.quantity);
        const averageEntryCost = Number(editFormState.averageEntryCost);

        if (!ticker) {
            setEditFormError('Ticker is required.');
            return;
        }
        if (!Number.isFinite(quantity) || quantity <= 0) {
            setEditFormError('Quantity must be greater than zero.');
            return;
        }
        if (!Number.isFinite(averageEntryCost) || averageEntryCost <= 0) {
            setEditFormError('Average entry cost must be greater than zero.');
            return;
        }

        setEditFormLoading(true);
        try {
            await stockApi.updateTradingPosition(position.id, {
                ticker,
                quantity,
                average_entry_cost: averageEntryCost,
            });
            await fetchPositions();
            resetEditForm();
        } catch (err) {
            setEditFormError(getErrorMessage(err));
        } finally {
            setEditFormLoading(false);
        }
    };

    const handleDelete = async (position: TradingPosition) => {
        if (!window.confirm(`Remove ${position.ticker} from ${position.account_label}?`)) {
            return;
        }
        setError(null);
        try {
            await stockApi.deleteTradingPosition(position.id);
            if (editingId === position.id) {
                resetEditForm();
            }
            await fetchPositions();
        } catch (err) {
            setError(getErrorMessage(err));
        }
    };

    const handleImportSubmit = async () => {
        if (importLoading) return;

        setImportError(null);
        setImportResult(null);

        const accountLabel = importAccountLabel.trim();
        if (importFiles.length === 0) {
            setImportError('Please select one or more broker screenshots.');
            return;
        }
        if (importFiles.some((file) => !file.type.startsWith('image/') && !/\.(png|jpe?g|webp)$/i.test(file.name))) {
            setImportError('Only PNG, JPG, JPEG, or WEBP screenshots are supported.');
            return;
        }
        if (!importBrokerId) {
            setImportError('Please select a broker.');
            return;
        }
        if (!accountLabel) {
            setImportError('Account label is required.');
            return;
        }
        const formData = new FormData();
        importFiles.forEach((file) => {
            formData.append('file', file);
        });
        formData.append('broker_id', importBrokerId);
        formData.append('account_label', accountLabel);
        if (importOpenedDate.trim()) {
            formData.append('opened_date', importOpenedDate.trim());
        }

        setImportLoading(true);
        try {
            const response = await stockApi.importTradingPositions(formData);
            setImportResult(response);
            await fetchPositions();
        } catch (err) {
            setImportError(getErrorMessage(err));
        } finally {
            setImportLoading(false);
        }
    };

    if (!user) {
        return (
            <div className="p-4">
                <div className="alert alert-info shadow-lg">
                    <span>Sign in to manage your trading positions.</span>
                </div>
            </div>
        );
    }

    return (
        <div className="space-y-6 p-4">
            <div className="flex items-center justify-between flex-wrap gap-2">
                <div>
                    <h2 className="text-2xl font-bold text-base-content">Trading</h2>
                    <p className="text-sm text-base-content/60">
                        Monitor active trading positions by broker account without changing your long-term portfolio.
                    </p>
                </div>
                <div className="text-sm text-base-content/60">
                    {positions.length} active position{positions.length === 1 ? '' : 's'}
                </div>
            </div>

            <div className="card bg-base-100 shadow-md border border-base-300">
                <div className="card-body p-4 space-y-4">
                    <div className="flex items-center justify-between flex-wrap gap-3">
                        <div className="flex items-center gap-3">
                            <h3 className="card-title text-base">Open Trades</h3>
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
                                disabled={importLoading}
                            >
                                Import Screenshots
                            </button>
                        </div>
                        <div className="text-right">
                            <div className="text-xs text-base-content/60">Filtered Market Value</div>
                            <div className="flex items-baseline justify-end gap-3">
                                <div className={`text-sm font-semibold ${totalNetPnlClassName}`}>
                                    {totalNetPnl.pricedCount > 0
                                        ? `Open P&L ${formatSignedNumber(totalNetPnl.total, { maximumFractionDigits: 2 })} VND`
                                        : 'Open P&L --'}
                                </div>
                                <div className="text-lg font-semibold text-base-content">
                                    {totalMarketValue.pricedCount > 0
                                        ? `${formatNumber(totalMarketValue.total, { maximumFractionDigits: 2 })} VND`
                                        : '--'}
                                </div>
                            </div>
                        </div>
                    </div>

                    <div className="flex items-center justify-between flex-wrap gap-3">
                        <div className="flex items-center gap-2 flex-wrap">
                            <div className="relative">
                                <input
                                    type="text"
                                    placeholder="Search ticker or account..."
                                    className="input input-sm input-bordered w-56 md:w-72 pl-8"
                                    value={searchQuery}
                                    onChange={(event) => setSearchQuery(event.target.value)}
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
                            <select
                                className="select select-bordered select-sm"
                                value={accountFilter}
                                onChange={(event) => setAccountFilter(event.target.value)}
                            >
                                <option value="all">All accounts</option>
                                {accountOptions.map((accountLabel) => (
                                    <option key={accountLabel} value={accountLabel}>
                                        {accountLabel}
                                    </option>
                                ))}
                            </select>
                        </div>
                        <div className="text-sm text-base-content/60">
                            {filteredPositions.length} shown across {accountFilter === 'all' ? accountOptions.length : 1} account
                            {(accountFilter === 'all' ? accountOptions.length : 1) === 1 ? '' : 's'}
                        </div>
                    </div>

                    {loading ? (
                        <div className="flex flex-col items-center justify-center h-52 gap-3">
                            <span className="loading loading-spinner loading-lg text-primary"></span>
                            <p className="text-base-content/70">Loading trading positions...</p>
                        </div>
                    ) : error ? (
                        <div className="alert alert-error text-sm">
                            <span>{error}</span>
                        </div>
                    ) : (
                        <div className="space-y-3">
                            {renderImportResultSummary()}
                            {importError ? (
                                <div className="alert alert-error text-sm">
                                    <span>{importError}</span>
                                </div>
                            ) : null}
                            {positions.length === 0 ? (
                                <div className="text-sm text-base-content/60">
                                    No trading positions yet. Add your first active trade below.
                                </div>
                            ) : null}
                            {positions.length > 0 && sortedFilteredPositions.length === 0 ? (
                                <div className="text-sm text-base-content/60">
                                    No trading positions match the current filters.
                                </div>
                            ) : null}
                            <div className="dashboard-adaptive-table-wrap">
                                <table className="table table-zebra table-sm w-max min-w-full">
                                    <thead>
                                        <tr>
                                            <th>{renderSortHeader('Ticker', 'ticker')}</th>
                                            <th>{renderSortHeader('Quantity', 'quantity')}</th>
                                            <th>{renderSortHeader('Entry', 'averageEntryCost')}</th>
                                            <th>{renderSortHeader('Current', 'currentPrice')}</th>
                                            <th>{renderSortHeader('Market Value', 'marketValue')}</th>
                                            <th>{renderSortHeader('P&L', 'pnl')}</th>
                                            <th>Actions</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {sortedFilteredPositions.map((position) => {
                                            const isEditing = editingId === position.id;
                                            const display = isEditing ? editFormState : null;
                                            const quantity = isEditing ? Number(display?.quantity) : position.quantity;
                                            const averageEntryCost = isEditing ? Number(display?.averageEntryCost) : position.average_entry_cost;
                                            const quote = quotes[position.ticker.toUpperCase()];
                                            const companyName = quote?.company_name?.trim() ?? '';
                                            const exchangeName = quote?.exchange?.trim() ?? '';
                                            const fullNameWithExchange = companyName
                                                ? (exchangeName ? `${exchangeName} - ${companyName}` : companyName)
                                                : exchangeName;
                                            const price = quote?.price ?? null;
                                            const marketValue = typeof price === 'number' && Number.isFinite(price) && Number.isFinite(quantity)
                                                ? quantity * price
                                                : null;
                                            const costBasis = Number.isFinite(quantity) && Number.isFinite(averageEntryCost)
                                                ? quantity * averageEntryCost
                                                : null;
                                            const pnl = marketValue !== null && costBasis !== null ? marketValue - costBasis : null;
                                            const pnlPercent = pnl !== null && costBasis !== null && costBasis > 0
                                                ? (pnl / costBasis) * 100
                                                : null;
                                            const pnlClassName = pnl === null
                                                ? 'text-base-content/50'
                                                : pnl > 0
                                                    ? 'text-success'
                                                    : pnl < 0
                                                        ? 'text-error'
                                                        : 'text-base-content';

                                            return (
                                                <tr key={position.id}>
                                                    <td className="font-semibold">
                                                        {isEditing ? (
                                                            <input
                                                                type="text"
                                                                className="input input-bordered input-xs w-24"
                                                                value={editFormState.ticker}
                                                                onChange={handleEditInputChange('ticker')}
                                                            />
                                                        ) : fullNameWithExchange ? (
                                                            <div className="tooltip tooltip-right" data-tip={fullNameWithExchange}>
                                                                <span className="cursor-help">{position.ticker}</span>
                                                            </div>
                                                        ) : (
                                                            position.ticker
                                                        )}
                                                    </td>
                                                    <td>
                                                        {isEditing ? (
                                                            <input
                                                                type="number"
                                                                min="1"
                                                                step="1"
                                                                className="input input-bordered input-xs w-24"
                                                                value={editFormState.quantity}
                                                                onChange={handleEditInputChange('quantity')}
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
                                                                value={editFormState.averageEntryCost}
                                                                onChange={handleEditInputChange('averageEntryCost')}
                                                            />
                                                        ) : (
                                                            formatNumber(position.average_entry_cost, { maximumFractionDigits: 2 })
                                                        )}
                                                    </td>
                                                    <td>
                                                        {price !== null ? formatNumber(price, { maximumFractionDigits: 2 }) : '--'}
                                                    </td>
                                                    <td>{marketValue !== null ? formatNumber(marketValue, { maximumFractionDigits: 2 }) : '--'}</td>
                                                    <td className={pnlClassName}>
                                                        {pnl !== null ? (
                                                            <div className="flex flex-col">
                                                                <span>{formatSignedNumber(pnl, { maximumFractionDigits: 2 })}</span>
                                                                {pnlPercent !== null ? (
                                                                    <span className="text-xs">{formatPercent(pnlPercent)}</span>
                                                                ) : null}
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
                                                                {editFormError ? (
                                                                    <span className="text-xs text-error">{editFormError}</span>
                                                                ) : null}
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
                                                />
                                            </td>
                                            <td>
                                                <input
                                                    type="number"
                                                    min="0.01"
                                                    step="0.01"
                                                    className="input input-bordered input-xs w-28"
                                                    placeholder="Entry"
                                                    value={addFormState.averageEntryCost}
                                                    onChange={handleAddInputChange('averageEntryCost')}
                                                />
                                            </td>
                                            <td>
                                                --
                                            </td>
                                            <td>
                                                --
                                            </td>
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
                                                    {addFormError ? (
                                                        <span className="text-xs text-error">{addFormError}</span>
                                                    ) : null}
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
                    <h3 className="font-bold text-lg">Import trading positions from screenshots</h3>
                    <p className="text-sm text-base-content/70 mt-1">
                        Upload one or more broker app screenshots. The LLM will extract ticker, quantity, and average entry cost.
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
                                        onChange={(event) => setImportBrokerId(event.target.value)}
                                    >
                                        <option value="" disabled>Select a broker</option>
                                        {importBrokers.map((broker) => (
                                            <option key={broker.id} value={broker.id}>
                                                {broker.name}
                                            </option>
                                        ))}
                                    </select>
                                </label>

                                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                                    <label className="form-control w-full">
                                        <div className="label">
                                            <span className="label-text">Account label</span>
                                        </div>
                                        <input
                                            type="text"
                                            className="input input-bordered w-full"
                                            placeholder="SSI Swing"
                                            list={accountOptions.length > 0 ? 'trading-import-account-options' : undefined}
                                            value={importAccountLabel}
                                            onChange={(event) => setImportAccountLabel(event.target.value)}
                                        />
                                        {accountOptions.length > 0 ? (
                                            <>
                                                <datalist id="trading-import-account-options">
                                                    {accountOptions.map((accountLabel) => (
                                                        <option key={accountLabel} value={accountLabel} />
                                                    ))}
                                                </datalist>
                                                <div className="label">
                                                    <span className="label-text-alt text-base-content/60">
                                                        Choose an existing account label or type a new one.
                                                    </span>
                                                </div>
                                            </>
                                        ) : null}
                                    </label>
                                    <label className="form-control w-full">
                                        <div className="label">
                                            <span className="label-text">Opened date for new positions (optional)</span>
                                        </div>
                                        <input
                                            type="date"
                                            className="input input-bordered w-full"
                                            value={importOpenedDate}
                                            onChange={(event) => setImportOpenedDate(event.target.value)}
                                        />
                                    </label>
                                </div>

                                <label className="form-control w-full">
                                    <div className="label">
                                        <span className="label-text">Screenshots</span>
                                    </div>
                                    <input
                                        type="file"
                                        accept=".png,.jpg,.jpeg,.webp,image/png,image/jpeg,image/webp"
                                        multiple
                                        className="file-input file-input-bordered w-full"
                                        onChange={(event) => {
                                            setImportFiles(Array.from(event.target.files ?? []));
                                        }}
                                    />
                                </label>

                                <p className="text-xs text-base-content/60">
                                    Existing trades in the same account are refreshed with the imported quantity and average entry cost.
                                </p>
                            </>
                        )}

                        {renderImportResultSummary()}

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

export default TradingTab;
