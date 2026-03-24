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
    accountLabel: string;
    ticker: string;
    quantity: string;
    averageEntryCost: string;
    openedDate: string;
    targetPrice: string;
    stopLoss: string;
    notes: string;
}

const buildEmptyFormState = (): FormState => ({
    accountLabel: '',
    ticker: '',
    quantity: '',
    averageEntryCost: '',
    openedDate: '',
    targetPrice: '',
    stopLoss: '',
    notes: '',
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
                position.notes ?? '',
            ].join(' ').toUpperCase();

            return haystack.includes(query);
        });
    }, [accountFilter, positions, searchQuery]);

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
        setImportAccountLabel('');
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

    const parseOptionalPositiveNumber = (value: string, fieldLabel: string) => {
        const trimmed = value.trim();
        if (!trimmed) {
            return { value: null, error: null };
        }
        const parsed = Number(trimmed);
        if (!Number.isFinite(parsed) || parsed <= 0) {
            return { value: null, error: `${fieldLabel} must be greater than zero when provided.` };
        }
        return { value: parsed, error: null };
    };

    const handleAddSubmit = async () => {
        if (addFormLoading) return;

        setAddFormError(null);
        const accountLabel = addFormState.accountLabel.trim();
        const ticker = addFormState.ticker.trim().toUpperCase();
        const quantity = Number(addFormState.quantity);
        const averageEntryCost = Number(addFormState.averageEntryCost);
        const openedDate = addFormState.openedDate.trim();
        const targetPriceResult = parseOptionalPositiveNumber(addFormState.targetPrice, 'Target price');
        const stopLossResult = parseOptionalPositiveNumber(addFormState.stopLoss, 'Stop loss');

        if (!accountLabel) {
            setAddFormError('Account label is required.');
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
        if (targetPriceResult.error) {
            setAddFormError(targetPriceResult.error);
            return;
        }
        if (stopLossResult.error) {
            setAddFormError(stopLossResult.error);
            return;
        }

        setAddFormLoading(true);
        try {
            await stockApi.createTradingPosition({
                account_label: accountLabel,
                ticker,
                quantity,
                average_entry_cost: averageEntryCost,
                opened_date: openedDate || null,
                target_price: targetPriceResult.value,
                stop_loss: stopLossResult.value,
                notes: addFormState.notes.trim() || null,
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
            accountLabel: position.account_label,
            ticker: position.ticker,
            quantity: String(position.quantity),
            averageEntryCost: String(position.average_entry_cost),
            openedDate: position.opened_date ? position.opened_date.slice(0, 10) : '',
            targetPrice: position.target_price !== null ? String(position.target_price) : '',
            stopLoss: position.stop_loss !== null ? String(position.stop_loss) : '',
            notes: position.notes ?? '',
        });
    };

    const handleEditSubmit = async (position: TradingPosition) => {
        if (editFormLoading) return;

        setEditFormError(null);
        const accountLabel = editFormState.accountLabel.trim();
        const ticker = editFormState.ticker.trim().toUpperCase();
        const quantity = Number(editFormState.quantity);
        const averageEntryCost = Number(editFormState.averageEntryCost);
        const openedDate = editFormState.openedDate.trim();
        const targetPriceResult = parseOptionalPositiveNumber(editFormState.targetPrice, 'Target price');
        const stopLossResult = parseOptionalPositiveNumber(editFormState.stopLoss, 'Stop loss');

        if (!accountLabel) {
            setEditFormError('Account label is required.');
            return;
        }
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
        if (targetPriceResult.error) {
            setEditFormError(targetPriceResult.error);
            return;
        }
        if (stopLossResult.error) {
            setEditFormError(stopLossResult.error);
            return;
        }

        setEditFormLoading(true);
        try {
            await stockApi.updateTradingPosition(position.id, {
                account_label: accountLabel,
                ticker,
                quantity,
                average_entry_cost: averageEntryCost,
                opened_date: openedDate || null,
                target_price: targetPriceResult.value,
                stop_loss: stopLossResult.value,
                notes: editFormState.notes.trim() || null,
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
                                    placeholder="Search ticker, account, note..."
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
                            {importResult ? (
                                <div className="alert alert-success text-sm">
                                    <div className="space-y-1">
                                        <div>
                                            Imported {importResult.imported_positions.length} ticker
                                            {importResult.imported_positions.length === 1 ? '' : 's'} from screenshots.
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
                            {positions.length === 0 ? (
                                <div className="text-sm text-base-content/60">
                                    No trading positions yet. Add your first active trade below.
                                </div>
                            ) : null}
                            {positions.length > 0 && filteredPositions.length === 0 ? (
                                <div className="text-sm text-base-content/60">
                                    No trading positions match the current filters.
                                </div>
                            ) : null}
                            <div className="dashboard-adaptive-table-wrap">
                                <table className="table table-zebra table-sm w-max min-w-full">
                                    <thead>
                                        <tr>
                                            <th>Account</th>
                                            <th>Ticker</th>
                                            <th>Quantity</th>
                                            <th>Entry</th>
                                            <th>Opened</th>
                                            <th>Target</th>
                                            <th>Stop</th>
                                            <th>Current</th>
                                            <th>Market Value</th>
                                            <th>P&amp;L</th>
                                            <th>Notes</th>
                                            <th>Actions</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {filteredPositions.map((position) => {
                                            const isEditing = editingId === position.id;
                                            const display = isEditing ? editFormState : null;
                                            const quantity = isEditing ? Number(display?.quantity) : position.quantity;
                                            const averageEntryCost = isEditing ? Number(display?.averageEntryCost) : position.average_entry_cost;
                                            const targetPrice = isEditing
                                                ? (display?.targetPrice.trim() ? Number(display.targetPrice) : null)
                                                : position.target_price;
                                            const stopLoss = isEditing
                                                ? (display?.stopLoss.trim() ? Number(display.stopLoss) : null)
                                                : position.stop_loss;
                                            const notes = isEditing ? display?.notes ?? '' : position.notes ?? '';
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
                                                    <td>
                                                        {isEditing ? (
                                                            <input
                                                                type="text"
                                                                className="input input-bordered input-xs w-32"
                                                                value={editFormState.accountLabel}
                                                                onChange={handleEditInputChange('accountLabel')}
                                                            />
                                                        ) : (
                                                            <span className="font-medium">{position.account_label}</span>
                                                        )}
                                                    </td>
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
                                                        {isEditing ? (
                                                            <input
                                                                type="date"
                                                                className="input input-bordered input-xs"
                                                                value={editFormState.openedDate}
                                                                onChange={handleEditInputChange('openedDate')}
                                                            />
                                                        ) : (
                                                            position.opened_date || '--'
                                                        )}
                                                    </td>
                                                    <td>
                                                        {isEditing ? (
                                                            <input
                                                                type="number"
                                                                min="0.01"
                                                                step="0.01"
                                                                className="input input-bordered input-xs w-24"
                                                                value={editFormState.targetPrice}
                                                                onChange={handleEditInputChange('targetPrice')}
                                                            />
                                                        ) : targetPrice !== null ? (
                                                            formatNumber(targetPrice, { maximumFractionDigits: 2 })
                                                        ) : '--'}
                                                    </td>
                                                    <td>
                                                        {isEditing ? (
                                                            <input
                                                                type="number"
                                                                min="0.01"
                                                                step="0.01"
                                                                className="input input-bordered input-xs w-24"
                                                                value={editFormState.stopLoss}
                                                                onChange={handleEditInputChange('stopLoss')}
                                                            />
                                                        ) : stopLoss !== null ? (
                                                            formatNumber(stopLoss, { maximumFractionDigits: 2 })
                                                        ) : '--'}
                                                    </td>
                                                    <td>{price !== null ? formatNumber(price, { maximumFractionDigits: 2 }) : '--'}</td>
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
                                                    <td className="max-w-56">
                                                        {isEditing ? (
                                                            <input
                                                                type="text"
                                                                className="input input-bordered input-xs w-56"
                                                                value={editFormState.notes}
                                                                onChange={handleEditInputChange('notes')}
                                                            />
                                                        ) : notes ? (
                                                            <div className="tooltip tooltip-left" data-tip={notes}>
                                                                <span className="block truncate max-w-56 cursor-help">{notes}</span>
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
                                                    className="input input-bordered input-xs w-32"
                                                    placeholder="Account"
                                                    value={addFormState.accountLabel}
                                                    onChange={handleAddInputChange('accountLabel')}
                                                />
                                            </td>
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
                                                <input
                                                    type="date"
                                                    className="input input-bordered input-xs"
                                                    value={addFormState.openedDate}
                                                    onChange={handleAddInputChange('openedDate')}
                                                    aria-label="Opened date (optional)"
                                                />
                                            </td>
                                            <td>
                                                <input
                                                    type="number"
                                                    min="0.01"
                                                    step="0.01"
                                                    className="input input-bordered input-xs w-24"
                                                    placeholder="Target"
                                                    value={addFormState.targetPrice}
                                                    onChange={handleAddInputChange('targetPrice')}
                                                />
                                            </td>
                                            <td>
                                                <input
                                                    type="number"
                                                    min="0.01"
                                                    step="0.01"
                                                    className="input input-bordered input-xs w-24"
                                                    placeholder="Stop"
                                                    value={addFormState.stopLoss}
                                                    onChange={handleAddInputChange('stopLoss')}
                                                />
                                            </td>
                                            <td>--</td>
                                            <td>--</td>
                                            <td className="text-base-content/50">--</td>
                                            <td>
                                                <input
                                                    type="text"
                                                    className="input input-bordered input-xs w-56"
                                                    placeholder="Notes"
                                                    value={addFormState.notes}
                                                    onChange={handleAddInputChange('notes')}
                                                />
                                            </td>
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
                                            value={importAccountLabel}
                                            onChange={(event) => setImportAccountLabel(event.target.value)}
                                        />
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
                                    Existing trades in the same account keep their target, stop, and notes; only quantity and average entry cost are refreshed.
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

export default TradingTab;
