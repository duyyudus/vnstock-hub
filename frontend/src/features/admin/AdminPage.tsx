import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { AuthWidget } from '../auth/AuthWidget';
import { useAuthUser } from '../auth/useAuthUser';
import {
    type IndexInfo,
    stockApi,
    type PriceAuditActionResponse,
    type PriceSyncActionResponse,
    type SyncStatusResponse,
} from '../../api/stockApi';

const REFRESH_INTERVAL_MS = 5000;

type AdminTab = 'price' | 'finance';

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
    return 'Request failed.';
};

const formatDateTime = (value: string | null | undefined) => {
    if (!value) {
        return '-';
    }
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) {
        return value;
    }
    return parsed.toLocaleString();
};

const parseSymbolsInput = (value: string): string[] => {
    return value
        .split(/[\s,]+/)
        .map((symbol) => symbol.trim().toUpperCase())
        .filter(Boolean);
};

export const AdminPage: React.FC = () => {
    const user = useAuthUser();

    const [activeTab, setActiveTab] = useState<AdminTab>('price');

    const [syncStatus, setSyncStatus] = useState<SyncStatusResponse | null>(null);
    const [loadingStatus, setLoadingStatus] = useState(false);
    const [statusError, setStatusError] = useState<string | null>(null);

    const [syncSymbols, setSyncSymbols] = useState('');
    const [syncIndexSymbol, setSyncIndexSymbol] = useState('');
    const [forceRestart, setForceRestart] = useState(false);

    const [indexOptions, setIndexOptions] = useState<IndexInfo[]>([]);
    const [auditSymbols, setAuditSymbols] = useState('');
    const [auditIndexSymbol, setAuditIndexSymbol] = useState('');
    const [auditStartDate, setAuditStartDate] = useState('');
    const [auditEndDate, setAuditEndDate] = useState('');
    const [auditAutoRepair, setAuditAutoRepair] = useState(false);
    const [auditResult, setAuditResult] = useState<PriceAuditActionResponse | null>(null);

    const [repairSymbols, setRepairSymbols] = useState('');
    const [repairStartDate, setRepairStartDate] = useState('');
    const [repairEndDate, setRepairEndDate] = useState('');

    const [financeSymbols, setFinanceSymbols] = useState('');
    const [financeIndexSymbol, setFinanceIndexSymbol] = useState('');
    const [financeForceRestart, setFinanceForceRestart] = useState(false);

    const [syncRunning, setSyncRunning] = useState(false);
    const [auditRunning, setAuditRunning] = useState(false);
    const [repairRunning, setRepairRunning] = useState(false);
    const [financeRunning, setFinanceRunning] = useState(false);
    const [actionMessage, setActionMessage] = useState<string | null>(null);
    const [actionError, setActionError] = useState<string | null>(null);

    const canAccess = Boolean(user);

    const loadStatuses = useCallback(async () => {
        if (!canAccess) {
            return;
        }

        setLoadingStatus(true);
        setStatusError(null);

        try {
            const sync = await stockApi.getSyncStatus();
            setSyncStatus(sync);
        } catch (error) {
            setStatusError(getErrorMessage(error));
        } finally {
            setLoadingStatus(false);
        }
    }, [canAccess]);

    const loadIndexOptions = useCallback(async () => {
        if (!canAccess) {
            return;
        }
        try {
            const indices = await stockApi.getIndices();
            setIndexOptions(indices.indices);
        } catch {
            setIndexOptions([]);
        }
    }, [canAccess]);

    useEffect(() => {
        loadStatuses();
        loadIndexOptions();

        if (!canAccess) {
            return;
        }

        const intervalId = window.setInterval(() => {
            loadStatuses();
        }, REFRESH_INTERVAL_MS);

        return () => {
            window.clearInterval(intervalId);
        };
    }, [canAccess, loadIndexOptions, loadStatuses]);

    const runAction = async <T extends PriceSyncActionResponse>(
        fn: () => Promise<T>,
        setLoading: React.Dispatch<React.SetStateAction<boolean>>,
        onSuccess?: (result: T) => void,
    ) => {
        setLoading(true);
        setActionMessage(null);
        setActionError(null);

        try {
            const result = await fn();
            setActionMessage(result.message);
            onSuccess?.(result);
            await loadStatuses();
        } catch (error) {
            setActionError(getErrorMessage(error));
        } finally {
            setLoading(false);
        }
    };

    const handleRunSync = async () => {
        const symbols = parseSymbolsInput(syncSymbols);
        await runAction(
            () => stockApi.runPriceSync(
                forceRestart,
                symbols.length > 0 ? symbols : undefined,
                syncIndexSymbol || undefined,
            ),
            setSyncRunning,
        );
    };

    const handleRunAudit = async () => {
        if (!auditStartDate || !auditEndDate) {
            setActionError('Please provide both start date and end date for audit.');
            return;
        }

        const symbols = parseSymbolsInput(auditSymbols);

        await runAction(
            () => stockApi.runPriceAudit(
                auditStartDate,
                auditEndDate,
                symbols.length > 0 ? symbols : undefined,
                auditIndexSymbol || undefined,
                auditAutoRepair,
            ),
            setAuditRunning,
            (result) => setAuditResult(result as PriceAuditActionResponse),
        );
    };

    const handleRunRepair = async () => {
        const symbols = parseSymbolsInput(repairSymbols);

        if (symbols.length === 0) {
            setActionError('Please provide at least one symbol for repair.');
            return;
        }

        if (!repairStartDate || !repairEndDate) {
            setActionError('Please provide both start date and end date for repair.');
            return;
        }

        await runAction(
            () => stockApi.runPriceRepairSync(symbols, repairStartDate, repairEndDate),
            setRepairRunning,
        );
    };

    const handleRunFinanceSync = async () => {
        const symbols = parseSymbolsInput(financeSymbols);
        await runAction(
            () => stockApi.runFinanceSync(
                financeForceRestart,
                symbols.length > 0 ? symbols : undefined,
                financeIndexSymbol || undefined,
            ),
            setFinanceRunning,
        );
    };

    const runtimeSync = syncStatus?.price_sync.sync;
    const runtimeAudit = syncStatus?.price_sync.audit;
    const runtimeRepair = syncStatus?.price_sync.repair;
    const runtimeFinance = syncStatus?.finance_sync;

    const syncActive = syncRunning || Boolean(runtimeSync?.is_running);
    const auditActive = auditRunning || Boolean(runtimeAudit?.is_running);
    const repairActive = repairRunning || Boolean(runtimeRepair?.is_running);
    const financeActive = financeRunning || Boolean(runtimeFinance?.is_running);
    const anyJobActive = syncActive || auditActive || repairActive || financeActive;

    const syncProgressPercent = useMemo(() => {
        const value = runtimeSync?.progress ?? 0;
        return Math.max(0, Math.min(100, Math.round(value * 100)));
    }, [runtimeSync?.progress]);

    const financeProgressPercent = useMemo(() => {
        const value = runtimeFinance?.progress ?? 0;
        return Math.max(0, Math.min(100, Math.round(value * 100)));
    }, [runtimeFinance?.progress]);

    if (!canAccess) {
        return (
            <div className="min-h-screen bg-base-300">
                <header className="navbar bg-base-100 shadow-lg px-4 md:px-6">
                    <div className="max-w-[64rem] mx-auto w-full flex items-center">
                        <div className="flex-1">
                            <a href="/" className="btn btn-ghost btn-sm">← Back to Dashboard</a>
                        </div>
                        <div className="flex-none">
                            <AuthWidget />
                        </div>
                    </div>
                </header>
                <main className="max-w-[64rem] mx-auto p-6">
                    <div className="alert alert-warning">
                        <span>Please sign in to access the admin page.</span>
                    </div>
                </main>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-base-300">
            <header className="navbar bg-base-100 shadow-lg px-4 md:px-6">
                <div className="max-w-[100rem] mx-auto w-full flex items-center gap-3">
                    <div className="flex-1 flex items-center gap-2">
                        <a href="/" className="btn btn-ghost btn-sm">← Dashboard</a>
                        <h1 className="text-xl font-bold">Admin Sync Control</h1>
                    </div>
                    <AuthWidget />
                </div>
            </header>

            <main className="max-w-[100rem] mx-auto p-6 space-y-6">
                {statusError ? (
                    <div className="alert alert-error">
                        <span>Status load failed: {statusError}</span>
                    </div>
                ) : null}

                {actionMessage ? (
                    <div className="alert alert-success">
                        <span>{actionMessage}</span>
                    </div>
                ) : null}

                {actionError ? (
                    <div className="alert alert-error">
                        <span>{actionError}</span>
                    </div>
                ) : null}

                <div role="tablist" className="tabs tabs-boxed w-fit">
                    <button
                        role="tab"
                        className={`tab ${activeTab === 'price' ? 'tab-active' : ''}`}
                        onClick={() => setActiveTab('price')}
                    >
                        Price Sync
                    </button>
                    <button
                        role="tab"
                        className={`tab ${activeTab === 'finance' ? 'tab-active' : ''}`}
                        onClick={() => setActiveTab('finance')}
                    >
                        Finance Sync
                    </button>
                </div>

                {activeTab === 'price' ? (
                    <>
                        <section className="grid gap-4 md:grid-cols-3">
                            <div className="card bg-base-100 shadow-lg">
                                <div className="card-body">
                                    <h2 className="card-title text-base">Price Sync Status</h2>
                                    <p>Running: <strong>{runtimeSync?.is_running ? 'Yes' : 'No'}</strong></p>
                                    <p>Progress: <strong>{syncProgressPercent}%</strong></p>
                                    <progress className="progress progress-primary w-full" value={syncProgressPercent} max={100}></progress>
                                    <p>Processed: {runtimeSync?.processed_symbols ?? 0} / {runtimeSync?.total_symbols ?? 0}</p>
                                    <p>Success: {runtimeSync?.success_symbols ?? 0}</p>
                                    <p>Failed: {runtimeSync?.failed_symbols ?? 0}</p>
                                    <p>Current symbol: {runtimeSync?.current_symbol ?? '-'}</p>
                                    <p>Last run: {formatDateTime(runtimeSync?.last_run_at)}</p>
                                    <p>Started: {formatDateTime(runtimeSync?.started_at)}</p>
                                    <p>Error: {runtimeSync?.error ?? '-'}</p>
                                </div>
                            </div>

                            <div className="card bg-base-100 shadow-lg">
                                <div className="card-body">
                                    <h2 className="card-title text-base">Gap Audit Status</h2>
                                    <p>Running: <strong>{runtimeAudit?.is_running ? 'Yes' : 'No'}</strong></p>
                                    <p>Processed: {runtimeAudit?.processed_symbols ?? 0} / {runtimeAudit?.total_symbols ?? 0}</p>
                                    <p>Success: {runtimeAudit?.success_symbols ?? 0}</p>
                                    <p>Failed: {runtimeAudit?.failed_symbols ?? 0}</p>
                                    <p>Current symbol: {runtimeAudit?.current_symbol ?? '-'}</p>
                                    <p>Last run: {formatDateTime(runtimeAudit?.last_run_at)}</p>
                                    <p>Error: {runtimeAudit?.error ?? '-'}</p>
                                </div>
                            </div>

                            <div className="card bg-base-100 shadow-lg">
                                <div className="card-body">
                                    <h2 className="card-title text-base">Repair Status</h2>
                                    <p>Running: <strong>{runtimeRepair?.is_running ? 'Yes' : 'No'}</strong></p>
                                    <p>Processed: {runtimeRepair?.processed_symbols ?? 0} / {runtimeRepair?.total_symbols ?? 0}</p>
                                    <p>Success: {runtimeRepair?.success_symbols ?? 0}</p>
                                    <p>Failed: {runtimeRepair?.failed_symbols ?? 0}</p>
                                    <p>Current symbol: {runtimeRepair?.current_symbol ?? '-'}</p>
                                    <p>Last run: {formatDateTime(runtimeRepair?.last_run_at)}</p>
                                    <p>Error: {runtimeRepair?.error ?? '-'}</p>
                                </div>
                            </div>
                        </section>

                        <section className="grid gap-4 lg:grid-cols-3">
                            <div className="card bg-base-100 shadow-lg">
                                <div className="card-body space-y-3">
                                    <h2 className="card-title">Run Price Sync</h2>
                                    <label className="form-control">
                                        <span className="label-text">Index scope (optional)</span>
                                        <select
                                            className="select select-bordered"
                                            value={syncIndexSymbol}
                                            onChange={(event) => setSyncIndexSymbol(event.target.value)}
                                        >
                                            <option value="">All indices</option>
                                            {indexOptions.map((index) => (
                                                <option key={index.symbol} value={index.symbol}>
                                                    {index.symbol} - {index.name}
                                                </option>
                                            ))}
                                        </select>
                                    </label>
                                    <label className="form-control">
                                        <span className="label-text">Symbols (optional, comma/space separated)</span>
                                        <input
                                            type="text"
                                            className="input input-bordered"
                                            value={syncSymbols}
                                            onChange={(event) => setSyncSymbols(event.target.value)}
                                            placeholder="All symbols if empty"
                                        />
                                    </label>
                                    <label className="label cursor-pointer justify-start gap-3">
                                        <input
                                            type="checkbox"
                                            className="checkbox"
                                            checked={forceRestart}
                                            onChange={(event) => setForceRestart(event.target.checked)}
                                        />
                                        <span className="label-text">Force restart if already running</span>
                                    </label>
                                    <button
                                        className="btn btn-primary"
                                        onClick={handleRunSync}
                                        disabled={loadingStatus || anyJobActive}
                                    >
                                        {syncActive ? <span className="loading loading-spinner loading-xs"></span> : null}
                                        {syncActive
                                            ? 'Syncing...'
                                            : anyJobActive
                                                ? 'Waiting for current job...'
                                                : 'Run Price Sync'}
                                    </button>
                                </div>
                            </div>

                            <div className="card bg-base-100 shadow-lg">
                                <div className="card-body space-y-3">
                                    <h2 className="card-title">Run Gap Audit</h2>
                                    <label className="form-control">
                                        <span className="label-text">Index scope (optional)</span>
                                        <select
                                            className="select select-bordered"
                                            value={auditIndexSymbol}
                                            onChange={(event) => setAuditIndexSymbol(event.target.value)}
                                        >
                                            <option value="">All indices</option>
                                            {indexOptions.map((index) => (
                                                <option key={index.symbol} value={index.symbol}>
                                                    {index.symbol} - {index.name}
                                                </option>
                                            ))}
                                        </select>
                                    </label>
                                    <label className="form-control">
                                        <span className="label-text">Symbols (optional, comma/space separated)</span>
                                        <input
                                            type="text"
                                            className="input input-bordered"
                                            value={auditSymbols}
                                            onChange={(event) => setAuditSymbols(event.target.value)}
                                            placeholder="All symbols if empty"
                                        />
                                    </label>
                                    <div className="grid grid-cols-2 gap-2">
                                        <label className="form-control">
                                            <span className="label-text">Start date</span>
                                            <input
                                                type="date"
                                                className="input input-bordered"
                                                value={auditStartDate}
                                                onChange={(event) => setAuditStartDate(event.target.value)}
                                            />
                                        </label>
                                        <label className="form-control">
                                            <span className="label-text">End date</span>
                                            <input
                                                type="date"
                                                className="input input-bordered"
                                                value={auditEndDate}
                                                onChange={(event) => setAuditEndDate(event.target.value)}
                                            />
                                        </label>
                                    </div>
                                    <label className="label cursor-pointer justify-start gap-3">
                                        <input
                                            type="checkbox"
                                            className="checkbox"
                                            checked={auditAutoRepair}
                                            onChange={(event) => setAuditAutoRepair(event.target.checked)}
                                        />
                                        <span className="label-text">Auto repair detected gaps</span>
                                    </label>
                                    <button
                                        className="btn btn-secondary"
                                        onClick={handleRunAudit}
                                        disabled={loadingStatus || anyJobActive}
                                    >
                                        {auditActive ? <span className="loading loading-spinner loading-xs"></span> : null}
                                        {auditActive
                                            ? 'Auditing...'
                                            : anyJobActive
                                                ? 'Waiting for current job...'
                                                : 'Run Gap Audit'}
                                    </button>
                                </div>
                            </div>

                            <div className="card bg-base-100 shadow-lg">
                                <div className="card-body space-y-3">
                                    <h2 className="card-title">Run Repair</h2>
                                    <label className="form-control">
                                        <span className="label-text">Symbols (comma/space separated)</span>
                                        <input
                                            type="text"
                                            className="input input-bordered"
                                            value={repairSymbols}
                                            onChange={(event) => setRepairSymbols(event.target.value)}
                                            placeholder="VCB,FPT,SSI"
                                        />
                                    </label>
                                    <div className="grid grid-cols-2 gap-2">
                                        <label className="form-control">
                                            <span className="label-text">Start date</span>
                                            <input
                                                type="date"
                                                className="input input-bordered"
                                                value={repairStartDate}
                                                onChange={(event) => setRepairStartDate(event.target.value)}
                                            />
                                        </label>
                                        <label className="form-control">
                                            <span className="label-text">End date</span>
                                            <input
                                                type="date"
                                                className="input input-bordered"
                                                value={repairEndDate}
                                                onChange={(event) => setRepairEndDate(event.target.value)}
                                            />
                                        </label>
                                    </div>
                                    <button
                                        className="btn btn-accent"
                                        onClick={handleRunRepair}
                                        disabled={loadingStatus || anyJobActive}
                                    >
                                        {repairActive ? <span className="loading loading-spinner loading-xs"></span> : null}
                                        {repairActive
                                            ? 'Repairing...'
                                            : anyJobActive
                                                ? 'Waiting for current job...'
                                                : 'Run Repair Sync'}
                                    </button>
                                </div>
                            </div>
                        </section>

                        {auditResult ? (
                            <section className="card bg-base-100 shadow-lg">
                                <div className="card-body space-y-3">
                                    <h2 className="card-title">Latest Audit Results</h2>
                                    <p>Audited symbols: <strong>{auditResult.audited_symbols}</strong></p>
                                    <p>Symbols with gaps: <strong>{auditResult.symbols_with_gaps}</strong></p>
                                    <p>Total missing dates: <strong>{auditResult.total_missing_dates}</strong></p>
                                    <p>Total repaired dates: <strong>{auditResult.total_repaired_dates}</strong></p>
                                    <div className="overflow-x-auto">
                                        <table className="table table-zebra">
                                            <thead>
                                                <tr>
                                                    <th>Symbol</th>
                                                    <th>Missing</th>
                                                    <th>Repaired</th>
                                                    <th>Samples</th>
                                                    <th>Error</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {auditResult.results.map((row) => (
                                                    <tr key={row.symbol}>
                                                        <td>{row.symbol}</td>
                                                        <td>{row.missing_dates}</td>
                                                        <td>{row.repaired_dates}</td>
                                                        <td>{row.missing_date_samples.join(', ') || '-'}</td>
                                                        <td>{row.error || '-'}</td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            </section>
                        ) : null}
                    </>
                ) : null}

                {activeTab === 'finance' ? (
                    <section className="grid gap-4 lg:grid-cols-2">
                        <div className="card bg-base-100 shadow-lg">
                            <div className="card-body">
                                <h2 className="card-title text-base">Finance Sync Status</h2>
                                <p>Running: <strong>{runtimeFinance?.is_running ? 'Yes' : 'No'}</strong></p>
                                <p>Progress: <strong>{financeProgressPercent}%</strong></p>
                                <progress className="progress progress-info w-full" value={financeProgressPercent} max={100}></progress>
                                <p>Processed: {runtimeFinance?.processed_symbols ?? 0} / {runtimeFinance?.total_symbols ?? 0}</p>
                                <p>Success: {runtimeFinance?.success_symbols ?? 0}</p>
                                <p>Failed: {runtimeFinance?.failed_symbols ?? 0}</p>
                                <p>Current symbol: {runtimeFinance?.current_symbol ?? '-'}</p>
                                <p>Last run: {formatDateTime(runtimeFinance?.last_run_at)}</p>
                                <p>Started: {formatDateTime(runtimeFinance?.started_at)}</p>
                                <p>Error: {runtimeFinance?.error ?? '-'}</p>
                            </div>
                        </div>

                        <div className="card bg-base-100 shadow-lg">
                            <div className="card-body space-y-3">
                                <h2 className="card-title">Run Finance Sync</h2>
                                <label className="form-control">
                                    <span className="label-text">Index scope (optional)</span>
                                    <select
                                        className="select select-bordered"
                                        value={financeIndexSymbol}
                                        onChange={(event) => setFinanceIndexSymbol(event.target.value)}
                                    >
                                        <option value="">All indices</option>
                                        {indexOptions.map((index) => (
                                            <option key={index.symbol} value={index.symbol}>
                                                {index.symbol} - {index.name}
                                            </option>
                                        ))}
                                    </select>
                                </label>
                                <label className="form-control">
                                    <span className="label-text">Symbols (optional, comma/space separated)</span>
                                    <input
                                        type="text"
                                        className="input input-bordered"
                                        value={financeSymbols}
                                        onChange={(event) => setFinanceSymbols(event.target.value)}
                                        placeholder="All symbols if empty"
                                    />
                                </label>
                                <label className="label cursor-pointer justify-start gap-3">
                                    <input
                                        type="checkbox"
                                        className="checkbox"
                                        checked={financeForceRestart}
                                        onChange={(event) => setFinanceForceRestart(event.target.checked)}
                                    />
                                    <span className="label-text">Force restart if already running</span>
                                </label>
                                <button
                                    className="btn btn-info"
                                    onClick={handleRunFinanceSync}
                                    disabled={loadingStatus || anyJobActive}
                                >
                                    {financeActive ? <span className="loading loading-spinner loading-xs"></span> : null}
                                    {financeActive
                                        ? 'Syncing...'
                                        : anyJobActive
                                            ? 'Waiting for current job...'
                                            : 'Run Finance Sync'}
                                </button>
                            </div>
                        </div>
                    </section>
                ) : null}
            </main>
        </div>
    );
};

export default AdminPage;
