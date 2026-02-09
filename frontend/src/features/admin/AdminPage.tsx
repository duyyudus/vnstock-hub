import React, { useEffect, useMemo, useState } from 'react';
import { AuthWidget } from '../auth/AuthWidget';
import { useAuthUser } from '../auth/useAuthUser';
import {
    stockApi,
    type PriceBootstrapDetailedStatusResponse,
    type PriceSyncActionResponse,
    type SyncStatusResponse,
} from '../../api/stockApi';

const REFRESH_INTERVAL_MS = 5000;

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
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
        return value;
    }
    return date.toLocaleString();
};

export const AdminPage: React.FC = () => {
    const user = useAuthUser();

    const [syncStatus, setSyncStatus] = useState<SyncStatusResponse | null>(null);
    const [bootstrapStatus, setBootstrapStatus] = useState<PriceBootstrapDetailedStatusResponse | null>(null);
    const [loadingStatus, setLoadingStatus] = useState(false);
    const [statusError, setStatusError] = useState<string | null>(null);

    const [forceRestart, setForceRestart] = useState(false);
    const [healWindowDays, setHealWindowDays] = useState(7);
    const [repairSymbols, setRepairSymbols] = useState('');
    const [repairStartDate, setRepairStartDate] = useState('');
    const [repairEndDate, setRepairEndDate] = useState('');

    const [bootstrapRunning, setBootstrapRunning] = useState(false);
    const [incrementalRunning, setIncrementalRunning] = useState(false);
    const [repairRunning, setRepairRunning] = useState(false);
    const [actionMessage, setActionMessage] = useState<string | null>(null);
    const [actionError, setActionError] = useState<string | null>(null);

    const canAccess = Boolean(user);

    const loadStatuses = async () => {
        if (!canAccess) {
            return;
        }

        setLoadingStatus(true);
        setStatusError(null);

        try {
            const [sync, bootstrap] = await Promise.all([
                stockApi.getSyncStatus(),
                stockApi.getPriceBootstrapStatus(),
            ]);
            setSyncStatus(sync);
            setBootstrapStatus(bootstrap);
        } catch (error) {
            setStatusError(getErrorMessage(error));
        } finally {
            setLoadingStatus(false);
        }
    };

    useEffect(() => {
        loadStatuses();

        if (!canAccess) {
            return;
        }

        const intervalId = window.setInterval(() => {
            loadStatuses();
        }, REFRESH_INTERVAL_MS);

        return () => {
            window.clearInterval(intervalId);
        };
    }, [canAccess]);

    const runAction = async (
        fn: () => Promise<PriceSyncActionResponse>,
        setLoading: React.Dispatch<React.SetStateAction<boolean>>,
    ) => {
        setLoading(true);
        setActionMessage(null);
        setActionError(null);

        try {
            const result = await fn();
            setActionMessage(result.message);
            await loadStatuses();
        } catch (error) {
            setActionError(getErrorMessage(error));
        } finally {
            setLoading(false);
        }
    };

    const handleStartBootstrap = async () => {
        await runAction(
            () => stockApi.startPriceBootstrap(forceRestart),
            setBootstrapRunning,
        );
    };

    const handleRunIncremental = async () => {
        await runAction(
            () => stockApi.runPriceIncrementalSync(healWindowDays),
            setIncrementalRunning,
        );
    };

    const handleRunRepair = async () => {
        const symbols = repairSymbols
            .split(/[\s,]+/)
            .map((symbol) => symbol.trim().toUpperCase())
            .filter(Boolean);

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

    const runtimeBootstrap = syncStatus?.price_sync.bootstrap;
    const runtimeIncremental = syncStatus?.price_sync.incremental;
    const runtimeRepair = syncStatus?.price_sync.repair;

    const bootstrapProgressPercent = useMemo(() => {
        const value = bootstrapStatus?.progress ?? runtimeBootstrap?.progress ?? 0;
        return Math.max(0, Math.min(100, Math.round(value * 100)));
    }, [bootstrapStatus?.progress, runtimeBootstrap?.progress]);

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

                <section className="grid gap-4 md:grid-cols-3">
                    <div className="card bg-base-100 shadow-lg">
                        <div className="card-body">
                            <h2 className="card-title text-base">Bootstrap</h2>
                            <p>State: <strong>{bootstrapStatus?.state ?? runtimeBootstrap?.state ?? '-'}</strong></p>
                            <p>Progress: <strong>{bootstrapProgressPercent}%</strong></p>
                            <progress className="progress progress-primary w-full" value={bootstrapProgressPercent} max={100}></progress>
                            <p>Processed: {bootstrapStatus?.processed_symbols ?? runtimeBootstrap?.processed_symbols ?? 0} / {bootstrapStatus?.total_symbols ?? runtimeBootstrap?.total_symbols ?? 0}</p>
                            <p>Current symbol: {bootstrapStatus?.current_symbol ?? runtimeBootstrap?.current_symbol ?? '-'}</p>
                            <p>Started: {formatDateTime(bootstrapStatus?.started_at ?? runtimeBootstrap?.started_at ?? null)}</p>
                            <p>Completed: {formatDateTime(bootstrapStatus?.completed_at ?? runtimeBootstrap?.completed_at ?? null)}</p>
                            <p>Error: {bootstrapStatus?.error ?? runtimeBootstrap?.error ?? '-'}</p>
                        </div>
                    </div>

                    <div className="card bg-base-100 shadow-lg">
                        <div className="card-body">
                            <h2 className="card-title text-base">Incremental</h2>
                            <p>Running: <strong>{runtimeIncremental?.is_running ? 'Yes' : 'No'}</strong></p>
                            <p>Processed: {runtimeIncremental?.processed_symbols ?? 0} / {runtimeIncremental?.total_symbols ?? 0}</p>
                            <p>Current symbol: {runtimeIncremental?.current_symbol ?? '-'}</p>
                            <p>Last run: {formatDateTime(runtimeIncremental?.last_run_at)}</p>
                            <p>Started: {formatDateTime(runtimeIncremental?.started_at)}</p>
                            <p>Error: {runtimeIncremental?.error ?? '-'}</p>
                        </div>
                    </div>

                    <div className="card bg-base-100 shadow-lg">
                        <div className="card-body">
                            <h2 className="card-title text-base">Repair</h2>
                            <p>Running: <strong>{runtimeRepair?.is_running ? 'Yes' : 'No'}</strong></p>
                            <p>Processed: {runtimeRepair?.processed_symbols ?? 0} / {runtimeRepair?.total_symbols ?? 0}</p>
                            <p>Current symbol: {runtimeRepair?.current_symbol ?? '-'}</p>
                            <p>Last run: {formatDateTime(runtimeRepair?.last_run_at)}</p>
                            <p>Started: {formatDateTime(runtimeRepair?.started_at)}</p>
                            <p>Error: {runtimeRepair?.error ?? '-'}</p>
                        </div>
                    </div>
                </section>

                <section className="grid gap-4 lg:grid-cols-3">
                    <div className="card bg-base-100 shadow-lg">
                        <div className="card-body space-y-3">
                            <h2 className="card-title">Start Bootstrap</h2>
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
                                className={`btn btn-primary ${bootstrapRunning ? 'loading' : ''}`}
                                onClick={handleStartBootstrap}
                                disabled={bootstrapRunning || loadingStatus}
                            >
                                Start Bootstrap
                            </button>
                        </div>
                    </div>

                    <div className="card bg-base-100 shadow-lg">
                        <div className="card-body space-y-3">
                            <h2 className="card-title">Run Incremental</h2>
                            <label className="form-control">
                                <span className="label-text">Heal window (days)</span>
                                <input
                                    type="number"
                                    min={1}
                                    max={60}
                                    className="input input-bordered"
                                    value={healWindowDays}
                                    onChange={(event) => setHealWindowDays(Number(event.target.value) || 7)}
                                />
                            </label>
                            <button
                                className={`btn btn-secondary ${incrementalRunning ? 'loading' : ''}`}
                                onClick={handleRunIncremental}
                                disabled={incrementalRunning || loadingStatus}
                            >
                                Run Incremental Sync
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
                                className={`btn btn-accent ${repairRunning ? 'loading' : ''}`}
                                onClick={handleRunRepair}
                                disabled={repairRunning || loadingStatus}
                            >
                                Run Repair Sync
                            </button>
                        </div>
                    </div>
                </section>

                <section className="card bg-base-100 shadow-lg">
                    <div className="card-body">
                        <h2 className="card-title">Bootstrap DB Summary</h2>
                        <div className="overflow-x-auto">
                            <table className="table table-zebra">
                                <tbody>
                                    <tr>
                                        <td>Total symbols</td>
                                        <td>{bootstrapStatus?.db_summary?.total_symbols ?? '-'}</td>
                                    </tr>
                                    {Object.entries(bootstrapStatus?.db_summary?.by_status ?? {}).map(([statusKey, count]) => (
                                        <tr key={statusKey}>
                                            <td>{statusKey}</td>
                                            <td>{String(count)}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </section>
            </main>
        </div>
    );
};

export default AdminPage;
