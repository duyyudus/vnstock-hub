import React from 'react';
import { type IndexInfo, type HistoryAuditActionResponse, type HistoryJobStatus } from '../../../api/stockApi';
import { FailedTickerList } from '../components/FailedTickerList';
import { formatDateTime } from '../adminUtils';

interface HistorySyncTabProps {
    runtimeSync: HistoryJobStatus | undefined;
    runtimeAudit: HistoryJobStatus | undefined;
    runtimeRepair: HistoryJobStatus | undefined;
    syncProgressPercent: number;
    auditProgressPercent: number;
    repairProgressPercent: number;
    indexOptions: IndexInfo[];
    syncIndexSymbol: string;
    onSyncIndexSymbolChange: (value: string) => void;
    syncSymbols: string;
    onSyncSymbolsChange: (value: string) => void;
    forceRestart: boolean;
    onForceRestartChange: (checked: boolean) => void;
    onRunSync: () => void | Promise<void>;
    syncActive: boolean;
    auditIndexSymbol: string;
    onAuditIndexSymbolChange: (value: string) => void;
    auditSymbols: string;
    onAuditSymbolsChange: (value: string) => void;
    auditStartDate: string;
    onAuditStartDateChange: (value: string) => void;
    auditEndDate: string;
    onAuditEndDateChange: (value: string) => void;
    auditAutoRepair: boolean;
    onAuditAutoRepairChange: (checked: boolean) => void;
    onRunAudit: () => void | Promise<void>;
    auditActive: boolean;
    repairIndexSymbol: string;
    onRepairIndexSymbolChange: (value: string) => void;
    repairSymbols: string;
    onRepairSymbolsChange: (value: string) => void;
    repairStartDate: string;
    onRepairStartDateChange: (value: string) => void;
    repairEndDate: string;
    onRepairEndDateChange: (value: string) => void;
    onRunRepair: () => void | Promise<void>;
    repairActive: boolean;
    anyJobActive: boolean;
    actionDisabled: boolean;
    auditResult: HistoryAuditActionResponse | null;
}

export const HistorySyncTab: React.FC<HistorySyncTabProps> = ({
    runtimeSync,
    runtimeAudit,
    runtimeRepair,
    syncProgressPercent,
    auditProgressPercent,
    repairProgressPercent,
    indexOptions,
    syncIndexSymbol,
    onSyncIndexSymbolChange,
    syncSymbols,
    onSyncSymbolsChange,
    forceRestart,
    onForceRestartChange,
    onRunSync,
    syncActive,
    auditIndexSymbol,
    onAuditIndexSymbolChange,
    auditSymbols,
    onAuditSymbolsChange,
    auditStartDate,
    onAuditStartDateChange,
    auditEndDate,
    onAuditEndDateChange,
    auditAutoRepair,
    onAuditAutoRepairChange,
    onRunAudit,
    auditActive,
    repairIndexSymbol,
    onRepairIndexSymbolChange,
    repairSymbols,
    onRepairSymbolsChange,
    repairStartDate,
    onRepairStartDateChange,
    repairEndDate,
    onRepairEndDateChange,
    onRunRepair,
    repairActive,
    anyJobActive,
    actionDisabled,
    auditResult,
}) => {
    return (
        <>
            <section className="grid gap-4 md:grid-cols-3">
                <div className="card bg-base-100 shadow-lg">
                    <div className="card-body">
                        <h2 className="card-title text-base">History Sync Status</h2>
                        <p>Running: <strong>{runtimeSync?.is_running ? 'Yes' : 'No'}</strong></p>
                        <p>Progress: <strong>{syncProgressPercent}%</strong></p>
                        <progress className="progress progress-primary w-full" value={syncProgressPercent} max={100}></progress>
                        <p>Processed: {runtimeSync?.processed_symbols ?? 0} / {runtimeSync?.total_symbols ?? 0}</p>
                        <p>Success: {runtimeSync?.success_symbols ?? 0}</p>
                        <p>Failed: {runtimeSync?.failed_symbols ?? 0}</p>
                        <FailedTickerList tickers={runtimeSync?.failed_tickers ?? []} />
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
                        <p>Progress: <strong>{auditProgressPercent}%</strong></p>
                        <progress className="progress progress-primary w-full" value={auditProgressPercent} max={100}></progress>
                        <p>Processed: {runtimeAudit?.processed_symbols ?? 0} / {runtimeAudit?.total_symbols ?? 0}</p>
                        <p>Success: {runtimeAudit?.success_symbols ?? 0}</p>
                        <p>Failed: {runtimeAudit?.failed_symbols ?? 0}</p>
                        <FailedTickerList tickers={runtimeAudit?.failed_tickers ?? []} />
                        <p>Current symbol: {runtimeAudit?.current_symbol ?? '-'}</p>
                        <p>Last run: {formatDateTime(runtimeAudit?.last_run_at)}</p>
                        <p>Error: {runtimeAudit?.error ?? '-'}</p>
                    </div>
                </div>

                <div className="card bg-base-100 shadow-lg">
                    <div className="card-body">
                        <h2 className="card-title text-base">Repair Status</h2>
                        <p>Running: <strong>{runtimeRepair?.is_running ? 'Yes' : 'No'}</strong></p>
                        <p>Progress: <strong>{repairProgressPercent}%</strong></p>
                        <progress className="progress progress-primary w-full" value={repairProgressPercent} max={100}></progress>
                        <p>Processed: {runtimeRepair?.processed_symbols ?? 0} / {runtimeRepair?.total_symbols ?? 0}</p>
                        <p>Success: {runtimeRepair?.success_symbols ?? 0}</p>
                        <p>Failed: {runtimeRepair?.failed_symbols ?? 0}</p>
                        <FailedTickerList tickers={runtimeRepair?.failed_tickers ?? []} />
                        <p>Current symbol: {runtimeRepair?.current_symbol ?? '-'}</p>
                        <p>Last run: {formatDateTime(runtimeRepair?.last_run_at)}</p>
                        <p>Error: {runtimeRepair?.error ?? '-'}</p>
                    </div>
                </div>
            </section>

            <section className="grid gap-4 lg:grid-cols-3">
                <div className="card bg-base-100 shadow-lg">
                    <div className="card-body space-y-3">
                        <h2 className="card-title">Run History Sync</h2>
                        <label className="form-control">
                            <span className="label-text">Index scope (optional)</span>
                            <select
                                className="select select-bordered"
                                value={syncIndexSymbol}
                                onChange={(event) => onSyncIndexSymbolChange(event.target.value)}
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
                                onChange={(event) => onSyncSymbolsChange(event.target.value)}
                                placeholder="All symbols if empty"
                            />
                        </label>
                        <label className="label cursor-pointer justify-start gap-3">
                            <input
                                type="checkbox"
                                className="checkbox"
                                checked={forceRestart}
                                onChange={(event) => onForceRestartChange(event.target.checked)}
                            />
                            <span className="label-text">Force restart if already running</span>
                        </label>
                        <button
                            className="btn btn-primary"
                            onClick={onRunSync}
                            disabled={actionDisabled}
                        >
                            {syncActive ? <span className="loading loading-spinner loading-xs"></span> : null}
                            {syncActive
                                ? 'Syncing...'
                                : anyJobActive
                                    ? 'Waiting for current job...'
                                    : 'Run History Sync'}
                        </button>
                    </div>
                </div>

                <div className="card bg-base-100 shadow-lg">
                    <div className="card-body space-y-3">
                        <h2 className="card-title">Run History Audit</h2>
                        <label className="form-control">
                            <span className="label-text">Index scope (optional)</span>
                            <select
                                className="select select-bordered"
                                value={auditIndexSymbol}
                                onChange={(event) => onAuditIndexSymbolChange(event.target.value)}
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
                                onChange={(event) => onAuditSymbolsChange(event.target.value)}
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
                                    onChange={(event) => onAuditStartDateChange(event.target.value)}
                                />
                            </label>
                            <label className="form-control">
                                <span className="label-text">End date</span>
                                <input
                                    type="date"
                                    className="input input-bordered"
                                    value={auditEndDate}
                                    onChange={(event) => onAuditEndDateChange(event.target.value)}
                                />
                            </label>
                        </div>
                        <label className="label cursor-pointer justify-start gap-3">
                            <input
                                type="checkbox"
                                className="checkbox"
                                checked={auditAutoRepair}
                                onChange={(event) => onAuditAutoRepairChange(event.target.checked)}
                            />
                            <span className="label-text">Auto repair detected gaps</span>
                        </label>
                        <button
                            className="btn btn-secondary"
                            onClick={onRunAudit}
                            disabled={actionDisabled}
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
                            <span className="label-text">Index scope (optional)</span>
                            <select
                                className="select select-bordered"
                                value={repairIndexSymbol}
                                onChange={(event) => onRepairIndexSymbolChange(event.target.value)}
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
                                value={repairSymbols}
                                onChange={(event) => onRepairSymbolsChange(event.target.value)}
                                placeholder="All symbols in selected index if empty"
                            />
                        </label>
                        <div className="grid grid-cols-2 gap-2">
                            <label className="form-control">
                                <span className="label-text">Start date</span>
                                <input
                                    type="date"
                                    className="input input-bordered"
                                    value={repairStartDate}
                                    onChange={(event) => onRepairStartDateChange(event.target.value)}
                                />
                            </label>
                            <label className="form-control">
                                <span className="label-text">End date</span>
                                <input
                                    type="date"
                                    className="input input-bordered"
                                    value={repairEndDate}
                                    onChange={(event) => onRepairEndDateChange(event.target.value)}
                                />
                            </label>
                        </div>
                        <button
                            className="btn btn-accent"
                            onClick={onRunRepair}
                            disabled={actionDisabled}
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
    );
};
