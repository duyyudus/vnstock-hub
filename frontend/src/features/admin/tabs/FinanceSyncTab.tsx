import React from 'react';
import { type IndexInfo, type PriceJobStatus } from '../../../api/stockApi';
import { formatDateTime } from '../adminUtils';

interface FinanceSyncTabProps {
    runtimeFinance: PriceJobStatus | undefined;
    financeProgressPercent: number;
    indexOptions: IndexInfo[];
    financeIndexSymbol: string;
    onFinanceIndexSymbolChange: (value: string) => void;
    financeSymbols: string;
    onFinanceSymbolsChange: (value: string) => void;
    financeForceRestart: boolean;
    onFinanceForceRestartChange: (checked: boolean) => void;
    financeQuickSync: boolean;
    onFinanceQuickSyncChange: (checked: boolean) => void;
    onRunFinanceSync: () => void | Promise<void>;
    financeActive: boolean;
    anyJobActive: boolean;
    actionDisabled: boolean;
}

export const FinanceSyncTab: React.FC<FinanceSyncTabProps> = ({
    runtimeFinance,
    financeProgressPercent,
    indexOptions,
    financeIndexSymbol,
    onFinanceIndexSymbolChange,
    financeSymbols,
    onFinanceSymbolsChange,
    financeForceRestart,
    onFinanceForceRestartChange,
    financeQuickSync,
    onFinanceQuickSyncChange,
    onRunFinanceSync,
    financeActive,
    anyJobActive,
    actionDisabled,
}) => {
    return (
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
                            onChange={(event) => onFinanceIndexSymbolChange(event.target.value)}
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
                            onChange={(event) => onFinanceSymbolsChange(event.target.value)}
                            placeholder="All symbols if empty"
                        />
                    </label>
                    <label className="label cursor-pointer justify-start gap-3">
                        <input
                            type="checkbox"
                            className="checkbox"
                            checked={financeForceRestart}
                            onChange={(event) => onFinanceForceRestartChange(event.target.checked)}
                        />
                        <span className="label-text">Force restart if already running</span>
                    </label>
                    <label className="label cursor-pointer justify-start gap-3">
                        <input
                            type="checkbox"
                            className="checkbox"
                            checked={financeQuickSync}
                            onChange={(event) => onFinanceQuickSyncChange(event.target.checked)}
                        />
                        <span className="label-text">Quick sync</span>
                    </label>
                    <button
                        className="btn btn-info"
                        onClick={onRunFinanceSync}
                        disabled={actionDisabled}
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
    );
};
