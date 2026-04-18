import React from 'react';
import { type IndexInfo, type HistoryJobStatus } from '../../../api/stockApi';
import { SyncCollectionSelector } from '../components/SyncCollectionSelector';
import { FailedTickerList } from '../components/FailedTickerList';
import { formatDateTime, type SyncCollectionScope } from '../adminUtils';

interface FinanceSyncTabProps {
    runtimeFinance: HistoryJobStatus | undefined;
    financeProgressPercent: number;
    indexOptions: IndexInfo[];
    financeIndexSymbol: string;
    onFinanceIndexSymbolChange: (value: string) => void;
    financeCollectionScope: SyncCollectionScope;
    onFinanceCollectionScopeChange: (value: SyncCollectionScope) => void;
    financeSymbols: string;
    onFinanceSymbolsChange: (value: string) => void;
    financeForceRestart: boolean;
    onFinanceForceRestartChange: (checked: boolean) => void;
    financeQuickSync: boolean;
    onFinanceQuickSyncChange: (checked: boolean) => void;
    financeForceRefresh: boolean;
    onFinanceForceRefreshChange: (checked: boolean) => void;
    onRunFinanceSync: () => void | Promise<void>;
    financeActive: boolean;
    anyJobActive: boolean;
    actionDisabled: boolean;
    portfolioCollectionCount: number;
    tradingCollectionCount: number;
}

export const FinanceSyncTab: React.FC<FinanceSyncTabProps> = ({
    runtimeFinance,
    financeProgressPercent,
    indexOptions,
    financeIndexSymbol,
    onFinanceIndexSymbolChange,
    financeCollectionScope,
    onFinanceCollectionScopeChange,
    financeSymbols,
    onFinanceSymbolsChange,
    financeForceRestart,
    onFinanceForceRestartChange,
    financeQuickSync,
    onFinanceQuickSyncChange,
    financeForceRefresh,
    onFinanceForceRefreshChange,
    onRunFinanceSync,
    financeActive,
    anyJobActive,
    actionDisabled,
    portfolioCollectionCount,
    tradingCollectionCount,
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
                    <FailedTickerList tickers={runtimeFinance?.failed_tickers ?? []} />
                    <p>Current symbol: {runtimeFinance?.current_symbol ?? '-'}</p>
                    <p>Last run: {formatDateTime(runtimeFinance?.last_run_at)}</p>
                    <p>Started: {formatDateTime(runtimeFinance?.started_at)}</p>
                    <p>Error: {runtimeFinance?.error ?? '-'}</p>
                </div>
            </div>

            <div className="card bg-base-100 shadow-lg">
                <div className="card-body space-y-3">
                    <h2 className="card-title">Run Finance Sync</h2>
                    <SyncCollectionSelector
                        value={financeCollectionScope}
                        onChange={onFinanceCollectionScopeChange}
                        portfolioCount={portfolioCollectionCount}
                        tradingCount={tradingCollectionCount}
                    />
                    <label className="form-control">
                        <span className="label-text">Index scope (optional)</span>
                        <select
                            className="select select-bordered"
                            value={financeIndexSymbol}
                            onChange={(event) => onFinanceIndexSymbolChange(event.target.value)}
                            disabled={financeCollectionScope !== 'manual'}
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
                            disabled={financeCollectionScope !== 'manual'}
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
                    <label className="label cursor-pointer justify-start gap-3">
                        <input
                            type="checkbox"
                            className="checkbox"
                            checked={financeForceRefresh}
                            onChange={(event) => onFinanceForceRefreshChange(event.target.checked)}
                        />
                        <span className="label-text">Force refresh cached finance data</span>
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
