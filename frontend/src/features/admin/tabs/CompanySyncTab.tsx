import React from 'react';
import { type IndexInfo, type HistoryJobStatus } from '../../../api/stockApi';
import { SyncCollectionSelector } from '../components/SyncCollectionSelector';
import { FailedTickerList } from '../components/FailedTickerList';
import { formatDateTime, type SyncCollectionScope } from '../adminUtils';

interface CompanySyncTabProps {
    runtimeCompany: HistoryJobStatus | undefined;
    companyProgressPercent: number;
    indexOptions: IndexInfo[];
    companyIndexSymbol: string;
    onCompanyIndexSymbolChange: (value: string) => void;
    companyCollectionScope: SyncCollectionScope;
    onCompanyCollectionScopeChange: (value: SyncCollectionScope) => void;
    companySymbols: string;
    onCompanySymbolsChange: (value: string) => void;
    companyForceRestart: boolean;
    onCompanyForceRestartChange: (checked: boolean) => void;
    companyQuickSync: boolean;
    onCompanyQuickSyncChange: (checked: boolean) => void;
    companyForceRefresh: boolean;
    onCompanyForceRefreshChange: (checked: boolean) => void;
    onRunCompanySync: () => void | Promise<void>;
    companyActive: boolean;
    anyJobActive: boolean;
    actionDisabled: boolean;
    portfolioCollectionCount: number;
    tradingCollectionCount: number;
}

export const CompanySyncTab: React.FC<CompanySyncTabProps> = ({
    runtimeCompany,
    companyProgressPercent,
    indexOptions,
    companyIndexSymbol,
    onCompanyIndexSymbolChange,
    companyCollectionScope,
    onCompanyCollectionScopeChange,
    companySymbols,
    onCompanySymbolsChange,
    companyForceRestart,
    onCompanyForceRestartChange,
    companyQuickSync,
    onCompanyQuickSyncChange,
    companyForceRefresh,
    onCompanyForceRefreshChange,
    onRunCompanySync,
    companyActive,
    anyJobActive,
    actionDisabled,
    portfolioCollectionCount,
    tradingCollectionCount,
}) => {
    return (
        <section className="grid gap-4 lg:grid-cols-2">
            <div className="card bg-base-100 shadow-lg">
                <div className="card-body">
                    <h2 className="card-title text-base">Company Sync Status</h2>
                    <p>Running: <strong>{runtimeCompany?.is_running ? 'Yes' : 'No'}</strong></p>
                    <p>Progress: <strong>{companyProgressPercent}%</strong></p>
                    <progress className="progress progress-warning w-full" value={companyProgressPercent} max={100}></progress>
                    <p>Processed: {runtimeCompany?.processed_symbols ?? 0} / {runtimeCompany?.total_symbols ?? 0}</p>
                    <p>Success: {runtimeCompany?.success_symbols ?? 0}</p>
                    <p>Failed: {runtimeCompany?.failed_symbols ?? 0}</p>
                    <FailedTickerList tickers={runtimeCompany?.failed_tickers ?? []} />
                    <p>Current symbol: {runtimeCompany?.current_symbol ?? '-'}</p>
                    <p>Last run: {formatDateTime(runtimeCompany?.last_run_at)}</p>
                    <p>Started: {formatDateTime(runtimeCompany?.started_at)}</p>
                    <p>Error: {runtimeCompany?.error ?? '-'}</p>
                </div>
            </div>

            <div className="card bg-base-100 shadow-lg">
                <div className="card-body space-y-3">
                    <h2 className="card-title">Run Company Sync</h2>
                    <SyncCollectionSelector
                        value={companyCollectionScope}
                        onChange={onCompanyCollectionScopeChange}
                        portfolioCount={portfolioCollectionCount}
                        tradingCount={tradingCollectionCount}
                    />
                    <label className="form-control">
                        <span className="label-text">Index scope (optional)</span>
                        <select
                            className="select select-bordered"
                            value={companyIndexSymbol}
                            onChange={(event) => onCompanyIndexSymbolChange(event.target.value)}
                            disabled={companyCollectionScope !== 'manual'}
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
                            value={companySymbols}
                            onChange={(event) => onCompanySymbolsChange(event.target.value)}
                            placeholder="All symbols if empty"
                            disabled={companyCollectionScope !== 'manual'}
                        />
                    </label>
                    <label className="label cursor-pointer justify-start gap-3">
                        <input
                            type="checkbox"
                            className="checkbox"
                            checked={companyForceRestart}
                            onChange={(event) => onCompanyForceRestartChange(event.target.checked)}
                        />
                        <span className="label-text">Force restart if already running</span>
                    </label>
                    <label className="label cursor-pointer justify-start gap-3">
                        <input
                            type="checkbox"
                            className="checkbox"
                            checked={companyQuickSync}
                            onChange={(event) => onCompanyQuickSyncChange(event.target.checked)}
                        />
                        <span className="label-text">Quick sync</span>
                    </label>
                    <label className="label cursor-pointer justify-start gap-3">
                        <input
                            type="checkbox"
                            className="checkbox"
                            checked={companyForceRefresh}
                            onChange={(event) => onCompanyForceRefreshChange(event.target.checked)}
                        />
                        <span className="label-text">Force refresh cached datasets</span>
                    </label>
                    <button
                        className="btn btn-warning"
                        onClick={onRunCompanySync}
                        disabled={actionDisabled}
                    >
                        {companyActive ? <span className="loading loading-spinner loading-xs"></span> : null}
                        {companyActive
                            ? 'Syncing...'
                            : anyJobActive
                                ? 'Waiting for current job...'
                                : 'Run Company Sync'}
                    </button>
                </div>
            </div>
        </section>
    );
};
