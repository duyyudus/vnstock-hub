import React from 'react';
import { type IndexInfo, type PriceJobStatus } from '../../../api/stockApi';
import { formatDateTime } from '../adminUtils';

interface CompanySyncTabProps {
    runtimeCompany: PriceJobStatus | undefined;
    companyProgressPercent: number;
    indexOptions: IndexInfo[];
    companyIndexSymbol: string;
    onCompanyIndexSymbolChange: (value: string) => void;
    companySymbols: string;
    onCompanySymbolsChange: (value: string) => void;
    companyForceRestart: boolean;
    onCompanyForceRestartChange: (checked: boolean) => void;
    companyQuickSync: boolean;
    onCompanyQuickSyncChange: (checked: boolean) => void;
    onRunCompanySync: () => void | Promise<void>;
    companyActive: boolean;
    anyJobActive: boolean;
    actionDisabled: boolean;
}

export const CompanySyncTab: React.FC<CompanySyncTabProps> = ({
    runtimeCompany,
    companyProgressPercent,
    indexOptions,
    companyIndexSymbol,
    onCompanyIndexSymbolChange,
    companySymbols,
    onCompanySymbolsChange,
    companyForceRestart,
    onCompanyForceRestartChange,
    companyQuickSync,
    onCompanyQuickSyncChange,
    onRunCompanySync,
    companyActive,
    anyJobActive,
    actionDisabled,
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
                    <p>Current symbol: {runtimeCompany?.current_symbol ?? '-'}</p>
                    <p>Last run: {formatDateTime(runtimeCompany?.last_run_at)}</p>
                    <p>Started: {formatDateTime(runtimeCompany?.started_at)}</p>
                    <p>Error: {runtimeCompany?.error ?? '-'}</p>
                </div>
            </div>

            <div className="card bg-base-100 shadow-lg">
                <div className="card-body space-y-3">
                    <h2 className="card-title">Run Company Sync</h2>
                    <label className="form-control">
                        <span className="label-text">Index scope (optional)</span>
                        <select
                            className="select select-bordered"
                            value={companyIndexSymbol}
                            onChange={(event) => onCompanyIndexSymbolChange(event.target.value)}
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
