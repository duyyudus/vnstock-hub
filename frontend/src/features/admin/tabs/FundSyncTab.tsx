import React from 'react';
import { type FundSyncCategory, type SyncStatusItem } from '../../../api/stockApi';
import { formatDateTime } from '../adminUtils';

interface FundSyncTabProps {
    runtimeFund: SyncStatusItem | undefined;
    fundProgressPercent: number;
    fundCategory: FundSyncCategory;
    onFundCategoryChange: (value: FundSyncCategory) => void;
    onRunFundSync: () => void | Promise<void>;
    fundActive: boolean;
    anyJobActive: boolean;
    actionDisabled: boolean;
}

const FUND_CATEGORY_OPTIONS: Array<{ value: FundSyncCategory; label: string }> = [
    { value: 'ALL', label: 'All' },
    { value: 'STOCK', label: 'Stock' },
    { value: 'BOND', label: 'Bond' },
    { value: 'BALANCED', label: 'Balanced' },
];

export const FundSyncTab: React.FC<FundSyncTabProps> = ({
    runtimeFund,
    fundProgressPercent,
    fundCategory,
    onFundCategoryChange,
    onRunFundSync,
    fundActive,
    anyJobActive,
    actionDisabled,
}) => {
    return (
        <section className="grid gap-4 lg:grid-cols-2">
            <div className="card bg-base-100 shadow-lg">
                <div className="card-body">
                    <h2 className="card-title text-base">Fund Sync Status</h2>
                    <p>Running: <strong>{runtimeFund?.is_syncing ? 'Yes' : 'No'}</strong></p>
                    <p>Progress: <strong>{fundProgressPercent}%</strong></p>
                    <progress className="progress progress-success w-full" value={fundProgressPercent} max={100}></progress>
                    <p>Processed: {runtimeFund?.processed_symbols ?? 0} / {runtimeFund?.total_symbols ?? 0}</p>
                    <p>Last sync: {formatDateTime(runtimeFund?.last_sync)}</p>
                    <p>Started: {formatDateTime(runtimeFund?.started_at)}</p>
                    <p>Error: {runtimeFund?.error ?? '-'}</p>
                </div>
            </div>

            <div className="card bg-base-100 shadow-lg">
                <div className="card-body space-y-3">
                    <h2 className="card-title">Run Fund Sync</h2>
                    <label className="form-control">
                        <span className="label-text">Fund category</span>
                        <select
                            className="select select-bordered"
                            value={fundCategory}
                            onChange={(event) => onFundCategoryChange(event.target.value as FundSyncCategory)}
                            disabled={fundActive}
                        >
                            {FUND_CATEGORY_OPTIONS.map((option) => (
                                <option key={option.value} value={option.value}>
                                    {option.label}
                                </option>
                            ))}
                        </select>
                    </label>
                    <button
                        className="btn btn-success"
                        onClick={onRunFundSync}
                        disabled={actionDisabled}
                    >
                        {fundActive ? <span className="loading loading-spinner loading-xs"></span> : null}
                        {fundActive
                            ? 'Syncing...'
                            : anyJobActive
                                ? 'Waiting for current job...'
                                : 'Run Fund Sync'}
                    </button>
                </div>
            </div>
        </section>
    );
};
