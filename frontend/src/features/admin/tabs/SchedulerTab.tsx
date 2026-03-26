import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
    type IndexInfo,
    type ScheduledSyncAction,
    type ScheduledSyncIntervalUnit,
    type ScheduledSyncJob,
    type ScheduledSyncJobCreateRequest,
    type ScheduledSyncJobRun,
    type ScheduledSyncType,
    type ScheduledSyncJobUpdateRequest,
    stockApi,
} from '../../../api/stockApi';
import {
    formatDateTimeInTimezone,
    getErrorMessage,
    parseSymbolsInput,
    toDateTimeLocalValue,
} from '../adminUtils';

const SCHEDULER_TIMEZONE = 'Asia/Ho_Chi_Minh';
const REFRESH_INTERVAL_MS = 5000;

type SchedulerFormState = {
    name: string;
    syncType: ScheduledSyncType;
    syncAction: ScheduledSyncAction;
    indexSymbol: string;
    symbolsText: string;
    dateFrom: string;
    dateTo: string;
    autoRepair: boolean;
    startsAt: string;
    intervalValue: string;
    intervalUnit: ScheduledSyncIntervalUnit;
    maxRetries: string;
    enabled: boolean;
};

const DEFAULT_FORM_STATE: SchedulerFormState = {
    name: '',
    syncType: 'history',
    syncAction: 'sync',
    indexSymbol: '',
    symbolsText: '',
    dateFrom: '',
    dateTo: '',
    autoRepair: false,
    startsAt: '',
    intervalValue: '1',
    intervalUnit: 'days',
    maxRetries: '0',
    enabled: true,
};

const ACTION_OPTIONS: Record<ScheduledSyncType, Array<{ value: ScheduledSyncAction; label: string }>> = {
    history: [
        { value: 'sync', label: 'Normal sync' },
        { value: 'audit', label: 'Gap audit' },
        { value: 'repair', label: 'Repair sync' },
    ],
    finance: [
        { value: 'full', label: 'Full sync' },
        { value: 'quick', label: 'Quick sync' },
    ],
    company: [
        { value: 'full', label: 'Full sync' },
        { value: 'quick', label: 'Quick sync' },
    ],
};

const isDateRangeAction = (syncType: ScheduledSyncType, syncAction: ScheduledSyncAction) => {
    return syncType === 'history' && (syncAction === 'audit' || syncAction === 'repair');
};

const isAutoRepairAction = (syncType: ScheduledSyncType, syncAction: ScheduledSyncAction) => {
    return syncType === 'history' && syncAction === 'audit';
};

const describeScope = (job: ScheduledSyncJob) => {
    const parts: string[] = [];
    if (job.index_symbol) {
        parts.push(job.index_symbol);
    }
    if (job.symbols.length > 0) {
        parts.push(job.symbols.join(', '));
    }
    if (parts.length === 0) {
        parts.push('All symbols');
    }
    if (job.date_from && job.date_to) {
        parts.push(`${job.date_from} to ${job.date_to}`);
    }
    return parts.join(' | ');
};

const summarizeRun = (run: ScheduledSyncJobRun) => {
    if (run.error) {
        return run.error;
    }
    const processed = run.summary.processed_symbols;
    const success = run.summary.success_symbols;
    const failed = run.summary.failed_symbols;
    if (typeof processed === 'number' || typeof success === 'number' || typeof failed === 'number') {
        return `Processed ${processed ?? 0}, success ${success ?? 0}, failed ${failed ?? 0}`;
    }
    const message = run.summary.message;
    if (typeof message === 'string' && message.trim()) {
        return message;
    }
    return '-';
};

const getStatusBadgeClass = (status: ScheduledSyncJobRun['status']) => {
    switch (status) {
        case 'running':
            return 'badge-info';
        case 'succeeded':
            return 'badge-success';
        case 'failed':
            return 'badge-error';
        default:
            return 'badge-ghost';
    }
};

interface SchedulerTabProps {
    indexOptions: IndexInfo[];
}

export const SchedulerTab: React.FC<SchedulerTabProps> = ({ indexOptions }) => {
    const [jobs, setJobs] = useState<ScheduledSyncJob[]>([]);
    const [runs, setRuns] = useState<ScheduledSyncJobRun[]>([]);
    const [form, setForm] = useState<SchedulerFormState>(DEFAULT_FORM_STATE);
    const [editingJobId, setEditingJobId] = useState<number | null>(null);
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [busyJobId, setBusyJobId] = useState<number | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState<string | null>(null);

    const loadSchedulerData = useCallback(async (showSpinner: boolean = false) => {
        if (showSpinner) {
            setLoading(true);
        }
        try {
            const [jobsResponse, runsResponse] = await Promise.all([
                stockApi.getScheduledSyncJobs(),
                stockApi.getScheduledSyncRuns(20),
            ]);
            setJobs(jobsResponse.jobs);
            setRuns(runsResponse.runs);
            setError(null);
        } catch (err) {
            setError(getErrorMessage(err));
        } finally {
            if (showSpinner) {
                setLoading(false);
            }
        }
    }, []);

    useEffect(() => {
        void loadSchedulerData(true);
        const intervalId = window.setInterval(() => {
            void loadSchedulerData(false);
        }, REFRESH_INTERVAL_MS);
        return () => {
            window.clearInterval(intervalId);
        };
    }, [loadSchedulerData]);

    const actionOptions = useMemo(() => ACTION_OPTIONS[form.syncType], [form.syncType]);
    const showDateRange = isDateRangeAction(form.syncType, form.syncAction);
    const showAutoRepair = isAutoRepairAction(form.syncType, form.syncAction);

    useEffect(() => {
        if (!actionOptions.some((option) => option.value === form.syncAction)) {
            setForm((current) => ({
                ...current,
                syncAction: actionOptions[0].value,
                autoRepair: actionOptions[0].value === 'audit' ? current.autoRepair : false,
            }));
        }
    }, [actionOptions, form.syncAction]);

    const resetForm = useCallback(() => {
        setForm(DEFAULT_FORM_STATE);
        setEditingJobId(null);
    }, []);

    const handleEdit = useCallback((job: ScheduledSyncJob) => {
        setEditingJobId(job.id);
        setSuccess(null);
        setError(null);
        setForm({
            name: job.name,
            syncType: job.sync_type,
            syncAction: job.sync_action,
            indexSymbol: job.index_symbol ?? '',
            symbolsText: job.symbols.join(', '),
            dateFrom: job.date_from ?? '',
            dateTo: job.date_to ?? '',
            autoRepair: job.auto_repair,
            startsAt: toDateTimeLocalValue(job.starts_at, SCHEDULER_TIMEZONE),
            intervalValue: String(job.interval_value),
            intervalUnit: job.interval_unit,
            maxRetries: String(job.max_retries),
            enabled: job.enabled,
        });
    }, []);

    const buildCreatePayload = useCallback((): ScheduledSyncJobCreateRequest => {
        const payload: ScheduledSyncJobCreateRequest = {
            name: form.name.trim(),
            enabled: form.enabled,
            sync_type: form.syncType,
            sync_action: form.syncAction,
            symbols: parseSymbolsInput(form.symbolsText),
            starts_at: form.startsAt,
            interval_value: Number(form.intervalValue),
            interval_unit: form.intervalUnit,
            timezone: SCHEDULER_TIMEZONE,
            max_retries: Number(form.maxRetries),
        };

        if (form.indexSymbol) {
            payload.index_symbol = form.indexSymbol;
        }
        if (showDateRange) {
            payload.date_from = form.dateFrom;
            payload.date_to = form.dateTo;
        }
        if (showAutoRepair) {
            payload.auto_repair = form.autoRepair;
        }
        return payload;
    }, [form, showAutoRepair, showDateRange]);

    const buildUpdatePayload = useCallback((): ScheduledSyncJobUpdateRequest => {
        const payload = buildCreatePayload();
        return {
            ...payload,
            index_symbol: form.indexSymbol || null,
            date_from: showDateRange ? form.dateFrom : null,
            date_to: showDateRange ? form.dateTo : null,
            auto_repair: showAutoRepair ? form.autoRepair : false,
        };
    }, [buildCreatePayload, form.autoRepair, form.dateFrom, form.dateTo, form.indexSymbol, showAutoRepair, showDateRange]);

    const handleSubmit = async () => {
        setSaving(true);
        setError(null);
        setSuccess(null);
        try {
            if (editingJobId === null) {
                await stockApi.createScheduledSyncJob(buildCreatePayload());
                setSuccess('Scheduled job created.');
            } else {
                await stockApi.updateScheduledSyncJob(editingJobId, buildUpdatePayload());
                setSuccess('Scheduled job updated.');
            }
            resetForm();
            await loadSchedulerData(false);
        } catch (err) {
            setError(getErrorMessage(err));
        } finally {
            setSaving(false);
        }
    };

    const handleToggleEnabled = async (job: ScheduledSyncJob) => {
        setBusyJobId(job.id);
        setError(null);
        setSuccess(null);
        try {
            await stockApi.updateScheduledSyncJob(job.id, { enabled: !job.enabled });
            setSuccess(job.enabled ? 'Scheduled job paused.' : 'Scheduled job resumed.');
            await loadSchedulerData(false);
        } catch (err) {
            setError(getErrorMessage(err));
        } finally {
            setBusyJobId(null);
        }
    };

    const handleDelete = async (job: ScheduledSyncJob) => {
        if (!window.confirm(`Delete scheduled job "${job.name}"?`)) {
            return;
        }
        setBusyJobId(job.id);
        setError(null);
        setSuccess(null);
        try {
            await stockApi.deleteScheduledSyncJob(job.id);
            if (editingJobId === job.id) {
                resetForm();
            }
            setSuccess('Scheduled job deleted.');
            await loadSchedulerData(false);
        } catch (err) {
            setError(getErrorMessage(err));
        } finally {
            setBusyJobId(null);
        }
    };

    return (
        <div className="space-y-6">
            {error ? (
                <div className="alert alert-error">
                    <span>{error}</span>
                </div>
            ) : null}

            {success ? (
                <div className="alert alert-success">
                    <span>{success}</span>
                </div>
            ) : null}

            <section className="grid gap-4 xl:grid-cols-[1.2fr_0.8fr]">
                <div className="card bg-base-100 shadow-lg">
                    <div className="card-body space-y-4">
                        <div className="flex flex-wrap items-center justify-between gap-3">
                            <div>
                                <h2 className="card-title">Scheduled Jobs</h2>
                                <p className="text-sm text-base-content/70">
                                    All times use {SCHEDULER_TIMEZONE}.
                                </p>
                            </div>
                            <button className="btn btn-sm btn-ghost" onClick={() => void loadSchedulerData(true)}>
                                Refresh
                            </button>
                        </div>

                        {loading ? (
                            <div className="flex items-center gap-2 text-base-content/70">
                                <span className="loading loading-spinner loading-sm"></span>
                                <span>Loading scheduled jobs...</span>
                            </div>
                        ) : null}

                        {!loading && jobs.length === 0 ? (
                            <div className="rounded-lg border border-dashed border-base-300 p-4 text-sm text-base-content/70">
                                No scheduled jobs yet.
                            </div>
                        ) : null}

                        {jobs.length > 0 ? (
                            <div className="overflow-x-auto">
                                <table className="table table-zebra table-sm">
                                    <thead>
                                        <tr>
                                            <th>Name</th>
                                            <th>Type</th>
                                            <th>Scope</th>
                                            <th>Schedule</th>
                                            <th>Status</th>
                                            <th></th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {jobs.map((job) => (
                                            <tr key={job.id}>
                                                <td>
                                                    <div className="font-medium">{job.name}</div>
                                                    <div className="text-xs text-base-content/60">
                                                        Retries: {job.max_retries}
                                                    </div>
                                                </td>
                                                <td>
                                                    <div>{job.sync_type}</div>
                                                    <div className="text-xs text-base-content/60">{job.sync_action}</div>
                                                </td>
                                                <td className="max-w-xs whitespace-normal text-xs">
                                                    {describeScope(job)}
                                                </td>
                                                <td className="text-xs">
                                                    <div>Start: {formatDateTimeInTimezone(job.starts_at, SCHEDULER_TIMEZONE)}</div>
                                                    <div>Every {job.interval_value} {job.interval_unit}</div>
                                                    <div>Next: {formatDateTimeInTimezone(job.next_run_at, SCHEDULER_TIMEZONE)}</div>
                                                </td>
                                                <td className="text-xs">
                                                    <div className={`badge ${job.enabled ? 'badge-success' : 'badge-ghost'}`}>
                                                        {job.enabled ? 'Enabled' : 'Paused'}
                                                    </div>
                                                    <div className="mt-2">Last: {formatDateTimeInTimezone(job.last_run_at, SCHEDULER_TIMEZONE)}</div>
                                                </td>
                                                <td>
                                                    <div className="flex flex-wrap gap-2">
                                                        <button
                                                            className="btn btn-xs btn-outline"
                                                            onClick={() => handleEdit(job)}
                                                        >
                                                            Edit
                                                        </button>
                                                        <button
                                                            className="btn btn-xs btn-outline"
                                                            onClick={() => void handleToggleEnabled(job)}
                                                            disabled={busyJobId === job.id}
                                                        >
                                                            {job.enabled ? 'Pause' : 'Resume'}
                                                        </button>
                                                        <button
                                                            className="btn btn-xs btn-error btn-outline"
                                                            onClick={() => void handleDelete(job)}
                                                            disabled={busyJobId === job.id}
                                                        >
                                                            Delete
                                                        </button>
                                                    </div>
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        ) : null}
                    </div>
                </div>

                <div className="card bg-base-100 shadow-lg">
                    <div className="card-body space-y-3">
                        <div className="flex items-center justify-between gap-3">
                            <h2 className="card-title">
                                {editingJobId === null ? 'Create Job' : 'Edit Job'}
                            </h2>
                            {editingJobId !== null ? (
                                <button className="btn btn-sm btn-ghost" onClick={resetForm}>
                                    Cancel
                                </button>
                            ) : null}
                        </div>

                        <label className="form-control">
                            <span className="label-text">Job name</span>
                            <input
                                type="text"
                                className="input input-bordered"
                                value={form.name}
                                onChange={(event) => setForm((current) => ({ ...current, name: event.target.value }))}
                                placeholder="Weekday finance quick sync"
                            />
                        </label>

                        <div className="grid grid-cols-2 gap-3">
                            <label className="form-control">
                                <span className="label-text">Sync type</span>
                                <select
                                    className="select select-bordered"
                                    value={form.syncType}
                                    onChange={(event) => setForm((current) => ({
                                        ...current,
                                        syncType: event.target.value as ScheduledSyncType,
                                    }))}
                                >
                                    <option value="history">History</option>
                                    <option value="finance">Finance</option>
                                    <option value="company">Company</option>
                                </select>
                            </label>
                            <label className="form-control">
                                <span className="label-text">Action</span>
                                <select
                                    className="select select-bordered"
                                    value={form.syncAction}
                                    onChange={(event) => setForm((current) => ({
                                        ...current,
                                        syncAction: event.target.value as ScheduledSyncAction,
                                    }))}
                                >
                                    {actionOptions.map((option) => (
                                        <option key={option.value} value={option.value}>
                                            {option.label}
                                        </option>
                                    ))}
                                </select>
                            </label>
                        </div>

                        <label className="form-control">
                            <span className="label-text">Index scope</span>
                            <select
                                className="select select-bordered"
                                value={form.indexSymbol}
                                onChange={(event) => setForm((current) => ({ ...current, indexSymbol: event.target.value }))}
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
                            <span className="label-text">Tickers</span>
                            <input
                                type="text"
                                className="input input-bordered"
                                value={form.symbolsText}
                                onChange={(event) => setForm((current) => ({ ...current, symbolsText: event.target.value }))}
                                placeholder="Optional, comma or space separated"
                            />
                        </label>

                        {showDateRange ? (
                            <div className="grid grid-cols-2 gap-3">
                                <label className="form-control">
                                    <span className="label-text">From date</span>
                                    <input
                                        type="date"
                                        className="input input-bordered"
                                        value={form.dateFrom}
                                        onChange={(event) => setForm((current) => ({ ...current, dateFrom: event.target.value }))}
                                    />
                                </label>
                                <label className="form-control">
                                    <span className="label-text">To date</span>
                                    <input
                                        type="date"
                                        className="input input-bordered"
                                        value={form.dateTo}
                                        onChange={(event) => setForm((current) => ({ ...current, dateTo: event.target.value }))}
                                    />
                                </label>
                            </div>
                        ) : null}

                        {showAutoRepair ? (
                            <label className="label cursor-pointer justify-start gap-3">
                                <input
                                    type="checkbox"
                                    className="checkbox"
                                    checked={form.autoRepair}
                                    onChange={(event) => setForm((current) => ({ ...current, autoRepair: event.target.checked }))}
                                />
                                <span className="label-text">Auto repair detected gaps</span>
                            </label>
                        ) : null}

                        <label className="form-control">
                            <span className="label-text">First run time</span>
                            <input
                                type="datetime-local"
                                className="input input-bordered"
                                value={form.startsAt}
                                onChange={(event) => setForm((current) => ({ ...current, startsAt: event.target.value }))}
                            />
                        </label>

                        <div className="grid grid-cols-2 gap-3">
                            <label className="form-control">
                                <span className="label-text">Interval value</span>
                                <input
                                    type="number"
                                    min={1}
                                    className="input input-bordered"
                                    value={form.intervalValue}
                                    onChange={(event) => setForm((current) => ({ ...current, intervalValue: event.target.value }))}
                                />
                            </label>
                            <label className="form-control">
                                <span className="label-text">Interval unit</span>
                                <select
                                    className="select select-bordered"
                                    value={form.intervalUnit}
                                    onChange={(event) => setForm((current) => ({
                                        ...current,
                                        intervalUnit: event.target.value as ScheduledSyncIntervalUnit,
                                    }))}
                                >
                                    <option value="minutes">Minutes</option>
                                    <option value="hours">Hours</option>
                                    <option value="days">Days</option>
                                </select>
                            </label>
                        </div>

                        <div className="grid grid-cols-2 gap-3">
                            <label className="form-control">
                                <span className="label-text">Max retries</span>
                                <input
                                    type="number"
                                    min={0}
                                    className="input input-bordered"
                                    value={form.maxRetries}
                                    onChange={(event) => setForm((current) => ({ ...current, maxRetries: event.target.value }))}
                                />
                            </label>
                            <label className="label cursor-pointer justify-start gap-3 pt-8">
                                <input
                                    type="checkbox"
                                    className="checkbox"
                                    checked={form.enabled}
                                    onChange={(event) => setForm((current) => ({ ...current, enabled: event.target.checked }))}
                                />
                                <span className="label-text">Enabled</span>
                            </label>
                        </div>

                        <button className="btn btn-primary" onClick={() => void handleSubmit()} disabled={saving}>
                            {saving ? <span className="loading loading-spinner loading-xs"></span> : null}
                            {editingJobId === null ? 'Create scheduled job' : 'Save changes'}
                        </button>
                    </div>
                </div>
            </section>

            <section className="card bg-base-100 shadow-lg">
                <div className="card-body space-y-3">
                    <div className="flex items-center justify-between gap-3">
                        <h2 className="card-title">Run Logs</h2>
                        <span className="text-sm text-base-content/60">Latest 20 attempts</span>
                    </div>
                    {runs.length === 0 ? (
                        <div className="rounded-lg border border-dashed border-base-300 p-4 text-sm text-base-content/70">
                            No scheduler runs yet.
                        </div>
                    ) : (
                        <div className="overflow-x-auto">
                            <table className="table table-zebra table-sm">
                                <thead>
                                    <tr>
                                        <th>Job</th>
                                        <th>Status</th>
                                        <th>Scheduled</th>
                                        <th>Started</th>
                                        <th>Finished</th>
                                        <th>Result</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {runs.map((run) => (
                                        <tr key={run.id}>
                                            <td>
                                                <div className="font-medium">{run.job_name}</div>
                                                <div className="text-xs text-base-content/60">
                                                    {run.sync_type}/{run.sync_action} · attempt {run.attempt_number}
                                                </div>
                                            </td>
                                            <td>
                                                <span className={`badge ${getStatusBadgeClass(run.status)}`}>
                                                    {run.status}
                                                </span>
                                            </td>
                                            <td className="text-xs">{formatDateTimeInTimezone(run.scheduled_for, SCHEDULER_TIMEZONE)}</td>
                                            <td className="text-xs">{formatDateTimeInTimezone(run.started_at, SCHEDULER_TIMEZONE)}</td>
                                            <td className="text-xs">{formatDateTimeInTimezone(run.finished_at, SCHEDULER_TIMEZONE)}</td>
                                            <td className="max-w-md whitespace-normal text-xs">
                                                {summarizeRun(run)}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}
                </div>
            </section>
        </div>
    );
};
