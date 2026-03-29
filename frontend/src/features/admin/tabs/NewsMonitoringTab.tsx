import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
    type NewsMonitoringOverviewResponse,
    type NewsMonitoringRun,
    type NewsRssSource,
    type NewsSourceKind,
    type NewsSourceSummary,
    type NewsSourcesResponse,
    stockApi,
} from '../../../api/stockApi';
import { formatDateTime, getErrorMessage } from '../adminUtils';

const REFRESH_INTERVAL_MS = 10000;

type MonitoringSourceRow = {
    id: number;
    kind: NewsSourceKind;
    title: string;
    sourceUrl: string;
    scopeLabel: string;
    enabled: boolean;
    validationStatus: NewsSourceSummary['validation_status'];
    lastValidatedAt: string | null;
    lastError: string | null;
    pollIntervalMinutes: number;
    siteName: string | null;
    siteUrl: string | null;
    discoveryMethod?: NewsRssSource['discovery_method'];
    excerptSelector?: string | null;
    paginationSelector?: string | null;
    articleLinkSelector?: string;
    contentSelector?: string;
};

const safeLabel = (value: string | null | undefined, fallback: string) => {
    return value && value.trim() ? value.trim() : fallback;
};

const getScopeLabel = (source: { siteName: string | null; siteUrl: string | null }, sources: NewsSourcesResponse) => {
    const normalizedSourceName = source.siteName?.trim().toLowerCase();
    const normalizedSourceUrl = source.siteUrl?.trim().toLowerCase();
    const matchedSite = sources.sites.find((site) => {
        const candidates = [
            site.display_name,
            site.homepage_url,
            site.domain,
            (() => {
                try {
                    return new URL(site.homepage_url).hostname;
                } catch {
                    return null;
                }
            })(),
        ]
            .filter(Boolean)
            .map((value) => value?.toLowerCase());

        return candidates.some((candidate) => candidate === normalizedSourceName || candidate === normalizedSourceUrl);
    });

    if (!matchedSite) {
        return 'Unknown';
    }
    return matchedSite.is_public ? 'Public default' : 'Private';
};

const flattenSources = (sources: NewsSourcesResponse): MonitoringSourceRow[] => {
    const rows: MonitoringSourceRow[] = [];

    for (const source of sources.rss_sources) {
        rows.push({
            id: source.id,
            kind: 'rss',
            title: safeLabel(source.title, source.feed_url),
            sourceUrl: source.feed_url,
            scopeLabel: getScopeLabel({ siteName: source.site_name, siteUrl: source.site_url }, sources),
            enabled: source.enabled,
            validationStatus: source.validation_status,
            lastValidatedAt: source.last_validated_at,
            lastError: source.last_error,
            pollIntervalMinutes: source.poll_interval_minutes,
            siteName: source.site_name,
            siteUrl: source.site_url,
            discoveryMethod: source.discovery_method,
        });
    }

    for (const source of sources.crawl_sources) {
        rows.push({
            id: source.id,
            kind: 'crawl',
            title: safeLabel(source.title, source.listing_url),
            sourceUrl: source.listing_url,
            scopeLabel: getScopeLabel({ siteName: source.site_name, siteUrl: source.site_url }, sources),
            enabled: source.enabled,
            validationStatus: source.validation_status,
            lastValidatedAt: source.last_validated_at,
            lastError: source.last_error,
            pollIntervalMinutes: source.poll_interval_minutes,
            siteName: source.site_name,
            siteUrl: source.site_url,
            excerptSelector: source.excerpt_selector,
            paginationSelector: source.pagination_selector,
            articleLinkSelector: source.article_link_selector,
            contentSelector: source.content_selector,
        });
    }

    return rows.sort((left, right) => {
        if (left.kind !== right.kind) {
            return left.kind.localeCompare(right.kind);
        }
        return left.title.localeCompare(right.title);
    });
};

const getValidationBadgeClass = (validationStatus: NewsSourceSummary['validation_status']) => {
    switch (validationStatus) {
        case 'valid':
            return 'badge-success';
        case 'invalid':
            return 'badge-error';
        default:
            return 'badge-warning';
    }
};

const getRunBadgeClass = (status: string) => {
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

export const NewsMonitoringTab: React.FC = () => {
    const [overview, setOverview] = useState<NewsMonitoringOverviewResponse | null>(null);
    const [currentDefaultPollInterval, setCurrentDefaultPollInterval] = useState<number | null>(null);
    const [defaultPollIntervalDraft, setDefaultPollIntervalDraft] = useState('');
    const [sources, setSources] = useState<NewsSourcesResponse | null>(null);
    const [runs, setRuns] = useState<NewsMonitoringRun[]>([]);
    const [loading, setLoading] = useState(true);
    const [busyAction, setBusyAction] = useState<'refresh' | 'ingest' | 'catalog' | 'repairTitles' | 'saveConfig' | 'applyDefault' | null>(null);
    const [busySourceKey, setBusySourceKey] = useState<string | null>(null);
    const [message, setMessage] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [warning, setWarning] = useState<string | null>(null);

    const loadData = useCallback(async (showSpinner: boolean = false) => {
        if (showSpinner) {
            setLoading(true);
        }

        setError(null);
        setWarning(null);

        const sourceLoad = async () => {
            try {
                return await stockApi.getNewsMonitoringSources();
            } catch {
                return await stockApi.getNewsSources();
            }
        };

        try {
            const [overviewResult, sourcesResult, runsResult, configResult] = await Promise.allSettled([
                stockApi.getNewsMonitoringOverview(),
                sourceLoad(),
                stockApi.getNewsMonitoringRuns(12),
                stockApi.getNewsAdminConfig(),
            ]);

            if (overviewResult.status === 'fulfilled') {
                setOverview(overviewResult.value);
            } else {
                setOverview(null);
                setWarning((current) => current ?? 'News monitoring overview is unavailable yet.');
            }

            if (sourcesResult.status === 'fulfilled') {
                setSources(sourcesResult.value);
            } else {
                setSources(null);
                setError(getErrorMessage(sourcesResult.reason));
            }

            if (runsResult.status === 'fulfilled') {
                setRuns(runsResult.value.runs);
            } else {
                setRuns([]);
                setWarning((current) => current ?? 'Recent run history is unavailable yet.');
            }

            if (configResult.status === 'fulfilled') {
                setCurrentDefaultPollInterval(configResult.value.default_poll_interval_minutes);
                setDefaultPollIntervalDraft(String(configResult.value.default_poll_interval_minutes));
            } else {
                setCurrentDefaultPollInterval(null);
                setWarning((current) => current ?? 'News admin config is unavailable yet.');
            }
        } catch (loadError) {
            setError(getErrorMessage(loadError));
        } finally {
            if (showSpinner) {
                setLoading(false);
            }
        }
    }, []);

    useEffect(() => {
        void loadData(true);
        const intervalId = window.setInterval(() => {
            void loadData(false);
        }, REFRESH_INTERVAL_MS);

        return () => {
            window.clearInterval(intervalId);
        };
    }, [loadData]);

    const sourceRows = useMemo(() => {
        if (!sources) {
            return [];
        }
        return flattenSources(sources);
    }, [sources]);

    const recentRuns = useMemo(() => {
        return [...runs].sort((left, right) => {
            const leftValue = left.started_at ? new Date(left.started_at).getTime() : 0;
            const rightValue = right.started_at ? new Date(right.started_at).getTime() : 0;
            return rightValue - leftValue;
        });
    }, [runs]);

    const summary = useMemo(() => {
        const totalSources = sourceRows.length;
        const enabledSources = sourceRows.filter((source) => source.enabled).length;
        const validSources = sourceRows.filter((source) => source.validationStatus === 'valid').length;
        const invalidSources = sourceRows.filter((source) => source.validationStatus === 'invalid').length;
        const publicSources = sourceRows.filter((source) => source.scopeLabel === 'Public default').length;
        const privateSources = sourceRows.filter((source) => source.scopeLabel === 'Private').length;
        const unknownSources = sourceRows.filter((source) => source.scopeLabel === 'Unknown').length;
        const activeRuns = overview?.active_runs ?? recentRuns.filter((run) => run.status === 'running').length;
        const lastRun = recentRuns[0] ?? null;

        return {
            totalSources,
            enabledSources,
            validSources,
            invalidSources,
            publicSources,
            privateSources,
            unknownSources,
            activeRuns,
            articlesTotal: overview?.articles_total ?? null,
            articlesLast24h: overview?.articles_last_24h ?? null,
            queueSize: overview?.queue_size ?? null,
            lastRun,
        };
    }, [overview, recentRuns, sourceRows]);

    const handleRefresh = useCallback(async () => {
        setBusyAction('refresh');
        setMessage(null);
        setError(null);
        try {
            await loadData(true);
            setMessage('News monitoring data refreshed.');
        } catch (refreshError) {
            setError(getErrorMessage(refreshError));
        } finally {
            setBusyAction(null);
        }
    }, [loadData]);

    const handleTriggerIngestion = useCallback(async () => {
        setBusyAction('ingest');
        setMessage(null);
        setError(null);
        try {
            const response = await stockApi.triggerNewsIngestion();
            setMessage(response.message);
            await loadData(false);
        } catch (ingestError) {
            setError(getErrorMessage(ingestError));
        } finally {
            setBusyAction(null);
        }
    }, [loadData]);

    const handleRefreshCatalog = useCallback(async () => {
        setBusyAction('catalog');
        setMessage(null);
        setError(null);
        try {
            const response = await stockApi.refreshNewsMonitoring();
            setMessage(response.message);
            await loadData(false);
        } catch (catalogError) {
            setError(getErrorMessage(catalogError));
        } finally {
            setBusyAction(null);
        }
    }, [loadData]);

    const handleRepairRssTitles = useCallback(async () => {
        setBusyAction('repairTitles');
        setMessage(null);
        setError(null);
        try {
            const response = await stockApi.repairNewsRssTitles();
            setMessage(response.message);
            await loadData(false);
        } catch (repairError) {
            setError(getErrorMessage(repairError));
        } finally {
            setBusyAction(null);
        }
    }, [loadData]);

    const handleSaveConfig = useCallback(async () => {
        const parsed = Number(defaultPollIntervalDraft);
        if (!Number.isFinite(parsed) || parsed < 5) {
            setError('Default poll interval must be at least 5 minutes.');
            setMessage(null);
            return;
        }

        setBusyAction('saveConfig');
        setMessage(null);
        setError(null);
        try {
            const response = await stockApi.updateNewsAdminConfig({
                default_poll_interval_minutes: parsed,
            });
            setCurrentDefaultPollInterval(response.default_poll_interval_minutes);
            setDefaultPollIntervalDraft(String(response.default_poll_interval_minutes));
            setMessage('News default poll interval saved.');
        } catch (saveError) {
            setError(getErrorMessage(saveError));
        } finally {
            setBusyAction(null);
        }
    }, [defaultPollIntervalDraft]);

    const handleApplyDefaultPollInterval = useCallback(async () => {
        const parsed = Number(defaultPollIntervalDraft);
        if (!Number.isFinite(parsed) || parsed < 5) {
            setError('Default poll interval must be at least 5 minutes.');
            setMessage(null);
            return;
        }

        setBusyAction('applyDefault');
        setMessage(null);
        setError(null);
        try {
            if (currentDefaultPollInterval !== parsed) {
                const configResponse = await stockApi.updateNewsAdminConfig({
                    default_poll_interval_minutes: parsed,
                });
                setCurrentDefaultPollInterval(configResponse.default_poll_interval_minutes);
                setDefaultPollIntervalDraft(String(configResponse.default_poll_interval_minutes));
            }
            const response = await stockApi.applyNewsDefaultPollIntervalToExistingSources();
            setMessage(response.message);
            await loadData(false);
        } catch (applyError) {
            setError(getErrorMessage(applyError));
        } finally {
            setBusyAction(null);
        }
    }, [currentDefaultPollInterval, defaultPollIntervalDraft, loadData]);

    const handleToggleEnabled = useCallback(async (source: MonitoringSourceRow) => {
        setBusySourceKey(`${source.kind}:${source.id}`);
        setMessage(null);
        setError(null);
        try {
            await stockApi.updateNewsSource(source.kind, source.id, { enabled: !source.enabled });
            setMessage(`${source.title} was ${source.enabled ? 'disabled' : 'enabled'}.`);
            await loadData(false);
        } catch (toggleError) {
            setError(getErrorMessage(toggleError));
        } finally {
            setBusySourceKey(null);
        }
    }, [loadData]);

    const handleRevalidate = useCallback(async (source: MonitoringSourceRow) => {
        setBusySourceKey(`${source.kind}:${source.id}:validate`);
        setMessage(null);
        setError(null);
        try {
            if (source.kind === 'rss') {
                await stockApi.validateNewsRss({
                    feed_url: source.sourceUrl,
                    site_url: source.siteUrl,
                    title: source.title,
                    enabled: source.enabled,
                    poll_interval_minutes: source.pollIntervalMinutes,
                    discovery_method: 'manual',
                });
            } else {
                await stockApi.validateNewsCrawl({
                    listing_url: source.sourceUrl,
                    article_link_selector: source.articleLinkSelector ?? 'a',
                    content_selector: source.contentSelector ?? 'article',
                    excerpt_selector: source.excerptSelector,
                    pagination_selector: source.paginationSelector,
                    title: source.title,
                    site_url: source.siteUrl,
                    enabled: source.enabled,
                    poll_interval_minutes: source.pollIntervalMinutes,
                });
            }
            setMessage(`Validation request submitted for ${source.title}.`);
            await loadData(false);
        } catch (validateError) {
            setError(getErrorMessage(validateError));
        } finally {
            setBusySourceKey(null);
        }
    }, [loadData]);

    const handleDeleteSource = useCallback(async (source: MonitoringSourceRow) => {
        if (!window.confirm(`Delete "${source.title}" from the source catalog?`)) {
            return;
        }

        setBusySourceKey(`${source.kind}:${source.id}:delete`);
        setMessage(null);
        setError(null);
        try {
            await stockApi.deleteNewsMonitoringSource(source.kind, source.id);
            setMessage(`${source.title} was deleted.`);
            await loadData(false);
        } catch (deleteError) {
            setError(getErrorMessage(deleteError));
        } finally {
            setBusySourceKey(null);
        }
    }, [loadData]);

    return (
        <section className="space-y-6">
            <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                    <h2 className="text-2xl font-bold">News Monitoring</h2>
                    <p className="text-sm opacity-70">
                        Monitor public source coverage, inspect recent ingestion runs, and trigger a fresh crawl when needed.
                    </p>
                </div>
                <div className="flex flex-wrap gap-2">
                    <button className="btn btn-outline btn-sm" onClick={handleRefresh} disabled={busyAction !== null || loading}>
                        {busyAction === 'refresh' ? <span className="loading loading-spinner loading-xs"></span> : null}
                        Refresh data
                    </button>
                    <button className="btn btn-primary btn-sm" onClick={handleTriggerIngestion} disabled={busyAction !== null || loading}>
                        {busyAction === 'ingest' ? <span className="loading loading-spinner loading-xs"></span> : null}
                        Trigger ingestion now
                    </button>
                    <button className="btn btn-secondary btn-sm" onClick={handleRefreshCatalog} disabled={busyAction !== null || loading}>
                        {busyAction === 'catalog' ? <span className="loading loading-spinner loading-xs"></span> : null}
                        Refresh source cache
                    </button>
                    <button className="btn btn-outline btn-sm" onClick={handleRepairRssTitles} disabled={busyAction !== null || loading}>
                        {busyAction === 'repairTitles' ? <span className="loading loading-spinner loading-xs"></span> : null}
                        Repair RSS titles
                    </button>
                </div>
            </div>

            {message ? (
                <div className="alert alert-success">
                    <span>{message}</span>
                </div>
            ) : null}

            {warning ? (
                <div className="alert alert-warning">
                    <span>{warning}</span>
                </div>
            ) : null}

            {error ? (
                <div className="alert alert-error">
                    <span>{error}</span>
                </div>
            ) : null}

            <section className="rounded-2xl border border-base-300 bg-base-100 px-5 py-4 shadow-sm">
                <div className="max-w-3xl">
                    <h3 className="text-sm font-semibold uppercase tracking-[0.2em] text-base-content/70">Settings</h3>
                    <p className="mt-5 text-sm text-base-content/70">
                        Default poll interval for new RSS and crawl sources when no custom value is provided.
                    </p>
                    <div className="mt-3 flex flex-wrap items-center gap-2">
                        <span className="text-sm font-medium text-base-content/80">Default poll</span>
                        <label className="join">
                            <input
                                type="number"
                                min="5"
                                className="input input-bordered input-sm join-item w-24"
                                value={defaultPollIntervalDraft}
                                onChange={(event) => setDefaultPollIntervalDraft(event.target.value)}
                                aria-label="Default poll interval"
                            />
                            <span className="join-item inline-flex items-center border border-base-300 bg-base-200 px-3 text-sm text-base-content/70">
                                minutes
                            </span>
                        </label>
                        <button className="btn btn-outline btn-sm" onClick={handleSaveConfig} disabled={busyAction !== null || loading}>
                            {busyAction === 'saveConfig' ? <span className="loading loading-spinner loading-xs"></span> : null}
                            Save
                        </button>
                        <button className="btn btn-ghost btn-sm" onClick={handleApplyDefaultPollInterval} disabled={busyAction !== null || loading}>
                            {busyAction === 'applyDefault' ? <span className="loading loading-spinner loading-xs"></span> : null}
                            Apply to existing sources
                        </button>
                    </div>
                </div>
            </section>

            <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                <div className="card bg-base-100 shadow-lg">
                    <div className="card-body">
                        <h3 className="card-title text-base">Source Coverage</h3>
                        <p className="text-3xl font-bold">{summary.totalSources}</p>
                        <p className="text-sm opacity-70">
                            {summary.enabledSources} enabled, {summary.validSources} valid, {summary.invalidSources} invalid
                        </p>
                    </div>
                </div>
                <div className="card bg-base-100 shadow-lg">
                    <div className="card-body">
                        <h3 className="card-title text-base">Scope Mix</h3>
                        <p className="text-3xl font-bold">{summary.publicSources}/{summary.privateSources}</p>
                        <p className="text-sm opacity-70">
                            Public default vs private/source-specific{summary.unknownSources > 0 ? `, ${summary.unknownSources} unknown` : ''}
                        </p>
                    </div>
                </div>
                <div className="card bg-base-100 shadow-lg">
                    <div className="card-body">
                        <h3 className="card-title text-base">Article Volume</h3>
                        <p className="text-3xl font-bold">{summary.articlesTotal ?? '-'}</p>
                        <p className="text-sm opacity-70">Last 24h: {summary.articlesLast24h ?? '-'}</p>
                    </div>
                </div>
                <div className="card bg-base-100 shadow-lg">
                    <div className="card-body">
                        <h3 className="card-title text-base">Ingestion Activity</h3>
                        <p className="text-3xl font-bold">{summary.activeRuns} active</p>
                        <p className="text-sm opacity-70">
                            Queue: {summary.queueSize ?? '-'} | Last run: {formatDateTime(summary.lastRun?.started_at)}
                        </p>
                        <p className="text-sm opacity-70">
                            Latest status: {summary.lastRun?.status ?? overview?.last_run_status ?? '-'}
                        </p>
                    </div>
                </div>
            </section>

            <section className="grid gap-4 xl:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
                <div className="card bg-base-100 shadow-lg">
                    <div className="card-body">
                        <div className="flex items-center justify-between gap-3">
                            <h3 className="card-title">News Sources</h3>
                            <span className="text-sm opacity-70">{sourceRows.length} total</span>
                        </div>
                        <div className="overflow-x-auto">
                            <table className="table">
                                <thead>
                                    <tr>
                                        <th>Source</th>
                                        <th>Kind</th>
                                        <th>Scope</th>
                                        <th>Status</th>
                                        <th>Interval</th>
                                        <th>Last check</th>
                                        <th className="text-right">Actions</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {sourceRows.length === 0 ? (
                                        <tr>
                                            <td colSpan={7}>
                                                <div className="py-8 text-center text-sm opacity-70">
                                                    No news sources available yet.
                                                </div>
                                            </td>
                                        </tr>
                                    ) : (
                                        sourceRows.map((source) => {
                                            const busyKey = `${source.kind}:${source.id}`;
                                            const validatingKey = `${busyKey}:validate`;
                                            const deletingKey = `${busyKey}:delete`;
                                            const isBusy = busySourceKey === busyKey || busySourceKey === validatingKey || busySourceKey === deletingKey;

                                            return (
                                                <tr key={busyKey}>
                                                    <td>
                                                        <div className="font-medium">{source.title}</div>
                                                        <div className="text-xs opacity-70 break-all">{source.sourceUrl}</div>
                                                        <div className="text-xs opacity-60">{source.siteName ?? source.siteUrl ?? '-'}</div>
                                                    </td>
                                                    <td>
                                                        <span className="badge badge-outline uppercase">{source.kind}</span>
                                                    </td>
                                                    <td>{source.scopeLabel}</td>
                                                    <td>
                                                        <span className={`badge ${getValidationBadgeClass(source.validationStatus)}`}>
                                                            {source.validationStatus}
                                                        </span>
                                                        {source.lastError ? (
                                                            <div className="mt-1 max-w-[16rem] text-xs text-error break-words">
                                                                {source.lastError}
                                                            </div>
                                                        ) : null}
                                                    </td>
                                                    <td>{source.pollIntervalMinutes}m</td>
                                                    <td>{formatDateTime(source.lastValidatedAt)}</td>
                                                    <td>
                                                        <div className="flex flex-wrap justify-end gap-2">
                                                            <button
                                                                className="btn btn-ghost btn-xs"
                                                                onClick={() => void handleRevalidate(source)}
                                                                disabled={isBusy}
                                                            >
                                                                {busySourceKey === validatingKey ? <span className="loading loading-spinner loading-xs"></span> : null}
                                                                Revalidate
                                                            </button>
                                                            <button
                                                                className="btn btn-outline btn-xs"
                                                                onClick={() => void handleToggleEnabled(source)}
                                                                disabled={isBusy}
                                                            >
                                                                {busySourceKey === busyKey ? <span className="loading loading-spinner loading-xs"></span> : null}
                                                                {source.enabled ? 'Disable' : 'Enable'}
                                                            </button>
                                                            <button
                                                                className="btn btn-error btn-outline btn-xs"
                                                                onClick={() => void handleDeleteSource(source)}
                                                                disabled={isBusy}
                                                            >
                                                                {busySourceKey === deletingKey ? <span className="loading loading-spinner loading-xs"></span> : null}
                                                                Delete
                                                            </button>
                                                        </div>
                                                    </td>
                                                </tr>
                                            );
                                        })
                                    )}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>

                <div className="card bg-base-100 shadow-lg">
                    <div className="card-body">
                        <div className="flex items-center justify-between gap-3">
                            <h3 className="card-title">Recent Runs</h3>
                            <span className="text-sm opacity-70">{recentRuns.length} items</span>
                        </div>
                        <div className="overflow-x-auto">
                            <table className="table">
                                <thead>
                                    <tr>
                                        <th>Status</th>
                                        <th>Source</th>
                                        <th>Counts</th>
                                        <th>Started</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {recentRuns.length === 0 ? (
                                        <tr>
                                            <td colSpan={4}>
                                                <div className="py-8 text-center text-sm opacity-70">
                                                    No recent runs are available yet.
                                                </div>
                                            </td>
                                        </tr>
                                    ) : (
                                        recentRuns.map((run) => (
                                            <tr key={run.id}>
                                                <td>
                                                    <span className={`badge ${getRunBadgeClass(run.status)}`}>{run.status}</span>
                                                    {run.error ? (
                                                        <div className="mt-1 max-w-[14rem] text-xs text-error break-words">
                                                            {run.error}
                                                        </div>
                                                    ) : null}
                                                </td>
                                                <td>
                                                    <div className="font-medium">{run.source_label ?? '-'}</div>
                                                    <div className="text-xs opacity-70">{run.source_type}</div>
                                                </td>
                                                <td className="text-sm">
                                                    <div>Fetched: {run.fetched_count}</div>
                                                    <div>Stored: {run.stored_count}</div>
                                                    <div>Filtered: {run.filtered_count}</div>
                                                </td>
                                                <td>
                                                    <div>{formatDateTime(run.started_at)}</div>
                                                    <div className="text-xs opacity-70">{formatDateTime(run.finished_at)}</div>
                                                </td>
                                            </tr>
                                        ))
                                    )}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </section>
        </section>
    );
};

export default NewsMonitoringTab;
