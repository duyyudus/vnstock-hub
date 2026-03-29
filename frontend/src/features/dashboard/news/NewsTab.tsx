import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { formatDateTime, getErrorMessage } from '../../admin/adminUtils';
import { useAuthUser } from '../../auth/useAuthUser';
import {
    type BookmarkGroup,
    stockApi,
    type NewsArticleDetail,
    type NewsArticleSummaryResponse,
    type NewsCrawlDiscoveryCandidate,
    type NewsCrawlSource,
    type NewsCrawlSourceCreateRequest,
    type NewsEventType,
    type NewsFeedItem,
    type NewsFeedQuery,
    type NewsImportanceLevel,
    type NewsRssDiscoveryCandidate,
    type NewsRssSource,
    type NewsRssSourceCreateRequest,
    type NewsSourceSummary,
    type NewsUserPreferences,
} from '../../../api/stockApi';

type FeedDraft = {
    source: string;
    topic: string;
    ticker: string;
    from: string;
    to: string;
    sort: 'latest' | 'relevance';
    scope: 'all' | 'portfolio' | 'bookmarks';
    bookmarkGroupId: string;
    eventType: '' | NewsEventType;
    importance: '' | NewsImportanceLevel;
    groupBy: 'article' | 'story';
};

type FeedViewKey = 'forYou' | 'latest' | 'portfolio' | 'bookmarks';

type RssDraft = {
    feedUrl: string;
    siteUrl: string;
    homepageUrl: string;
    title: string;
    pollIntervalMinutes: string;
};

type CrawlDraft = {
    listingUrl: string;
    articleLinkSelector: string;
    contentSelector: string;
    excerptSelector: string;
    paginationSelector: string;
    title: string;
    siteUrl: string;
    pollIntervalMinutes: string;
};

type PanelKey = 'rssDiscovery' | 'manualRss' | 'crawlSource' | 'blockedTopics' | 'sources';

const FEED_PAGE_SIZE = 20;
const NEWS_UTILITY_RAIL_STORAGE_KEY = 'news:utility-rail-open';

const NEWS_EVENT_FILTER_OPTIONS: Array<{ value: NewsEventType; label: string }> = [
    { value: 'earnings', label: 'Earnings' },
    { value: 'dividend', label: 'Dividend' },
    { value: 'capital_raise', label: 'Capital Raise' },
    { value: 'insider_trading', label: 'Insider Trading' },
    { value: 'management_change', label: 'Management Change' },
    { value: 'regulatory', label: 'Regulatory' },
    { value: 'mna', label: 'M&A' },
    { value: 'analyst_view', label: 'Analyst View' },
    { value: 'macro_policy', label: 'Macro Policy' },
    { value: 'other', label: 'Other' },
];

const NEWS_IMPORTANCE_OPTIONS: Array<{ value: NewsImportanceLevel; label: string }> = [
    { value: 'high', label: 'High' },
    { value: 'medium', label: 'Medium' },
    { value: 'low', label: 'Low' },
];

const createDefaultFeedDraft = (): FeedDraft => ({
    source: '',
    topic: '',
    ticker: '',
    from: '',
    to: '',
    sort: 'latest',
    scope: 'all',
    bookmarkGroupId: '',
    eventType: '',
    importance: '',
    groupBy: 'story',
});

const emptyRssDraft: RssDraft = {
    feedUrl: '',
    siteUrl: '',
    homepageUrl: '',
    title: '',
    pollIntervalMinutes: '',
};

const emptyCrawlDraft: CrawlDraft = {
    listingUrl: '',
    articleLinkSelector: '',
    contentSelector: '',
    excerptSelector: '',
    paginationSelector: '',
    title: '',
    siteUrl: '',
    pollIntervalMinutes: '',
};

const normalizeTextInput = (value: string) => value.trim();

const extractUrlDomain = (value: string | null | undefined) => {
    if (!value) {
        return null;
    }
    try {
        return new URL(value).hostname.replace(/^www\./, '').toLowerCase();
    } catch {
        return null;
    }
};

const formatRelativeTime = (value: string | null) => {
    if (!value) {
        return '-';
    }
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) {
        return value;
    }
    const diffMs = parsed.getTime() - Date.now();
    const absMinutes = Math.round(Math.abs(diffMs) / 60000);
    if (absMinutes < 1) {
        return 'just now';
    }
    if (absMinutes < 60) {
        return `${absMinutes}m ${diffMs < 0 ? 'ago' : 'from now'}`;
    }
    const absHours = Math.round(absMinutes / 60);
    if (absHours < 24) {
        return `${absHours}h ${diffMs < 0 ? 'ago' : 'from now'}`;
    }
    const absDays = Math.round(absHours / 24);
    return `${absDays}d ${diffMs < 0 ? 'ago' : 'from now'}`;
};

const formatValidationBadge = (status: NewsSourceSummary['validation_status']) => {
    switch (status) {
        case 'valid':
            return 'badge-success';
        case 'invalid':
            return 'badge-error';
        default:
            return 'badge-warning';
    }
};

const sourceLabel = (source: NewsSourceSummary) => {
    const siteName = source.site_name?.trim();
    if (siteName) {
        return siteName;
    }
    return source.title?.trim() || source.site_url?.trim() || 'Unlabeled source';
};

const joinTags = (values: string[]) => values.join(' · ');

const mergeNewsSources = (rssSources: NewsRssSource[], crawlSources: NewsCrawlSource[]) => {
    return [
        ...rssSources.map((source) => ({ ...source, type: 'rss' as const })),
        ...crawlSources.map((source) => ({ ...source, type: 'crawl' as const })),
    ];
};

type CollapsiblePanelProps = {
    title: string;
    description: string;
    isOpen: boolean;
    onToggle: () => void;
    children: React.ReactNode;
    action?: React.ReactNode;
};

const CollapsiblePanel: React.FC<CollapsiblePanelProps> = ({ title, description, isOpen, onToggle, children, action }) => {
    return (
        <section className="card bg-base-100 shadow-lg border border-base-300 overflow-hidden">
            <button
                type="button"
                className="flex w-full items-start justify-between gap-3 px-5 py-4 text-left transition-colors hover:bg-base-200/60"
                onClick={onToggle}
                aria-expanded={isOpen}
            >
                <div className="min-w-0">
                    <h3 className="card-title text-lg">{title}</h3>
                    <p className="mt-1 text-sm text-base-content/70">{description}</p>
                </div>
                <div className="flex items-center gap-3 pl-2">
                    {action}
                    <span className="text-lg leading-none text-base-content/50">{isOpen ? '−' : '+'}</span>
                </div>
            </button>
            {isOpen ? (
                <div className="border-t border-base-300 px-5 py-4">
                    <div className="space-y-4">{children}</div>
                </div>
            ) : null}
        </section>
    );
};

export const NewsTab: React.FC = () => {
    const user = useAuthUser();
    const isSignedIn = Boolean(user);
    const canManageSources = isSignedIn;

    const [feedItems, setFeedItems] = useState<NewsFeedItem[]>([]);
    const [feedCursor, setFeedCursor] = useState<string | null>(null);
    const [feedCount, setFeedCount] = useState(0);
    const [feedPersonalized, setFeedPersonalized] = useState(false);
    const [feedLoading, setFeedLoading] = useState(true);
    const [feedError, setFeedError] = useState<string | null>(null);
    const [feedDraft, setFeedDraft] = useState<FeedDraft>(() => createDefaultFeedDraft());
    const [appliedFeed, setAppliedFeed] = useState<FeedDraft>(() => createDefaultFeedDraft());

    const [sources, setSources] = useState<{ rss_sources: NewsRssSource[]; crawl_sources: NewsCrawlSource[] } | null>(null);
    const [sourcesLoading, setSourcesLoading] = useState(false);
    const [sourcesError, setSourcesError] = useState<string | null>(null);

    const [discoveryHomepageUrl, setDiscoveryHomepageUrl] = useState('');
    const [discoveryLoading, setDiscoveryLoading] = useState(false);
    const [discoveryError, setDiscoveryError] = useState<string | null>(null);
    const [discoveryMessage, setDiscoveryMessage] = useState<string | null>(null);
    const [discoveryResult, setDiscoveryResult] = useState<NewsRssDiscoveryCandidate[]>([]);
    const [discoveryCrawlResult, setDiscoveryCrawlResult] = useState<NewsCrawlDiscoveryCandidate[]>([]);
    const [discoveryAttempted, setDiscoveryAttempted] = useState(false);

    const [rssDraft, setRssDraft] = useState<RssDraft>(emptyRssDraft);
    const [crawlDraft, setCrawlDraft] = useState<CrawlDraft>(emptyCrawlDraft);
    const [rssActionMessage, setRssActionMessage] = useState<string | null>(null);
    const [crawlActionMessage, setCrawlActionMessage] = useState<string | null>(null);
    const [rssActionError, setRssActionError] = useState<string | null>(null);
    const [crawlActionError, setCrawlActionError] = useState<string | null>(null);

    const [preferences, setPreferences] = useState<NewsUserPreferences | null>(null);
    const [preferencesDraft, setPreferencesDraft] = useState('');
    const [preferencesLoading, setPreferencesLoading] = useState(false);
    const [preferencesSaving, setPreferencesSaving] = useState(false);
    const [preferencesError, setPreferencesError] = useState<string | null>(null);
    const [preferencesSuccess, setPreferencesSuccess] = useState<string | null>(null);
    const [bookmarkGroups, setBookmarkGroups] = useState<BookmarkGroup[]>([]);

    const [newsDetail, setNewsDetail] = useState<NewsArticleDetail | null>(null);
    const [detailLoadingId, setDetailLoadingId] = useState<number | null>(null);
    const [detailError, setDetailError] = useState<string | null>(null);
    const [detailRefreshLoading, setDetailRefreshLoading] = useState(false);
    const [summaryLoadingById, setSummaryLoadingById] = useState<Record<number, boolean>>({});
    const [summaryErrorById, setSummaryErrorById] = useState<Record<number, string | null>>({});
    const [isUtilityRailOpen, setIsUtilityRailOpen] = useState<boolean>(() => {
        if (typeof window === 'undefined') {
            return false;
        }
        try {
            const stored = window.localStorage.getItem(NEWS_UTILITY_RAIL_STORAGE_KEY);
            if (stored === null) {
                return false;
            }
            return stored === 'true';
        } catch {
            return false;
        }
    });
    const [openPanels, setOpenPanels] = useState<Record<PanelKey, boolean>>({
        rssDiscovery: true,
        manualRss: false,
        crawlSource: false,
        blockedTopics: true,
        sources: true,
    });
    const detailDialogRef = useRef<HTMLDialogElement>(null);
    const appliedFeedRef = useRef(appliedFeed);

    const activeNewsSources = useMemo(() => {
        if (!sources) {
            return [];
        }
        return mergeNewsSources(sources.rss_sources, sources.crawl_sources);
    }, [sources]);

    const sourceFilterOptions = useMemo(() => {
        const domains = new Set<string>();

        for (const source of activeNewsSources) {
            const domain = extractUrlDomain(
                source.site_url || (source.kind === 'rss' ? source.feed_url : source.listing_url),
            );
            if (domain) {
                domains.add(domain);
            }
        }

        for (const item of feedItems) {
            const domain = extractUrlDomain(item.canonical_url);
            if (domain) {
                domains.add(domain);
            }
        }

        if (feedDraft.source.trim()) {
            domains.add(feedDraft.source.trim().toLowerCase());
        }

        return Array.from(domains).sort((left, right) => left.localeCompare(right));
    }, [activeNewsSources, feedItems, feedDraft.source]);

    const activeFeedView = useMemo<FeedViewKey>(() => {
        if (feedDraft.scope === 'portfolio') {
            return 'portfolio';
        }
        if (feedDraft.scope === 'bookmarks') {
            return 'bookmarks';
        }
        if (feedDraft.sort === 'relevance') {
            return 'forYou';
        }
        return 'latest';
    }, [feedDraft.scope, feedDraft.sort]);

    useEffect(() => {
        appliedFeedRef.current = appliedFeed;
    }, [appliedFeed]);

    useEffect(() => {
        if (typeof window === 'undefined') {
            return;
        }
        try {
            window.localStorage.setItem(NEWS_UTILITY_RAIL_STORAGE_KEY, String(isUtilityRailOpen));
        } catch {
            // Ignore storage failures and keep the in-memory toggle working.
        }
    }, [isUtilityRailOpen]);

    useEffect(() => {
        const dialog = detailDialogRef.current;
        if (!dialog) {
            return;
        }
        if (newsDetail) {
            if (!dialog.open) {
                dialog.showModal();
            }
            return;
        }
        if (dialog.open) {
            dialog.close();
        }
    }, [newsDetail]);

    const buildFeedQuery = useCallback((draft: FeedDraft, cursor?: string | null): NewsFeedQuery => {
        return {
            source: normalizeTextInput(draft.source) || undefined,
            topic: normalizeTextInput(draft.topic) || undefined,
            ticker: normalizeTextInput(draft.ticker).toUpperCase() || undefined,
            from: normalizeTextInput(draft.from) || undefined,
            to: normalizeTextInput(draft.to) || undefined,
            sort: draft.sort,
            scope: draft.scope,
            bookmark_group_id: draft.scope === 'bookmarks' && draft.bookmarkGroupId ? Number(draft.bookmarkGroupId) : undefined,
            event_type: draft.eventType || undefined,
            importance: draft.importance || undefined,
            group_by: draft.groupBy,
            cursor: cursor || undefined,
            limit: FEED_PAGE_SIZE,
        };
    }, []);

    const loadNewsFeed = useCallback(async (draft: FeedDraft, append: boolean = false, cursorOverride?: string | null) => {
        if (!append) {
            setFeedLoading(true);
            setFeedError(null);
        }

        try {
            const payload = buildFeedQuery(draft, append ? cursorOverride : undefined);
            const response = await stockApi.getNewsFeed(payload);
            setFeedPersonalized(response.is_personalized);
            setFeedCount(response.count);
            setFeedCursor(response.next_cursor);
            setFeedItems((current) => append ? [...current, ...response.items] : response.items);
        } catch (error) {
            setFeedError(getErrorMessage(error));
        } finally {
            if (!append) {
                setFeedLoading(false);
            }
        }
    }, [buildFeedQuery]);

    const loadSourceManagement = useCallback(async () => {
        if (!canManageSources) {
            setSources(null);
            setPreferences(null);
            setPreferencesDraft('');
            setBookmarkGroups([]);
            setSourcesLoading(false);
            setPreferencesLoading(false);
            return;
        }

        setSourcesLoading(true);
        setSourcesError(null);
        setPreferencesLoading(true);
        setPreferencesError(null);

        try {
            const [sourcesResponse, preferencesResponse, bookmarkGroupsResponse] = await Promise.all([
                stockApi.getNewsSources(),
                stockApi.getNewsPreferences(),
                stockApi.getBookmarkGroups(),
            ]);
            setSources({
                rss_sources: sourcesResponse.rss_sources,
                crawl_sources: sourcesResponse.crawl_sources,
            });
            setPreferences(preferencesResponse);
            setPreferencesDraft(preferencesResponse.blocked_topics_text);
            setBookmarkGroups(bookmarkGroupsResponse.groups);
        } catch (error) {
            const message = getErrorMessage(error);
            setSourcesError(message);
            setPreferencesError(message);
        } finally {
            setSourcesLoading(false);
            setPreferencesLoading(false);
        }
    }, [canManageSources]);

    useEffect(() => {
        const nextDraft = createDefaultFeedDraft();
        setFeedDraft(nextDraft);
        setAppliedFeed(nextDraft);
        setFeedCursor(null);
        void loadNewsFeed(nextDraft, false);
    }, [isSignedIn, loadNewsFeed]);

    useEffect(() => {
        void loadSourceManagement();
    }, [loadSourceManagement]);

    const handleApplyFilters = async () => {
        const nextDraft = feedDraft;
        setAppliedFeed(nextDraft);
        setFeedCursor(null);
        await loadNewsFeed(nextDraft, false);
    };

    const handleFeedInputKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
        if (event.key !== 'Enter' || event.nativeEvent.isComposing) {
            return;
        }
        event.preventDefault();
        void handleApplyFilters();
    };

    const handleSourceFilterChange = async (value: string) => {
        const nextDraft = {
            ...feedDraft,
            source: value,
        };
        setFeedDraft(nextDraft);
        setAppliedFeed(nextDraft);
        setFeedCursor(null);
        await loadNewsFeed(nextDraft, false);
    };

    const handleResetFilters = async () => {
        const nextDraft = createDefaultFeedDraft();
        setFeedDraft(nextDraft);
        setAppliedFeed(nextDraft);
        setFeedCursor(null);
        await loadNewsFeed(nextDraft, false);
    };

    const handleFeedViewChange = async (view: FeedViewKey) => {
        const baseDraft = {
            ...feedDraft,
            bookmarkGroupId: view === 'bookmarks' ? feedDraft.bookmarkGroupId : '',
            groupBy: 'story' as const,
        };
        const nextDraft: FeedDraft =
            view === 'latest'
                ? { ...baseDraft, sort: 'latest', scope: 'all' }
                : view === 'portfolio'
                    ? { ...baseDraft, sort: 'relevance', scope: 'portfolio' }
                    : view === 'bookmarks'
                        ? { ...baseDraft, sort: 'relevance', scope: 'bookmarks' }
                        : { ...baseDraft, sort: 'relevance', scope: 'all' };
        setFeedDraft(nextDraft);
        setAppliedFeed(nextDraft);
        setFeedCursor(null);
        await loadNewsFeed(nextDraft, false);
    };

    const handleLoadMore = async () => {
        if (!feedCursor || feedLoading) {
            return;
        }
        setFeedLoading(true);
        await loadNewsFeed(appliedFeed, true, feedCursor);
        setFeedLoading(false);
    };

    const handleDiscoverRss = async () => {
        if (!canManageSources) {
            return;
        }
        const homepageUrl = discoveryHomepageUrl.trim();
        if (!homepageUrl) {
            setDiscoveryError('Please enter a homepage URL first.');
            setDiscoveryMessage(null);
            setDiscoveryAttempted(false);
            return;
        }

        setDiscoveryLoading(true);
        setDiscoveryError(null);
        setDiscoveryMessage(null);
        setDiscoveryResult([]);
        setDiscoveryCrawlResult([]);
        setDiscoveryAttempted(true);
        setRssDraft((current) => ({
            ...current,
            homepageUrl,
            siteUrl: current.siteUrl || homepageUrl,
        }));

        try {
            const response = await stockApi.discoverNewsRss({ homepage_url: homepageUrl });
            setDiscoveryResult(response.candidates);
            setDiscoveryCrawlResult(response.crawl_candidates);
            if (response.candidates.length > 0) {
                setDiscoveryMessage(`Found ${response.candidates.length} feed${response.candidates.length === 1 ? '' : 's'} from this homepage.`);
            } else if (response.crawl_candidates.length > 0) {
                setDiscoveryMessage(
                    `No RSS feeds found, but discovered ${response.crawl_candidates.length} stable section page${response.crawl_candidates.length === 1 ? '' : 's'} you can use for crawl setup.`
                );
                setOpenPanels((current) => ({
                    ...current,
                    crawlSource: true,
                }));
            } else {
                setDiscoveryMessage('No RSS or Atom feeds were detected for this homepage. You can still add a feed manually below or configure a crawl source.');
                setOpenPanels((current) => ({
                    ...current,
                    manualRss: true,
                }));
            }
        } catch (error) {
            setDiscoveryError(getErrorMessage(error));
            setDiscoveryMessage(null);
        } finally {
            setDiscoveryLoading(false);
        }
    };

    const handleUseCrawlCandidate = (candidate: NewsCrawlDiscoveryCandidate) => {
        setCrawlDraft((current) => ({
            ...current,
            listingUrl: candidate.listing_url,
            title: candidate.title || current.title,
            siteUrl: candidate.site_url || discoveryHomepageUrl.trim() || current.siteUrl,
        }));
        setOpenPanels((current) => ({
            ...current,
            crawlSource: true,
        }));
        setCrawlActionMessage(null);
        setCrawlActionError(null);
    };

    const handleAddRssSource = async (candidate?: NewsRssDiscoveryCandidate) => {
        if (!canManageSources) {
            return;
        }

        const feedUrl = candidate?.feed_url ?? rssDraft.feedUrl.trim();
        if (!feedUrl) {
            setRssActionError('Please provide an RSS feed URL.');
            return;
        }

        setRssActionError(null);
        setRssActionMessage(null);

        const payload: NewsRssSourceCreateRequest = {
            feed_url: feedUrl,
            site_url: candidate?.site_url || rssDraft.siteUrl.trim() || null,
            homepage_url: candidate ? discoveryHomepageUrl.trim() || null : rssDraft.homepageUrl.trim() || null,
            title: candidate?.title || rssDraft.title.trim() || null,
            enabled: true,
            poll_interval_minutes: Number(rssDraft.pollIntervalMinutes) || undefined,
            discovery_method: candidate?.discovery_method || 'manual',
        };

        try {
            await stockApi.createNewsRssSource(payload);
            setRssActionMessage('RSS source saved.');
            setRssDraft(emptyRssDraft);
            setDiscoveryResult((current) => current.filter((item) => item.feed_url !== feedUrl));
            await loadSourceManagement();
        } catch (error) {
            setRssActionError(getErrorMessage(error));
        }
    };

    const handleValidateRss = async () => {
        if (!canManageSources) {
            return;
        }
        const feedUrl = rssDraft.feedUrl.trim();
        if (!feedUrl) {
            setRssActionError('Please enter an RSS feed URL first.');
            return;
        }

        setRssActionError(null);
        setRssActionMessage(null);

        try {
            const response = await stockApi.validateNewsRss({
                feed_url: feedUrl,
                site_url: rssDraft.siteUrl.trim() || null,
                title: rssDraft.title.trim() || null,
                enabled: true,
                poll_interval_minutes: Number(rssDraft.pollIntervalMinutes) || undefined,
                homepage_url: rssDraft.homepageUrl.trim() || null,
                discovery_method: 'manual',
            });
            setRssActionMessage(response.message);
        } catch (error) {
            setRssActionError(getErrorMessage(error));
        }
    };

    const handleValidateCrawl = async () => {
        if (!canManageSources) {
            return;
        }
        if (!crawlDraft.listingUrl.trim() || !crawlDraft.articleLinkSelector.trim() || !crawlDraft.contentSelector.trim()) {
            setCrawlActionError('Please provide listing URL, article selector, and content selector.');
            return;
        }

        setCrawlActionError(null);
        setCrawlActionMessage(null);

        try {
            const response = await stockApi.validateNewsCrawl({
                listing_url: crawlDraft.listingUrl.trim(),
                article_link_selector: crawlDraft.articleLinkSelector.trim(),
                content_selector: crawlDraft.contentSelector.trim(),
                excerpt_selector: crawlDraft.excerptSelector.trim() || null,
                pagination_selector: crawlDraft.paginationSelector.trim() || null,
                title: crawlDraft.title.trim() || null,
                site_url: crawlDraft.siteUrl.trim() || null,
                enabled: true,
                poll_interval_minutes: Number(crawlDraft.pollIntervalMinutes) || undefined,
            });
            setCrawlActionMessage(response.message);
        } catch (error) {
            setCrawlActionError(getErrorMessage(error));
        }
    };

    const handleAddCrawlSource = async () => {
        if (!canManageSources) {
            return;
        }
        if (!crawlDraft.listingUrl.trim() || !crawlDraft.articleLinkSelector.trim() || !crawlDraft.contentSelector.trim()) {
            setCrawlActionError('Please provide listing URL, article selector, and content selector.');
            return;
        }

        setCrawlActionError(null);
        setCrawlActionMessage(null);

        const payload: NewsCrawlSourceCreateRequest = {
            listing_url: crawlDraft.listingUrl.trim(),
            article_link_selector: crawlDraft.articleLinkSelector.trim(),
            content_selector: crawlDraft.contentSelector.trim(),
            excerpt_selector: crawlDraft.excerptSelector.trim() || null,
            pagination_selector: crawlDraft.paginationSelector.trim() || null,
            title: crawlDraft.title.trim() || null,
            site_url: crawlDraft.siteUrl.trim() || null,
            enabled: true,
            poll_interval_minutes: Number(crawlDraft.pollIntervalMinutes) || undefined,
        };

        try {
            await stockApi.createNewsCrawlSource(payload);
            setCrawlActionMessage('Crawl source saved.');
            setCrawlDraft(emptyCrawlDraft);
            await loadSourceManagement();
        } catch (error) {
            setCrawlActionError(getErrorMessage(error));
        }
    };

    const handleToggleSource = async (source: NewsSourceSummary) => {
        if (!canManageSources) {
            return;
        }

        try {
            await stockApi.updateNewsSource(source.kind, source.id, {
                enabled: !source.enabled,
            });
            await loadSourceManagement();
        } catch (error) {
            setSourcesError(getErrorMessage(error));
        }
    };

    const handleDeleteSource = async (source: NewsSourceSummary) => {
        if (!canManageSources) {
            return;
        }
        if (!window.confirm(`Delete "${sourceLabel(source)}"?`)) {
            return;
        }

        try {
            await stockApi.deleteNewsSource(source.kind, source.id);
            await loadSourceManagement();
        } catch (error) {
            setSourcesError(getErrorMessage(error));
        }
    };

    const handleSavePreferences = async () => {
        if (!canManageSources) {
            return;
        }

        setPreferencesSaving(true);
        setPreferencesError(null);
        setPreferencesSuccess(null);

        try {
            const response = await stockApi.updateNewsPreferences({
                blocked_topics_text: preferencesDraft,
            });
            setPreferences(response);
            setPreferencesDraft(response.blocked_topics_text);
            setPreferencesSuccess('Blocked-topic preferences saved.');
            await loadNewsFeed(appliedFeedRef.current, false);
        } catch (error) {
            setPreferencesError(getErrorMessage(error));
        } finally {
            setPreferencesSaving(false);
        }
    };

    const handleOpenArticle = async (item: NewsFeedItem) => {
        setDetailLoadingId(item.id);
        setDetailError(null);
        try {
            const response = await stockApi.getNewsArticle(item.id);
            setNewsDetail(response);
        } catch (error) {
            setDetailError(getErrorMessage(error));
        } finally {
            setDetailLoadingId(null);
        }
    };

    const handleOpenRelatedArticle = async (articleId: number) => {
        setDetailLoadingId(articleId);
        setDetailError(null);
        try {
            const response = await stockApi.getNewsArticle(articleId);
            setNewsDetail(response);
        } catch (error) {
            setDetailError(getErrorMessage(error));
        } finally {
            setDetailLoadingId(null);
        }
    };

    const handleRefreshArticleContent = async () => {
        if (!newsDetail) {
            return;
        }

        setDetailRefreshLoading(true);
        setDetailError(null);
        try {
            const response = await stockApi.refreshNewsArticleContent(newsDetail.id);
            setNewsDetail(response);
            setFeedItems((current) =>
                current.map((item) =>
                    item.id === response.id
                        ? {
                              ...item,
                              title: response.title,
                              excerpt: response.excerpt,
                              original_excerpt: response.original_excerpt,
                              llm_summary: response.llm_summary,
                              canonical_url: response.canonical_url,
                              published_at: response.published_at,
                              language: response.language,
                              image_url: response.image_url,
                              source_labels: response.source_labels,
                              topics: response.topics,
                              tickers: response.tickers,
                              sectors: response.sectors,
                              importance: response.importance,
                              sentiment: response.sentiment,
                              event_type: response.event_type,
                              event_labels: response.event_labels,
                              matched_tickers: response.matched_tickers,
                              why_relevant: response.why_relevant,
                              story_key: response.story_key,
                              story_source_count: response.story_source_count,
                              source_title: response.source_title,
                              source_kind: response.source_kind,
                              is_filtered_for_user: response.is_filtered_for_user,
                          }
                        : item,
                ),
            );
        } catch (error) {
            setDetailError(getErrorMessage(error));
        } finally {
            setDetailRefreshLoading(false);
        }
    };

    const handleSummarizeArticle = async (item: NewsFeedItem) => {
        setSummaryLoadingById((current) => ({ ...current, [item.id]: true }));
        setSummaryErrorById((current) => ({ ...current, [item.id]: null }));

        try {
            const response: NewsArticleSummaryResponse = await stockApi.summarizeNewsArticle(item.id, {
                forceRefresh: Boolean(item.llm_summary?.trim()),
            });
            const summaryText = response.llm_summary?.trim() || response.excerpt?.trim() || '';
            if (!summaryText) {
                throw new Error('No summary was returned.');
            }

            setFeedItems((current) =>
                current.map((entry) =>
                    entry.id === item.id
                        ? {
                              ...entry,
                              llm_summary: summaryText,
                          }
                        : entry,
                ),
            );

            setNewsDetail((current) =>
                current && current.id === item.id
                    ? {
                          ...current,
                          llm_summary: summaryText,
                      }
                    : current,
            );
        } catch (error) {
            setSummaryErrorById((current) => ({ ...current, [item.id]: getErrorMessage(error) }));
        } finally {
            setSummaryLoadingById((current) => ({ ...current, [item.id]: false }));
        }
    };

    const sourceCards = activeNewsSources;

    const togglePanel = (panel: PanelKey) => {
        setOpenPanels((current) => ({
            ...current,
            [panel]: !current[panel],
        }));
    };

    const closeDetailModal = () => {
        setNewsDetail(null);
        setDetailError(null);
        setDetailRefreshLoading(false);
    };

    const renderSourceCard = (source: NewsSourceSummary) => {
        const validationClass = formatValidationBadge(source.validation_status);
        const kindLabel = source.kind === 'rss' ? 'RSS' : 'Crawl';
        const detail = source.kind === 'rss'
            ? (source as NewsRssSource).feed_url
            : (source as NewsCrawlSource).listing_url;
        return (
            <div key={`${source.kind}-${source.id}`} className="rounded-xl border border-base-300 bg-base-100 p-4 shadow-sm space-y-3">
                <div className="flex flex-wrap items-start justify-between gap-3">
                    <div className="min-w-0">
                        <div className="flex flex-wrap items-center gap-2">
                            <h4 className="font-semibold text-base-content">{sourceLabel(source)}</h4>
                            <span className={`badge ${validationClass}`}>{source.validation_status}</span>
                            <span className="badge badge-ghost">{kindLabel}</span>
                        </div>
                        <p className="mt-1 text-sm text-base-content/70 break-all">{detail}</p>
                        <p className="mt-1 text-xs text-base-content/60">
                            Poll every {source.poll_interval_minutes} minute{source.poll_interval_minutes === 1 ? '' : 's'}
                            {source.site_url ? ` · ${source.site_url}` : ''}
                        </p>
                    </div>
                    <div className="flex flex-wrap gap-2">
                        <button
                            type="button"
                            className="btn btn-sm btn-outline"
                            onClick={() => handleToggleSource(source)}
                        >
                            {source.enabled ? 'Disable' : 'Enable'}
                        </button>
                        <button
                            type="button"
                            className="btn btn-sm btn-ghost text-error"
                            onClick={() => handleDeleteSource(source)}
                        >
                            Delete
                        </button>
                    </div>
                </div>
                {source.last_error ? (
                    <div className="alert alert-warning py-2 text-sm">
                        <span>{source.last_error}</span>
                    </div>
                ) : null}
                <div className="flex flex-wrap gap-2 text-xs text-base-content/60">
                    <span>{source.enabled ? 'Enabled' : 'Disabled'}</span>
                    <span>Last validated: {formatDateTime(source.last_validated_at)}</span>
                    <span>Created: {formatDateTime(source.created_at)}</span>
                </div>
            </div>
        );
    };

    const getItemDisplayText = (item: NewsFeedItem) => item.llm_summary?.trim() || item.excerpt;
    const getItemSummaryLoading = (itemId: number) => Boolean(summaryLoadingById[itemId]);
    const getItemSummaryError = (itemId: number) => summaryErrorById[itemId] || null;

    return (
        <div className="mx-auto w-full max-w-[101.5rem] space-y-6">
            <section className="space-y-3">
                <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
                    <div className="space-y-1">
                        <h2 className="text-3xl font-semibold text-base-content">News</h2>
                        <p className="max-w-3xl text-sm text-base-content/70">
                            Browse the public feed, add your own RSS or crawl sources, and hide topics you do not care about.
                        </p>
                    </div>
                    <div className="flex flex-wrap gap-2 text-sm">
                        <div className="rounded-full border border-base-300 bg-base-100 px-3 py-2 text-base-content/75">
                            Mode: <span className="font-medium text-base-content">{feedPersonalized ? 'Personalized' : 'Public'}</span>
                        </div>
                        <div className="rounded-full border border-base-300 bg-base-100 px-3 py-2 text-base-content/75">
                            Articles: <span className="font-medium text-base-content">{feedCount.toLocaleString()}</span>
                        </div>
                        <div className="rounded-full border border-base-300 bg-base-100 px-3 py-2 text-base-content/75">
                            Sources: <span className="font-medium text-base-content">{activeNewsSources.length}</span>
                        </div>
                    </div>
                </div>
                {!canManageSources ? (
                    <div className="rounded-2xl border border-base-300 bg-base-100 px-4 py-3 text-sm text-base-content/70">
                        Sign in to add sources and manage blocked-topic filters. The public feed is still available below.
                    </div>
                ) : null}
            </section>

            <div className={`grid grid-cols-1 gap-6 items-start ${isUtilityRailOpen ? 'xl:grid-cols-[minmax(0,1fr)_minmax(18rem,24rem)]' : 'xl:grid-cols-1'}`}>
                <section className={`card bg-base-100 shadow-lg border border-base-300 w-full ${isUtilityRailOpen ? 'xl:max-w-[76rem] xl:justify-self-start' : 'xl:max-w-none xl:justify-self-stretch'}`}>
                    <div className="card-body gap-5">
                        <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
                            <div>
                                <h3 className="card-title text-2xl">Feed</h3>
                                <p className="text-sm text-base-content/70">
                                    {feedPersonalized
                                        ? 'Prioritize portfolio and bookmark-relevant stories, then drill into clustered coverage when something matters.'
                                        : 'This is the public latest view from the default source pack.'}
                                </p>
                            </div>
                            <div className="flex flex-wrap gap-2">
                                <button
                                    type="button"
                                    className="btn btn-ghost btn-sm"
                                    onClick={() => setIsUtilityRailOpen((current) => !current)}
                                >
                                    {isUtilityRailOpen ? 'Hide setup rail' : 'Show setup rail'}
                                </button>
                                <button type="button" className="btn btn-primary btn-sm" onClick={handleApplyFilters}>
                                    Apply filters
                                </button>
                                <button type="button" className="btn btn-ghost btn-sm" onClick={handleResetFilters}>
                                    Reset
                                </button>
                            </div>
                        </div>

                        <div className="flex flex-wrap gap-2">
                            <button
                                type="button"
                                className={`btn btn-sm ${activeFeedView === 'forYou' ? 'btn-primary' : 'btn-outline'}`}
                                onClick={() => void handleFeedViewChange('forYou')}
                                disabled={!canManageSources}
                            >
                                For You
                            </button>
                            <button
                                type="button"
                                className={`btn btn-sm ${activeFeedView === 'latest' ? 'btn-primary' : 'btn-outline'}`}
                                onClick={() => void handleFeedViewChange('latest')}
                            >
                                Latest
                            </button>
                            <button
                                type="button"
                                className={`btn btn-sm ${activeFeedView === 'portfolio' ? 'btn-primary' : 'btn-outline'}`}
                                onClick={() => void handleFeedViewChange('portfolio')}
                                disabled={!canManageSources}
                            >
                                Portfolio
                            </button>
                            <button
                                type="button"
                                className={`btn btn-sm ${activeFeedView === 'bookmarks' ? 'btn-primary' : 'btn-outline'}`}
                                onClick={() => void handleFeedViewChange('bookmarks')}
                                disabled={!canManageSources}
                            >
                                Bookmarks
                            </button>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-7 gap-3">
                            <label className="form-control">
                                <div className="label">
                                    <span className="label-text">Source</span>
                                </div>
                                <select
                                    className="select select-bordered"
                                    value={feedDraft.source}
                                    onChange={(event) => {
                                        void handleSourceFilterChange(event.target.value);
                                    }}
                                >
                                    <option value="">All sources</option>
                                    {sourceFilterOptions.map((option) => (
                                        <option key={option} value={option}>
                                            {option}
                                        </option>
                                    ))}
                                </select>
                            </label>
                            <label className="form-control">
                                <div className="label">
                                    <span className="label-text">Ticker</span>
                                </div>
                                <input
                                    type="text"
                                    className="input input-bordered uppercase"
                                    placeholder="VCB"
                                    value={feedDraft.ticker}
                                    onChange={(event) => setFeedDraft((current) => ({ ...current, ticker: event.target.value.toUpperCase() }))}
                                    onKeyDown={handleFeedInputKeyDown}
                                />
                            </label>
                            <label className="form-control">
                                <div className="label">
                                    <span className="label-text">Event</span>
                                </div>
                                <select
                                    className="select select-bordered"
                                    value={feedDraft.eventType}
                                    onChange={(event) => setFeedDraft((current) => ({ ...current, eventType: event.target.value as FeedDraft['eventType'] }))}
                                >
                                    <option value="">All events</option>
                                    {NEWS_EVENT_FILTER_OPTIONS.map((option) => (
                                        <option key={option.value} value={option.value}>{option.label}</option>
                                    ))}
                                </select>
                            </label>
                            <label className="form-control">
                                <div className="label">
                                    <span className="label-text">Importance</span>
                                </div>
                                <select
                                    className="select select-bordered"
                                    value={feedDraft.importance}
                                    onChange={(event) => setFeedDraft((current) => ({ ...current, importance: event.target.value as FeedDraft['importance'] }))}
                                >
                                    <option value="">All levels</option>
                                    {NEWS_IMPORTANCE_OPTIONS.map((option) => (
                                        <option key={option.value} value={option.value}>{option.label}</option>
                                    ))}
                                </select>
                            </label>
                            <label className="form-control">
                                <div className="label">
                                    <span className="label-text">Topic</span>
                                </div>
                                <input
                                    type="text"
                                    className="input input-bordered"
                                    placeholder="bank, interest rates, M&A"
                                    value={feedDraft.topic}
                                    onChange={(event) => setFeedDraft((current) => ({ ...current, topic: event.target.value }))}
                                    onKeyDown={handleFeedInputKeyDown}
                                />
                                <div className="label">
                                    <span className="label-text-alt">Partial match against topic tags, for example bank matches banking.</span>
                                </div>
                            </label>
                            <label className="form-control">
                                <div className="label">
                                    <span className="label-text">From</span>
                                </div>
                                <input
                                    type="date"
                                    className="input input-bordered"
                                    value={feedDraft.from}
                                    onChange={(event) => setFeedDraft((current) => ({ ...current, from: event.target.value }))}
                                />
                            </label>
                            <label className="form-control">
                                <div className="label">
                                    <span className="label-text">To</span>
                                </div>
                                <input
                                    type="date"
                                    className="input input-bordered"
                                    value={feedDraft.to}
                                    onChange={(event) => setFeedDraft((current) => ({ ...current, to: event.target.value }))}
                                />
                            </label>
                        </div>

                        {feedDraft.scope === 'bookmarks' ? (
                            <label className="form-control max-w-sm">
                                <div className="label">
                                    <span className="label-text">Bookmark group</span>
                                </div>
                                <select
                                    className="select select-bordered"
                                    value={feedDraft.bookmarkGroupId}
                                    onChange={(event) => setFeedDraft((current) => ({ ...current, bookmarkGroupId: event.target.value }))}
                                >
                                    <option value="">All bookmark groups</option>
                                    {bookmarkGroups.map((group) => (
                                        <option key={group.id} value={group.id}>
                                            {group.name} ({group.tickers.length})
                                        </option>
                                    ))}
                                </select>
                                <div className="label">
                                    <span className="label-text-alt">
                                        {bookmarkGroups.length > 0 ? 'Optional: narrow the bookmark scope to one group.' : 'No bookmark groups yet. Create them from the indices view.'}
                                    </span>
                                </div>
                            </label>
                        ) : null}

                        {feedError ? (
                            <div className="alert alert-error">
                                <span>{feedError}</span>
                            </div>
                        ) : null}

                        {feedLoading && feedItems.length === 0 ? (
                            <div className="flex flex-col items-center justify-center py-16 text-base-content/70">
                                <span className="loading loading-spinner loading-lg text-primary" />
                                <p className="mt-4">Loading latest news...</p>
                            </div>
                        ) : null}

                        {!feedLoading && feedItems.length === 0 ? (
                            <div className="rounded-xl border border-dashed border-base-300 p-8 text-center text-base-content/70">
                                No articles matched your current filters yet.
                            </div>
                        ) : null}

                        <div className="space-y-4">
                            {feedItems.map((item) => (
                                <article key={item.id} className="rounded-2xl border border-base-300 bg-base-100 p-4 shadow-sm">
                                    <div className="flex flex-col gap-4 lg:flex-row">
                                        {item.image_url ? (
                                            <div className="w-full lg:w-40 flex-shrink-0">
                                                <img
                                                    src={item.image_url}
                                                    alt={item.title}
                                                    className="h-40 w-full rounded-xl object-cover bg-base-200"
                                                />
                                            </div>
                                        ) : null}

                                        <div className="min-w-0 flex-1 space-y-3">
                                            <div className="flex flex-wrap items-start justify-between gap-3">
                                                <div className="min-w-0">
                                                    <div className="flex flex-wrap items-center gap-2">
                                                        <h4 className="text-lg font-semibold text-base-content">{item.title}</h4>
                                                        {item.is_filtered_for_user ? (
                                                            <span className="badge badge-warning">Filtered</span>
                                                        ) : null}
                                                    </div>
                                                    <p className="mt-1 text-sm text-base-content/70">
                                                        {item.source_title || joinTags(item.source_labels) || 'Unknown source'}
                                                        {item.published_at ? ` · ${formatRelativeTime(item.published_at)}` : ''}
                                                    </p>
                                                </div>
                                                <div className="flex flex-wrap gap-2">
                                                    <button
                                                        type="button"
                                                        className="btn btn-sm btn-outline"
                                                        onClick={() => void handleOpenArticle(item)}
                                                        disabled={detailLoadingId === item.id}
                                                    >
                                                        {detailLoadingId === item.id ? 'Opening...' : 'Open detail'}
                                                    </button>
                                                    <button
                                                        type="button"
                                                        className="btn btn-sm btn-outline"
                                                        onClick={() => void handleSummarizeArticle(item)}
                                                        disabled={getItemSummaryLoading(item.id)}
                                                    >
                                                        {getItemSummaryLoading(item.id) ? 'Summarizing...' : item.llm_summary?.trim() ? 'Regenerate' : 'Summary'}
                                                    </button>
                                                    <a
                                                        href={item.canonical_url}
                                                        target="_blank"
                                                        rel="noreferrer"
                                                        className="btn btn-sm btn-primary"
                                                    >
                                                        Open source
                                                    </a>
                                                </div>
                                            </div>

                                            {getItemDisplayText(item) ? (
                                                <p className="text-sm leading-6 text-base-content/85">{getItemDisplayText(item)}</p>
                                            ) : null}

                                            {getItemSummaryError(item.id) ? (
                                                <p className="text-xs text-error">{getItemSummaryError(item.id)}</p>
                                            ) : null}

                                            {item.why_relevant.length > 0 ? (
                                                <div className="flex flex-wrap gap-2">
                                                    {item.why_relevant.map((reason) => (
                                                        <span key={`${item.id}-${reason}`} className="badge badge-primary badge-outline">{reason}</span>
                                                    ))}
                                                </div>
                                            ) : null}

                                            <div className="flex flex-wrap gap-2">
                                                {item.event_labels.slice(0, 2).map((label) => (
                                                    <span key={label} className="badge badge-warning badge-outline">{label}</span>
                                                ))}
                                                {item.topics.slice(0, 5).map((topic) => (
                                                    <span key={topic} className="badge badge-outline">{topic}</span>
                                                ))}
                                                {item.matched_tickers.slice(0, 5).map((ticker) => (
                                                    <span key={`matched-${ticker}`} className="badge badge-primary">{ticker}</span>
                                                ))}
                                                {item.tickers.slice(0, 5).map((ticker) => (
                                                    <span key={ticker} className="badge badge-secondary badge-outline">{ticker}</span>
                                                ))}
                                                {item.sectors.slice(0, 3).map((sector) => (
                                                    <span key={sector} className="badge badge-accent badge-outline">{sector}</span>
                                                ))}
                                            </div>

                                            <div className="flex flex-wrap gap-3 text-xs text-base-content/60">
                                                <span>Published: {formatDateTime(item.published_at)}</span>
                                                {item.importance ? <span>Importance: {item.importance}</span> : null}
                                                {item.sentiment ? <span>Sentiment: {item.sentiment}</span> : null}
                                                {item.story_source_count > 1 ? <span>Story coverage: {item.story_source_count} sources</span> : null}
                                                {item.related_article_ids.length > 0 ? <span>Related coverage: {item.related_article_ids.length} more articles</span> : null}
                                            </div>
                                        </div>
                                    </div>
                                </article>
                            ))}
                        </div>

                        <div className="flex items-center justify-between gap-3">
                            <p className="text-sm text-base-content/60">
                                Showing {feedItems.length.toLocaleString()} of {feedCount.toLocaleString()} {feedDraft.groupBy === 'story' ? 'stories' : 'articles'}
                            </p>
                            <button
                                type="button"
                                className="btn btn-outline"
                                onClick={() => void handleLoadMore()}
                                disabled={!feedCursor || feedLoading}
                            >
                                {feedLoading ? 'Loading...' : feedCursor ? 'Load more' : 'No more articles'}
                            </button>
                        </div>
                    </div>
                </section>

                {isUtilityRailOpen ? (
                <div className="space-y-4 xl:w-full xl:max-w-[24rem] xl:justify-self-end">
                    {canManageSources ? (
                        <>
                            <CollapsiblePanel
                                title="RSS discovery"
                                description="Enter a homepage URL and we will look for discoverable RSS or Atom feeds."
                                isOpen={openPanels.rssDiscovery}
                                onToggle={() => togglePanel('rssDiscovery')}
                                action={discoveryResult.length > 0 ? <span className="badge badge-ghost badge-sm">{discoveryResult.length}</span> : null}
                            >
                                <label className="form-control">
                                    <div className="label">
                                        <span className="label-text">Homepage URL</span>
                                    </div>
                                    <input
                                        type="url"
                                        className="input input-bordered"
                                        placeholder="https://example.com"
                                        value={discoveryHomepageUrl}
                                        onChange={(event) => setDiscoveryHomepageUrl(event.target.value)}
                                    />
                                </label>

                                <div className="flex flex-wrap gap-2">
                                    <button
                                        type="button"
                                        className="btn btn-primary btn-sm"
                                        onClick={() => void handleDiscoverRss()}
                                        disabled={discoveryLoading}
                                    >
                                        {discoveryLoading ? 'Searching...' : 'Discover feeds'}
                                    </button>
                                </div>

                                {discoveryLoading || discoveryError || discoveryMessage || discoveryAttempted ? (
                                    <div
                                        className={`rounded-xl border p-4 ${
                                            discoveryError
                                                ? 'border-error/30 bg-error/10'
                                                : discoveryResult.length > 0
                                                    ? 'border-success/30 bg-success/10'
                                                    : 'border-info/30 bg-info/10'
                                        }`}
                                    >
                                        <div className="space-y-2">
                                            <div className="flex items-center gap-2">
                                                <span className="text-sm font-semibold">
                                                    {discoveryLoading
                                                        ? 'Searching homepage'
                                                        : discoveryError
                                                            ? 'Discovery failed'
                                                            : discoveryResult.length > 0
                                                                ? 'Feeds found'
                                                                : 'No feeds found'}
                                                </span>
                                                {discoveryResult.length > 0 && !discoveryLoading ? (
                                                    <span className="badge badge-success badge-sm">
                                                        {discoveryResult.length} result{discoveryResult.length === 1 ? '' : 's'}
                                                    </span>
                                                ) : null}
                                            </div>

                                            <p className="text-sm text-base-content/75">
                                                {discoveryLoading
                                                    ? 'Looking for RSS and Atom feeds from the homepage and validating each candidate.'
                                                    : discoveryError
                                                        ? discoveryError
                                                        : discoveryMessage || 'No RSS discovery result yet.'}
                                            </p>

                                            {!discoveryLoading && !discoveryError && discoveryResult.length === 0 && discoveryCrawlResult.length === 0 && discoveryAttempted ? (
                                                <div className="flex flex-wrap gap-2 pt-1">
                                                    <button
                                                        type="button"
                                                        className="btn btn-outline btn-sm"
                                                        onClick={() => setOpenPanels((current) => ({ ...current, manualRss: true }))}
                                                    >
                                                        Open manual RSS form
                                                    </button>
                                                </div>
                                            ) : null}
                                        </div>
                                    </div>
                                ) : null}

                                {discoveryResult.length > 0 ? (
                                    <div className="space-y-3">
                                        {discoveryResult.map((candidate) => (
                                            <div key={candidate.feed_url} className="rounded-xl border border-base-300 p-4">
                                                <div className="flex flex-col gap-3">
                                                    <div className="min-w-0">
                                                        <div className="flex flex-wrap items-center gap-2">
                                                            <h4 className="font-semibold">{candidate.title || 'Untitled feed'}</h4>
                                                            <span className="badge badge-ghost">{candidate.kind.toUpperCase()}</span>
                                                            <span className={`badge ${formatValidationBadge(candidate.validation_status)}`}>
                                                                {candidate.validation_status}
                                                            </span>
                                                        </div>
                                                        <p className="mt-1 break-all text-sm text-base-content/70">{candidate.feed_url}</p>
                                                        <p className="mt-1 text-xs text-base-content/60">
                                                            {candidate.category_hint || 'Discovered from homepage'}
                                                        </p>
                                                    </div>
                                                    <button
                                                        type="button"
                                                        className="btn btn-sm btn-outline self-start"
                                                        onClick={() => void handleAddRssSource(candidate)}
                                                    >
                                                        Add source
                                                    </button>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                ) : null}

                                {discoveryCrawlResult.length > 0 ? (
                                    <div className="space-y-3 pt-2">
                                        <div className="flex items-center gap-2">
                                            <h4 className="text-sm font-semibold text-base-content/80">Crawl Suggestions</h4>
                                            <span className="badge badge-outline badge-sm">
                                                {discoveryCrawlResult.length} candidate{discoveryCrawlResult.length === 1 ? '' : 's'}
                                            </span>
                                        </div>
                                        {discoveryCrawlResult.map((candidate) => (
                                            <div key={candidate.listing_url} className="rounded-xl border border-base-300 p-4">
                                                <div className="flex flex-col gap-3">
                                                    <div className="min-w-0">
                                                        <div className="flex flex-wrap items-center gap-2">
                                                            <h4 className="font-semibold">{candidate.title || 'Suggested listing page'}</h4>
                                                            <span className="badge badge-ghost">CRAWL</span>
                                                        </div>
                                                        <p className="mt-1 break-all text-sm text-base-content/70">{candidate.listing_url}</p>
                                                        <p className="mt-1 text-xs text-base-content/60">
                                                            Stable section page suggested from sitemap discovery.
                                                        </p>
                                                    </div>
                                                    <button
                                                        type="button"
                                                        className="btn btn-sm btn-outline self-start"
                                                        onClick={() => handleUseCrawlCandidate(candidate)}
                                                    >
                                                        Use for crawl setup
                                                    </button>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                ) : null}
                            </CollapsiblePanel>

                            <CollapsiblePanel
                                title="Manual RSS source"
                                description="Keep this as the fallback when discovery does not find the feed you want."
                                isOpen={openPanels.manualRss}
                                onToggle={() => togglePanel('manualRss')}
                            >
                                <div className="grid grid-cols-1 gap-3">
                                    <label className="form-control">
                                        <div className="label">
                                            <span className="label-text">Feed URL</span>
                                        </div>
                                        <input
                                            type="url"
                                            className="input input-bordered"
                                            value={rssDraft.feedUrl}
                                            onChange={(event) => setRssDraft((current) => ({ ...current, feedUrl: event.target.value }))}
                                        />
                                    </label>
                                    <label className="form-control">
                                        <div className="label">
                                            <span className="label-text">Site URL</span>
                                        </div>
                                        <input
                                            type="url"
                                            className="input input-bordered"
                                            value={rssDraft.siteUrl}
                                            onChange={(event) => setRssDraft((current) => ({ ...current, siteUrl: event.target.value }))}
                                        />
                                    </label>
                                    <label className="form-control">
                                        <div className="label">
                                            <span className="label-text">Homepage URL</span>
                                        </div>
                                        <input
                                            type="url"
                                            className="input input-bordered"
                                            value={rssDraft.homepageUrl}
                                            onChange={(event) => setRssDraft((current) => ({ ...current, homepageUrl: event.target.value }))}
                                        />
                                    </label>
                                    <label className="form-control">
                                        <div className="label">
                                            <span className="label-text">Title</span>
                                        </div>
                                        <input
                                            type="text"
                                            className="input input-bordered"
                                            value={rssDraft.title}
                                            onChange={(event) => setRssDraft((current) => ({ ...current, title: event.target.value }))}
                                        />
                                    </label>
                                    <label className="form-control">
                                        <div className="label">
                                            <span className="label-text">Poll interval minutes</span>
                                        </div>
                                        <input
                                            type="number"
                                            min="5"
                                            className="input input-bordered"
                                            placeholder="Backend default"
                                            value={rssDraft.pollIntervalMinutes}
                                            onChange={(event) => setRssDraft((current) => ({ ...current, pollIntervalMinutes: event.target.value }))}
                                        />
                                        <div className="label">
                                            <span className="label-text-alt text-base-content/60">Leave blank to use the backend default configured in admin.</span>
                                        </div>
                                    </label>
                                </div>

                                <div className="flex flex-wrap gap-2">
                                    <button type="button" className="btn btn-outline btn-sm" onClick={() => void handleValidateRss()}>
                                        Validate
                                    </button>
                                    <button type="button" className="btn btn-primary btn-sm" onClick={() => void handleAddRssSource()}>
                                        Add RSS source
                                    </button>
                                </div>

                                {rssActionError ? (
                                    <div className="alert alert-error">
                                        <span>{rssActionError}</span>
                                    </div>
                                ) : null}
                                {rssActionMessage ? (
                                    <div className="alert alert-success">
                                        <span>{rssActionMessage}</span>
                                    </div>
                                ) : null}
                            </CollapsiblePanel>

                            <CollapsiblePanel
                                title="Crawl source"
                                description="Use this when a site does not expose RSS feeds but still has stable article pages."
                                isOpen={openPanels.crawlSource}
                                onToggle={() => togglePanel('crawlSource')}
                            >
                                <div className="grid grid-cols-1 gap-3">
                                    <label className="form-control">
                                        <div className="label">
                                            <span className="label-text">Listing URL</span>
                                        </div>
                                        <input
                                            type="url"
                                            className="input input-bordered"
                                            value={crawlDraft.listingUrl}
                                            onChange={(event) => setCrawlDraft((current) => ({ ...current, listingUrl: event.target.value }))}
                                        />
                                    </label>
                                    <label className="form-control">
                                        <div className="label">
                                            <span className="label-text">Article link selector</span>
                                        </div>
                                        <input
                                            type="text"
                                            className="input input-bordered font-mono text-sm"
                                            value={crawlDraft.articleLinkSelector}
                                            onChange={(event) => setCrawlDraft((current) => ({ ...current, articleLinkSelector: event.target.value }))}
                                            placeholder="a.article-link"
                                        />
                                    </label>
                                    <label className="form-control">
                                        <div className="label">
                                            <span className="label-text">Content selector</span>
                                        </div>
                                        <input
                                            type="text"
                                            className="input input-bordered font-mono text-sm"
                                            value={crawlDraft.contentSelector}
                                            onChange={(event) => setCrawlDraft((current) => ({ ...current, contentSelector: event.target.value }))}
                                            placeholder=".article-content"
                                        />
                                    </label>
                                    <label className="form-control">
                                        <div className="label">
                                            <span className="label-text">Excerpt selector</span>
                                        </div>
                                        <input
                                            type="text"
                                            className="input input-bordered font-mono text-sm"
                                            value={crawlDraft.excerptSelector}
                                            onChange={(event) => setCrawlDraft((current) => ({ ...current, excerptSelector: event.target.value }))}
                                            placeholder=".article-summary"
                                        />
                                    </label>
                                    <label className="form-control">
                                        <div className="label">
                                            <span className="label-text">Pagination selector</span>
                                        </div>
                                        <input
                                            type="text"
                                            className="input input-bordered font-mono text-sm"
                                            value={crawlDraft.paginationSelector}
                                            onChange={(event) => setCrawlDraft((current) => ({ ...current, paginationSelector: event.target.value }))}
                                            placeholder="a.next"
                                        />
                                    </label>
                                    <label className="form-control">
                                        <div className="label">
                                            <span className="label-text">Title</span>
                                        </div>
                                        <input
                                            type="text"
                                            className="input input-bordered"
                                            value={crawlDraft.title}
                                            onChange={(event) => setCrawlDraft((current) => ({ ...current, title: event.target.value }))}
                                        />
                                    </label>
                                    <label className="form-control">
                                        <div className="label">
                                            <span className="label-text">Site URL</span>
                                        </div>
                                        <input
                                            type="url"
                                            className="input input-bordered"
                                            value={crawlDraft.siteUrl}
                                            onChange={(event) => setCrawlDraft((current) => ({ ...current, siteUrl: event.target.value }))}
                                        />
                                    </label>
                                    <label className="form-control">
                                        <div className="label">
                                            <span className="label-text">Poll interval minutes</span>
                                        </div>
                                        <input
                                            type="number"
                                            min="5"
                                            className="input input-bordered"
                                            placeholder="Backend default"
                                            value={crawlDraft.pollIntervalMinutes}
                                            onChange={(event) => setCrawlDraft((current) => ({ ...current, pollIntervalMinutes: event.target.value }))}
                                        />
                                        <div className="label">
                                            <span className="label-text-alt text-base-content/60">Leave blank to use the backend default configured in admin.</span>
                                        </div>
                                    </label>
                                </div>

                                <div className="flex flex-wrap gap-2">
                                    <button type="button" className="btn btn-outline btn-sm" onClick={() => void handleValidateCrawl()}>
                                        Validate
                                    </button>
                                    <button type="button" className="btn btn-primary btn-sm" onClick={() => void handleAddCrawlSource()}>
                                        Add crawl source
                                    </button>
                                </div>

                                {crawlActionError ? (
                                    <div className="alert alert-error">
                                        <span>{crawlActionError}</span>
                                    </div>
                                ) : null}
                                {crawlActionMessage ? (
                                    <div className="alert alert-success">
                                        <span>{crawlActionMessage}</span>
                                    </div>
                                ) : null}
                            </CollapsiblePanel>
                        </>
                    ) : null}

                    <CollapsiblePanel
                        title="Blocked topics"
                        description="Add the topics you do not want in your feed. The backend will normalize these into semantic blocks."
                        isOpen={openPanels.blockedTopics}
                        onToggle={() => togglePanel('blockedTopics')}
                        action={preferences?.blocked_labels?.length ? <span className="badge badge-ghost badge-sm">{preferences.blocked_labels.length}</span> : null}
                    >
                        {canManageSources ? (
                            <>
                                <label className="form-control">
                                    <div className="label">
                                        <span className="label-text">Topics to hide</span>
                                    </div>
                                    <textarea
                                        className="textarea textarea-bordered min-h-32"
                                        value={preferencesDraft}
                                        onChange={(event) => setPreferencesDraft(event.target.value)}
                                        placeholder="crypto, celebrity gossip, layoffs"
                                    />
                                </label>

                                <div className="flex flex-wrap gap-2">
                                    <button
                                        type="button"
                                        className="btn btn-primary btn-sm"
                                        onClick={() => void handleSavePreferences()}
                                        disabled={preferencesSaving}
                                    >
                                        {preferencesSaving ? 'Saving...' : 'Save filters'}
                                    </button>
                                </div>

                                {preferencesLoading ? (
                                    <div className="text-sm text-base-content/60">Loading preferences...</div>
                                ) : null}
                                {preferencesError ? (
                                    <div className="alert alert-error">
                                        <span>{preferencesError}</span>
                                    </div>
                                ) : null}
                                {preferencesSuccess ? (
                                    <div className="alert alert-success">
                                        <span>{preferencesSuccess}</span>
                                    </div>
                                ) : null}

                                <div className="flex flex-wrap gap-2">
                                    {(preferences?.blocked_labels || []).map((label) => (
                                        <span key={label} className="badge badge-outline">{label}</span>
                                    ))}
                                </div>
                            </>
                        ) : (
                            <div className="alert alert-info">
                                <span>Sign in to manage your blocked-topic profile.</span>
                            </div>
                        )}
                    </CollapsiblePanel>

                    <CollapsiblePanel
                        title="Sources"
                        description="Enabled sources and their validation state."
                        isOpen={openPanels.sources}
                        onToggle={() => togglePanel('sources')}
                        action={sourcesLoading ? <span className="loading loading-spinner loading-sm text-primary" /> : sourceCards.length ? <span className="badge badge-ghost badge-sm">{sourceCards.length}</span> : null}
                    >
                        {sourcesError ? (
                            <div className="alert alert-error">
                                <span>{sourcesError}</span>
                            </div>
                        ) : null}

                        {!sourcesLoading && sourceCards.length === 0 ? (
                            <div className="rounded-xl border border-dashed border-base-300 p-6 text-center text-sm text-base-content/70">
                                No private sources yet.
                            </div>
                        ) : null}

                        <div className="space-y-3">
                            {sourceCards.map((source) => renderSourceCard(source))}
                        </div>

                        {canManageSources ? (
                            <div className="text-xs text-base-content/60">
                                Private subscriptions are stored per user and merged into the personalized feed automatically.
                            </div>
                        ) : null}
                    </CollapsiblePanel>
                </div>
                ) : null}
            </div>

            {detailError ? (
                <div className="alert alert-error">
                    <span>{detailError}</span>
                </div>
            ) : null}

            <dialog ref={detailDialogRef} className="modal" onClose={closeDetailModal}>
                <div className="modal-box max-w-4xl">
                    {newsDetail ? (
                        <div className="space-y-4">
                            <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                                <div className="space-y-2">
                                    <div className="flex flex-wrap items-center gap-2">
                                        <h3 className="card-title text-2xl">{newsDetail.title}</h3>
                                        {newsDetail.source_kind ? <span className="badge badge-ghost">{newsDetail.source_kind}</span> : null}
                                    </div>
                                    <p className="text-sm text-base-content/70">
                                        {newsDetail.source_title || joinTags(newsDetail.source_labels)}
                                        {newsDetail.published_at ? ` · ${formatDateTime(newsDetail.published_at)}` : ''}
                                    </p>
                                </div>
                                <div className="flex flex-wrap gap-2">
                                    <button
                                        type="button"
                                        className="btn btn-outline btn-sm"
                                        onClick={() => void handleRefreshArticleContent()}
                                        disabled={detailRefreshLoading}
                                    >
                                        {detailRefreshLoading ? 'Refreshing...' : 'Refresh content'}
                                    </button>
                                    <button type="button" className="btn btn-ghost btn-sm" onClick={closeDetailModal}>
                                        Close
                                    </button>
                                </div>
                            </div>

                            {detailError ? (
                                <div className="alert alert-error">
                                    <span>{detailError}</span>
                                </div>
                            ) : null}

                            {newsDetail.content_text ? (
                                <div className="max-h-[55vh] overflow-y-auto rounded-2xl border border-base-300 bg-base-100/80 p-4">
                                    <p className="whitespace-pre-wrap leading-7 text-base-content/85">{newsDetail.content_text}</p>
                                </div>
                            ) : (
                                <p className="text-sm text-base-content/60">No article body was returned.</p>
                            )}

                            {newsDetail.why_relevant.length > 0 ? (
                                <div className="flex flex-wrap gap-2">
                                    {newsDetail.why_relevant.map((reason) => (
                                        <span key={reason} className="badge badge-primary badge-outline">{reason}</span>
                                    ))}
                                </div>
                            ) : null}

                            <div className="flex flex-wrap gap-2">
                                {newsDetail.event_labels.map((label) => (
                                    <span key={label} className="badge badge-warning badge-outline">{label}</span>
                                ))}
                                {newsDetail.topics.map((topic) => (
                                    <span key={topic} className="badge badge-outline">{topic}</span>
                                ))}
                                {newsDetail.matched_tickers.map((ticker) => (
                                    <span key={`matched-${ticker}`} className="badge badge-primary">{ticker}</span>
                                ))}
                                {newsDetail.tickers.map((ticker) => (
                                    <span key={ticker} className="badge badge-secondary badge-outline">{ticker}</span>
                                ))}
                                {newsDetail.sectors.map((sector) => (
                                    <span key={sector} className="badge badge-accent badge-outline">{sector}</span>
                                ))}
                            </div>

                            <div className="flex flex-wrap gap-3 text-xs text-base-content/60">
                                {newsDetail.importance ? <span>Importance: {newsDetail.importance}</span> : null}
                                {newsDetail.sentiment ? <span>Sentiment: {newsDetail.sentiment}</span> : null}
                                {newsDetail.story_source_count > 1 ? <span>Story coverage: {newsDetail.story_source_count} sources</span> : null}
                            </div>

                            <div className="flex flex-wrap items-center gap-3">
                                <a href={newsDetail.canonical_url} target="_blank" rel="noreferrer" className="btn btn-primary btn-sm">
                                    Open source article
                                </a>
                                <div className="text-xs text-base-content/60">
                                    Source URLs: {newsDetail.source_urls.join(' · ')}
                                </div>
                            </div>

                            {newsDetail.related_articles.length > 0 ? (
                                <div className="space-y-3 rounded-2xl border border-base-300 bg-base-100/80 p-4">
                                    <div className="flex flex-wrap items-center justify-between gap-2">
                                        <h4 className="font-semibold text-base-content">Related Coverage</h4>
                                        <span className="badge badge-outline">{newsDetail.related_articles.length}</span>
                                    </div>
                                    <div className="space-y-2">
                                        {newsDetail.related_articles.map((related) => (
                                            <div key={related.id} className="flex flex-col gap-2 rounded-xl border border-base-300 p-3 md:flex-row md:items-center md:justify-between">
                                                <div className="min-w-0">
                                                    <p className="font-medium text-base-content">{related.title}</p>
                                                    <p className="text-xs text-base-content/60">
                                                        {related.source_title || 'Unknown source'}
                                                        {related.published_at ? ` · ${formatDateTime(related.published_at)}` : ''}
                                                    </p>
                                                </div>
                                                <div className="flex flex-wrap gap-2">
                                                    <button
                                                        type="button"
                                                        className="btn btn-sm btn-outline"
                                                        onClick={() => void handleOpenRelatedArticle(related.id)}
                                                        disabled={detailLoadingId === related.id}
                                                    >
                                                        {detailLoadingId === related.id ? 'Opening...' : 'Open detail'}
                                                    </button>
                                                    <a href={related.canonical_url} target="_blank" rel="noreferrer" className="btn btn-sm btn-ghost">
                                                        Open source
                                                    </a>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            ) : null}
                        </div>
                    ) : (
                        <div className="py-6 text-sm text-base-content/60">No article detail loaded.</div>
                    )}
                </div>
                <form method="dialog" className="modal-backdrop">
                    <button onClick={closeDetailModal}>close</button>
                </form>
            </dialog>
        </div>
    );
};

export default NewsTab;
