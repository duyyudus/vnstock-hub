import axios from 'axios';

const AUTH_TOKEN_KEY = 'vnstock_auth_token';
const AUTH_USER_KEY = 'vnstock_auth_user';
const AUTH_EXPIRES_AT_KEY = 'vnstock_auth_expires_at';
export const AUTH_EVENT = 'vnstock:auth';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// Create axios instance with base configuration
const apiClient = axios.create({
    baseURL: `${API_BASE_URL}/api/v1`,
    headers: {
        'Content-Type': 'application/json',
    },
});

const getStoredToken = () => {
    if (typeof window === 'undefined') {
        return null;
    }
    return window.localStorage.getItem(AUTH_TOKEN_KEY);
};

const sanitizeFilenamePart = (value: string) => {
    const normalized = value.replace(/[^A-Za-z0-9._-]+/g, '_').replace(/^[._-]+|[._-]+$/g, '');
    return normalized || 'user';
};

const buildPortfolioCsvFallbackFilename = () => {
    const user = authStorage.getUser();
    const emailLocalPart = user?.email?.split('@')[0] ?? 'user';
    const userPart = sanitizeFilenamePart(emailLocalPart);
    const datePart = new Date().toISOString().slice(0, 10);
    return `${userPart}_${datePart}.csv`;
};

const decodeBase64Url = (value: string) => {
    const normalized = value.replace(/-/g, '+').replace(/_/g, '/');
    const padding = normalized.length % 4;
    const padded = padding ? `${normalized}${'='.repeat(4 - padding)}` : normalized;
    return window.atob(padded);
};

const parseJwtPayload = (token: string) => {
    if (typeof window === 'undefined') {
        return null;
    }
    const parts = token.split('.');
    if (parts.length < 2) {
        return null;
    }
    try {
        return JSON.parse(decodeBase64Url(parts[1])) as { exp?: number };
    } catch {
        return null;
    }
};

const getAuthExpiresAtMs = () => {
    if (typeof window === 'undefined') {
        return null;
    }
    const raw = window.localStorage.getItem(AUTH_EXPIRES_AT_KEY);
    if (raw) {
        const parsed = Number(raw);
        if (Number.isFinite(parsed)) {
            return parsed;
        }
    }
    const token = getStoredToken();
    if (!token) {
        return null;
    }
    const payload = parseJwtPayload(token);
    if (payload?.exp) {
        const expiresAt = payload.exp * 1000;
        if (Number.isFinite(expiresAt)) {
            window.localStorage.setItem(AUTH_EXPIRES_AT_KEY, String(expiresAt));
            return expiresAt;
        }
    }
    return null;
};

let logoutTimerId: number | null = null;

const clearLogoutTimer = () => {
    if (typeof window === 'undefined') {
        return;
    }
    if (logoutTimerId !== null) {
        window.clearTimeout(logoutTimerId);
        logoutTimerId = null;
    }
};

const scheduleLogout = (expiresAtMs: number | null) => {
    if (typeof window === 'undefined') {
        return;
    }
    clearLogoutTimer();
    if (!expiresAtMs) {
        return;
    }
    const remainingMs = expiresAtMs - Date.now();
    if (remainingMs <= 0) {
        authStorage.clearAll();
        return;
    }
    logoutTimerId = window.setTimeout(() => {
        authStorage.clearAll();
    }, remainingMs);
};

const notifyAuthChange = () => {
    if (typeof window === 'undefined') {
        return;
    }
    window.dispatchEvent(new CustomEvent(AUTH_EVENT));
};

const normalizeAuthUser = (user: AuthUser): AuthUser => {
    return {
        ...user,
        download_folder: user.download_folder ?? null,
        company_export_category: user.company_export_category ?? null,
        finance_export_category: user.finance_export_category ?? null,
    };
};

export const authStorage = {
    getToken() {
        return getStoredToken();
    },
    getExpiresAt() {
        return getAuthExpiresAtMs();
    },
    setToken(token: string) {
        if (typeof window === 'undefined') {
            return;
        }
        window.localStorage.setItem(AUTH_TOKEN_KEY, token);
        window.localStorage.removeItem(AUTH_EXPIRES_AT_KEY);
        const expiresAt = getAuthExpiresAtMs();
        scheduleLogout(expiresAt);
        notifyAuthChange();
    },
    setSession(token: string, user: AuthUser, expiresInSeconds: number) {
        if (typeof window === 'undefined') {
            return;
        }
        const normalizedUser = normalizeAuthUser(user);
        const expiresAt = Date.now() + expiresInSeconds * 1000;
        window.localStorage.setItem(AUTH_TOKEN_KEY, token);
        window.localStorage.setItem(AUTH_USER_KEY, JSON.stringify(normalizedUser));
        window.localStorage.setItem(AUTH_EXPIRES_AT_KEY, String(expiresAt));
        scheduleLogout(expiresAt);
        notifyAuthChange();
    },
    clearToken() {
        if (typeof window === 'undefined') {
            return;
        }
        window.localStorage.removeItem(AUTH_TOKEN_KEY);
        window.localStorage.removeItem(AUTH_EXPIRES_AT_KEY);
        clearLogoutTimer();
        notifyAuthChange();
    },
    getUser() {
        if (typeof window === 'undefined') {
            return null;
        }
        const raw = window.localStorage.getItem(AUTH_USER_KEY);
        if (!raw) {
            return null;
        }
        try {
            return normalizeAuthUser(JSON.parse(raw) as AuthUser);
        } catch {
            return null;
        }
    },
    setUser(user: AuthUser) {
        if (typeof window === 'undefined') {
            return;
        }
        window.localStorage.setItem(AUTH_USER_KEY, JSON.stringify(normalizeAuthUser(user)));
        notifyAuthChange();
    },
    clearUser() {
        if (typeof window === 'undefined') {
            return;
        }
        window.localStorage.removeItem(AUTH_USER_KEY);
        notifyAuthChange();
    },
    clearAll() {
        if (typeof window === 'undefined') {
            return;
        }
        window.localStorage.removeItem(AUTH_TOKEN_KEY);
        window.localStorage.removeItem(AUTH_USER_KEY);
        window.localStorage.removeItem(AUTH_EXPIRES_AT_KEY);
        clearLogoutTimer();
        notifyAuthChange();
    }
};

apiClient.interceptors.request.use((config) => {
    const token = getStoredToken();
    if (token) {
        config.headers = config.headers || {};
        config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
});

apiClient.interceptors.response.use(
    (response) => response,
    (error) => {
        if (error?.response?.status === 401) {
            authStorage.clearAll();
        }
        return Promise.reject(error);
    }
);

const primeAuthSession = () => {
    if (typeof window === 'undefined') {
        return;
    }
    const token = getStoredToken();
    if (!token) {
        return;
    }
    const expiresAt = getAuthExpiresAtMs();
    if (!expiresAt) {
        return;
    }
    if (expiresAt <= Date.now()) {
        authStorage.clearAll();
        return;
    }
    scheduleLogout(expiresAt);
};

primeAuthSession();

const buildStockRangeParams = (options?: StockListRangeParams): URLSearchParams => {
    const params = new URLSearchParams();
    if (options?.rangeStart) {
        params.set('range_start', options.rangeStart);
    }
    if (options?.rangeEnd) {
        params.set('range_end', options.rangeEnd);
    }
    return params;
};

const buildNewsQueryParams = (options?: NewsFeedQuery): URLSearchParams => {
    const params = new URLSearchParams();
    if (options?.source) {
        params.set('source', options.source);
    }
    if (options?.topic) {
        params.set('topic', options.topic);
    }
    if (options?.ticker) {
        params.set('ticker', options.ticker);
    }
    if (options?.from) {
        params.set('from', options.from);
    }
    if (options?.to) {
        params.set('to', options.to);
    }
    if (options?.sort) {
        params.set('sort', options.sort);
    }
    if (options?.scope) {
        params.set('scope', options.scope);
    }
    if (typeof options?.bookmark_group_id === 'number') {
        params.set('bookmark_group_id', String(options.bookmark_group_id));
    }
    if (options?.event_type) {
        params.set('event_type', options.event_type);
    }
    if (options?.importance) {
        params.set('importance', options.importance);
    }
    if (options?.group_by) {
        params.set('group_by', options.group_by);
    }
    if (options?.cursor) {
        params.set('cursor', options.cursor);
    }
    if (typeof options?.limit === 'number') {
        params.set('limit', String(options.limit));
    }
    return params;
};

// App info
export interface AppInfoResponse {
    backend_version: string;
    build_number: string;
}

// Stock data types
export interface Stock {
    ticker: string;
    price: number;
    market_cap: number;
    company_name: string;
    exchange: string;
    charter_capital: number;
    pe_ratio: number | null;
    accumulated_value: number | null;  // In billion VND
    foreign_buy_value: number | null;  // In billion VND
    foreign_sell_value: number | null;  // In billion VND
    current_room: number | null;
    total_room: number | null;
    price_change_24h: number | null;
    price_change_1w: number | null;
    price_change_1m: number | null;
    price_change_6m?: number | null;
    price_change_1y: number | null;
    price_change_2y?: number | null;
    price_change_3y?: number | null;
    atl_price?: number | null;
    atl_date?: string | null;
    atl_diff_pct?: number | null;
    ath_price?: number | null;
    ath_date?: string | null;
    ath_diff_pct?: number | null;
    industry: string;  // ICB Level 2 industry classification
}

export interface StockListRangeParams {
    rangeStart?: string;
    rangeEnd?: string;
}

export interface BookmarkGroup {
    id: number;
    name: string;
    tickers: string[];
    created_at: string | null;
    updated_at: string | null;
}

export interface BookmarkGroupsResponse {
    groups: BookmarkGroup[];
    count: number;
}

export interface BookmarkGroupCreateRequest {
    name: string;
}

export interface BookmarkGroupUpdateRequest {
    name: string;
}

export interface BookmarkGroupStocksResponse {
    stocks: Stock[];
    count: number;
    group_id: number;
    group_name: string;
}

export interface IndexStocksResponse {
    stocks: Stock[];
    count: number;
}

export interface IndexInfo {
    symbol: string;
    name: string;
    group: string | null;
    description: string | null;
}

export interface IndexListResponse {
    indices: IndexInfo[];
    count: number;
}

export interface IndustryInfo {
    name: string;
    en_name: string;
    code: string;
    family_code?: string | null;
    family_name?: string | null;
    family_en_name?: string | null;
}

export interface IndustryListResponse {
    industries: IndustryInfo[];
    count: number;
}

export interface IndexValueInfo {
    symbol: string;
    name: string;
    value: number;
    change: number;
    change_value: number;
}

export interface IndexValuesResponse {
    indices: IndexValueInfo[];
    count: number;
}

export interface IndustryStocksResponse {
    stocks: Stock[];
    count: number;
    industry_name: string;
}

export interface FinancialDataResponse {
    symbol: string;
    data: Record<string, string | number | boolean | null>[];
    count: number;
}

export interface VolumeDataPoint {
    date: string;
    volume: number;
    // API response value already normalized for UI display.
    // `value` is derived server-side from cached DB fields and returned in billion VND.
    value: number | null;
    matched_volume: number | null;
    // These `*_value` fields are also API-normalized to billion VND, not raw DB units.
    matched_value: number | null;
    deal_volume: number | null;
    deal_value: number | null;
    total_volume: number | null;
    total_value: number | null;
    foreign_net_value: number | null;
    prop_buy_value: number | null;
    prop_sell_value: number | null;
    prop_net_value: number | null;
}

export interface VolumeHistoryResponse {
    symbol: string;
    company_name: string;
    data: VolumeDataPoint[];
    count: number;
    sync_performed: boolean;
    sync_timed_out: boolean;
    sync_error: string | null;
    updated_through: string | null;
    repaired_missing_dates: number;
}

export interface PriceDataPoint {
    date: string;
    close: number;
}

export interface PriceHistoryResponse {
    symbol: string;
    company_name: string;
    data: PriceDataPoint[];
    count: number;
    sync_performed: boolean;
    sync_timed_out: boolean;
    sync_error: string | null;
    updated_through: string | null;
    repaired_missing_dates: number;
}

export interface HistoryRequestOptions {
    autoSync?: boolean;
}

export interface OhlcvDataPoint {
    date: string;
    open: number | null;
    high: number | null;
    low: number | null;
    close: number;
    volume: number | null;
}

export interface OhlcvHistoryResponse {
    symbol: string;
    company_name: string;
    data: OhlcvDataPoint[];
    count: number;
}

// Weekly prices types for growth chart
export interface WeeklyPricePoint {
    date: string;
    close: number;
}

export interface StockWeeklyPriceData {
    symbol: string;
    ticker: string;
    company_name: string;
    prices: WeeklyPricePoint[];
}

export interface StocksWeeklyPricesResponse {
    stocks: StockWeeklyPriceData[];
    benchmarks: {
        VNINDEX?: WeeklyPricePoint[];
        VN30?: WeeklyPricePoint[];
    };
    start_date: string;
    end_date: string;
    is_stale: boolean;
    is_syncing: boolean;
}

export interface VolumeSeriesPoint {
    date: string;
    value: number | null;
}

export interface StockVolumeSeriesData {
    symbol: string;
    ticker: string;
    company_name: string;
    data: VolumeSeriesPoint[];
}

export interface StocksVolumeSeriesResponse {
    stocks: StockVolumeSeriesData[];
    start_date: string;
    end_date: string;
    is_stale: boolean;
    is_syncing: boolean;
}

// Fund data types
export interface FundDataResponse {
    symbol?: string;
    data: Record<string, string | number | boolean | null>[];
    count: number;
}

// Fund Performance Types
export interface FundRiskMetrics {
    annualized_return: number | null;
    annualized_volatility: number | null;
    sharpe_ratio: number | null;
}

export interface FundReturns {
    ytd?: number;
    '1y'?: number | null;
    '3y'?: number | null;
    '5y'?: number | null;
    'all'?: number | null;
}

export interface NavHistoryPoint {
    date: string;
    normalized_nav: number;
    raw_nav: number;
}

export interface FundPerformanceMetrics {
    symbol: string;
    name: string;
    data_start_date: string;
    nav_history: NavHistoryPoint[];
    returns: FundReturns;
    risk_metrics: FundRiskMetrics;
    yearly_returns: Record<string, number>;
}

export interface FundPerformanceData {
    funds: FundPerformanceMetrics[];
    benchmarks: Record<string, FundPerformanceMetrics>;
    common_start_date: string | null;
    last_updated: string | null;
    is_stale?: boolean;
    is_syncing?: boolean;
}

// Auth types
export interface AuthUser {
    id: number;
    email: string;
    download_folder: string | null;
    company_export_category: string | null;
    finance_export_category: string | null;
    is_active: boolean;
    created_at: string;
    last_login: string | null;
}

export interface AuthResponse {
    access_token: string;
    token_type: string;
    expires_in: number;
    user: AuthUser;
}

export interface RegisterRequest {
    email: string;
    password: string;
}

export interface LoginRequest {
    email: string;
    password: string;
}

export interface UserSettingsResponse {
    download_folder: string | null;
    company_export_category: string | null;
    finance_export_category: string | null;
}

export interface UpdateUserSettingsRequest {
    download_folder?: string | null;
    company_export_category?: string | null;
    finance_export_category?: string | null;
}

export type NewsSourceKind = 'rss' | 'crawl';
export type NewsDiscoveryMethod = 'homepage' | 'manual' | 'default_pack' | 'sitemap';
export type NewsValidationStatus = 'pending' | 'valid' | 'invalid';
export type NewsSortMode = 'latest' | 'relevance';
export type NewsScopeMode = 'all' | 'portfolio' | 'bookmarks';
export type NewsGroupMode = 'article' | 'story';
export type NewsEventType =
    | 'earnings'
    | 'dividend'
    | 'capital_raise'
    | 'insider_trading'
    | 'management_change'
    | 'regulatory'
    | 'mna'
    | 'analyst_view'
    | 'macro_policy'
    | 'other';
export type NewsImportanceLevel = 'low' | 'medium' | 'high';

export interface NewsSite {
    id: number;
    domain: string;
    homepage_url: string;
    display_name: string | null;
    is_public: boolean;
    created_at: string | null;
    updated_at: string | null;
}

export interface NewsSourceSummary {
    id: number;
    kind: NewsSourceKind;
    title: string | null;
    enabled: boolean;
    validation_status: NewsValidationStatus;
    last_validated_at: string | null;
    last_error: string | null;
    poll_interval_minutes: number;
    site_name: string | null;
    site_url: string | null;
    created_at: string | null;
    updated_at: string | null;
}

export interface NewsRssSource extends NewsSourceSummary {
    kind: 'rss';
    feed_url: string;
    discovery_method: NewsDiscoveryMethod;
}

export interface NewsCrawlSource extends NewsSourceSummary {
    kind: 'crawl';
    listing_url: string;
    article_link_selector: string;
    content_selector: string;
    excerpt_selector: string | null;
    pagination_selector: string | null;
}

export interface NewsSourceSubscription {
    id: number;
    user_id: number;
    source_kind: NewsSourceKind;
    source_id: number;
    enabled: boolean;
    source_title: string | null;
    created_at: string | null;
    updated_at: string | null;
}

export interface NewsFeedItem {
    id: number;
    title: string;
    excerpt: string | null;
    original_excerpt?: string | null;
    llm_summary: string | null;
    canonical_url: string;
    published_at: string | null;
    language: string | null;
    image_url: string | null;
    source_labels: string[];
    topics: string[];
    tickers: string[];
    sectors: string[];
    importance: string | null;
    sentiment: string | null;
    event_type: NewsEventType | null;
    event_labels: string[];
    matched_tickers: string[];
    why_relevant: string[];
    story_key: string | null;
    story_source_count: number;
    related_article_ids: number[];
    source_title: string | null;
    source_kind: NewsSourceKind | null;
    is_filtered_for_user: boolean;
}

export interface NewsRelatedArticle {
    id: number;
    title: string;
    published_at: string | null;
    canonical_url: string;
    source_title: string | null;
}

export interface NewsArticleDetail extends NewsFeedItem {
    content_text: string | null;
    source_urls: string[];
    related_articles: NewsRelatedArticle[];
}

export interface NewsArticleSummaryResponse {
    id: number;
    excerpt: string | null;
    llm_summary: string | null;
}

export interface NewsFeedResponse {
    items: NewsFeedItem[];
    count: number;
    next_cursor: string | null;
    is_personalized: boolean;
}

export interface NewsSourcesResponse {
    sites: NewsSite[];
    rss_sources: NewsRssSource[];
    crawl_sources: NewsCrawlSource[];
    subscriptions: NewsSourceSubscription[];
}

export interface NewsRssDiscoveryCandidate {
    feed_url: string;
    title: string | null;
    site_url: string | null;
    discovery_method: NewsDiscoveryMethod;
    kind: 'rss' | 'atom';
    validation_status: NewsValidationStatus;
    category_hint: string | null;
}

export interface NewsCrawlDiscoveryCandidate {
    listing_url: string;
    title: string | null;
    site_url: string | null;
    discovery_method: NewsDiscoveryMethod;
    category_hint: string | null;
}

export interface NewsRssDiscoveryResponse {
    homepage_url: string;
    site_title: string | null;
    candidates: NewsRssDiscoveryCandidate[];
    crawl_candidates: NewsCrawlDiscoveryCandidate[];
}

export interface NewsValidationResponse {
    valid: boolean;
    message: string;
    sample_title: string | null;
    sample_excerpt: string | null;
    candidate_count: number | null;
    suggestions: string[];
}

export interface NewsRssDiscoveryRequest {
    homepage_url: string;
}

export interface NewsRssSourceCreateRequest {
    feed_url: string;
    site_url?: string | null;
    homepage_url?: string | null;
    title?: string | null;
    enabled?: boolean;
    poll_interval_minutes?: number;
    discovery_method?: NewsDiscoveryMethod;
}

export interface NewsCrawlSourceCreateRequest {
    listing_url: string;
    article_link_selector: string;
    content_selector: string;
    excerpt_selector?: string | null;
    pagination_selector?: string | null;
    title?: string | null;
    site_url?: string | null;
    enabled?: boolean;
    poll_interval_minutes?: number;
}

export interface NewsSourceUpdateRequest {
    title?: string | null;
    enabled?: boolean;
    poll_interval_minutes?: number;
}

export interface NewsFeedQuery {
    source?: string;
    topic?: string;
    ticker?: string;
    from?: string;
    to?: string;
    sort?: NewsSortMode;
    scope?: NewsScopeMode;
    bookmark_group_id?: number;
    event_type?: NewsEventType;
    importance?: NewsImportanceLevel;
    group_by?: NewsGroupMode;
    cursor?: string;
    limit?: number;
}

export interface NewsUserPreferences {
    blocked_topics_text: string;
    blocked_labels: string[];
    updated_at: string | null;
}

export interface NewsUserPreferencesUpdateRequest {
    blocked_topics_text: string;
}

export interface NewsMonitoringOverviewResponse {
    total_sources?: number;
    enabled_sources?: number;
    valid_sources?: number;
    invalid_sources?: number;
    public_sources?: number;
    private_sources?: number;
    articles_total?: number;
    articles_last_24h?: number;
    active_runs?: number;
    queue_size?: number;
    last_run_at?: string | null;
    last_run_status?: string | null;
    last_run_error?: string | null;
    updated_at?: string | null;
}

export interface NewsMonitoringRun {
    id: number;
    source_type: NewsSourceKind;
    source_label: string | null;
    status: string;
    fetched_count: number;
    stored_count: number;
    filtered_count: number;
    error: string | null;
    started_at: string | null;
    finished_at: string | null;
}

export interface NewsMonitoringRunsResponse {
    runs: NewsMonitoringRun[];
    count: number;
}

export interface NewsMonitoringActionResponse {
    started: boolean;
    message: string;
    queued_count?: number;
    refreshed_count?: number;
    timestamp?: string | null;
}

export interface NewsAdminConfig {
    default_poll_interval_minutes: number;
}

// Portfolio types
export interface PortfolioPosition {
    id: number;
    ticker: string;
    quantity: number;
    average_cost: number | null;
    purchase_date: string | null;
    created_at: string | null;
    updated_at: string | null;
}

export interface PortfolioPositionsResponse {
    positions: PortfolioPosition[];
    count: number;
}

export interface PortfolioPositionCreateRequest {
    ticker: string;
    quantity: number;
    average_cost?: number | null;
    purchase_date?: string | null;
}

export interface PortfolioPositionUpdateRequest {
    quantity?: number;
    average_cost?: number | null;
    purchase_date?: string | null;
}

export interface PortfolioImportBroker {
    id: string;
    name: string;
    sheet: string | null;
    top_left: string;
    bottom_right: string;
    average_cost_multiplier?: number;
}

export interface PortfolioImportPosition {
    ticker: string;
    quantity?: number | null;
    average_cost?: number | null;
}

export interface PortfolioImportResponse {
    imported_positions: PortfolioImportPosition[];
    created_count: number;
    updated_count: number;
    skipped_count: number;
    positions: PortfolioPosition[];
}

export interface PortfolioFreshImportResponse {
    created_count: number;
    deleted_count: number;
    positions: PortfolioPosition[];
}

export interface PortfolioCsvExportResponse {
    blob: Blob;
    filename: string;
}

export interface TradingPosition {
    id: number;
    account_label: string;
    ticker: string;
    quantity: number;
    average_entry_cost: number;
    opened_date: string | null;
    target_price: number | null;
    stop_loss: number | null;
    notes: string | null;
    created_at: string | null;
    updated_at: string | null;
}

export interface TradingPositionsResponse {
    positions: TradingPosition[];
    count: number;
}

export interface TradingPositionCreateRequest {
    account_label: string;
    ticker: string;
    quantity: number;
    average_entry_cost: number;
    opened_date?: string | null;
    target_price?: number | null;
    stop_loss?: number | null;
    notes?: string | null;
}

export interface TradingPositionUpdateRequest {
    account_label?: string;
    ticker?: string;
    quantity?: number;
    average_entry_cost?: number;
    opened_date?: string | null;
    target_price?: number | null;
    stop_loss?: number | null;
    notes?: string | null;
}

export interface TradingImportPosition {
    ticker: string;
    quantity?: number | null;
    average_entry_cost?: number | null;
}

export interface TradingImportOutcome {
    ticker: string;
    quantity?: number | null;
    average_entry_cost?: number | null;
    status: 'created' | 'updated' | 'skipped';
    reason?: string | null;
}

export interface TradingImportResponse {
    imported_positions: TradingImportPosition[];
    import_outcomes: TradingImportOutcome[];
    created_count: number;
    updated_count: number;
    skipped_count: number;
    positions: TradingPosition[];
}

export interface StockQuotesResponse {
    stocks: Stock[];
    count: number;
}

// Sync Status Types
export interface SyncStatusItem {
    is_syncing: boolean;
    last_sync: string | null;
    error: string | null;
    started_at: string | null;
}

export interface HistoryJobStatus {
    is_running: boolean;
    total_symbols: number;
    processed_symbols: number;
    success_symbols: number;
    failed_symbols: number;
    failed_tickers: string[];
    current_symbol: string | null;
    last_run_at: string | null;
    started_at: string | null;
    error: string | null;
    progress: number;
}

export interface HistorySyncStatus {
    sync: HistoryJobStatus;
    audit: HistoryJobStatus;
    repair: HistoryJobStatus;
}

export interface SyncStatusResponse {
    fund_performance: SyncStatusItem;
    history_sync: HistorySyncStatus;
    finance_sync: HistoryJobStatus;
    company_sync: HistoryJobStatus;
    is_rate_limited: boolean;
    rate_limit_reset_at: string | null;
}

export interface HistorySyncActionResponse {
    started: boolean;
    message: string;
    processed_symbols: number;
    success_symbols: number;
    failed_symbols: number;
    state?: string | null;
    start_date?: string | null;
    end_date?: string | null;
}

export interface HistoryAuditSymbolResult {
    symbol: string;
    local_dates: number;
    upstream_dates: number;
    missing_dates: number;
    repaired_dates: number;
    missing_date_samples: string[];
    error?: string | null;
}

export interface HistoryAuditActionResponse extends HistorySyncActionResponse {
    audited_symbols: number;
    symbols_with_gaps: number;
    total_missing_dates: number;
    total_repaired_dates: number;
    results: HistoryAuditSymbolResult[];
}

export type ScheduledSyncType = 'history' | 'finance' | 'company';
export type ScheduledSyncAction = 'sync' | 'audit' | 'repair' | 'full' | 'quick';
export type ScheduledSyncIntervalUnit = 'minutes' | 'hours' | 'days';
export type ScheduledSyncRunStatus = 'queued' | 'running' | 'succeeded' | 'failed';

export interface ScheduledSyncJob {
    id: number;
    name: string;
    enabled: boolean;
    sync_type: ScheduledSyncType;
    sync_action: ScheduledSyncAction;
    index_symbol: string | null;
    symbols: string[];
    date_from: string | null;
    date_to: string | null;
    auto_repair: boolean;
    starts_at: string;
    interval_value: number;
    interval_unit: ScheduledSyncIntervalUnit;
    timezone: string;
    max_retries: number;
    next_run_at: string | null;
    last_run_at: string | null;
    created_at: string | null;
    updated_at: string | null;
}

export interface ScheduledSyncJobListResponse {
    jobs: ScheduledSyncJob[];
    count: number;
}

export interface ScheduledSyncJobRun {
    id: number;
    job_id: number;
    job_name: string;
    sync_type: ScheduledSyncType;
    sync_action: ScheduledSyncAction;
    attempt_number: number;
    status: ScheduledSyncRunStatus;
    scheduled_for: string;
    started_at: string | null;
    finished_at: string | null;
    error: string | null;
    summary: Record<string, unknown>;
    created_at: string | null;
    updated_at: string | null;
}

export interface ScheduledSyncJobRunListResponse {
    runs: ScheduledSyncJobRun[];
    count: number;
}

export interface ScheduledSyncJobCreateRequest {
    name: string;
    enabled?: boolean;
    sync_type: ScheduledSyncType;
    sync_action: ScheduledSyncAction;
    index_symbol?: string;
    symbols?: string[];
    date_from?: string;
    date_to?: string;
    auto_repair?: boolean;
    starts_at: string;
    interval_value: number;
    interval_unit: ScheduledSyncIntervalUnit;
    timezone?: string;
    max_retries?: number;
}

export interface ScheduledSyncJobUpdateRequest {
    name?: string;
    enabled?: boolean;
    sync_type?: ScheduledSyncType;
    sync_action?: ScheduledSyncAction;
    index_symbol?: string | null;
    symbols?: string[];
    date_from?: string | null;
    date_to?: string | null;
    auto_repair?: boolean;
    starts_at?: string;
    interval_value?: number;
    interval_unit?: ScheduledSyncIntervalUnit;
    timezone?: string;
    max_retries?: number;
}

// Stock API functions
export const stockApi = {
    /**
     * Fetch latest values for major market indices
     */
    async getIndexValues(): Promise<IndexValuesResponse> {
        const response = await apiClient.get<IndexValuesResponse>('/stocks/index-values');
        return response.data;
    },

    async register(payload: RegisterRequest): Promise<AuthResponse> {
        const response = await apiClient.post<AuthResponse>('/auth/register', payload);
        return response.data;
    },

    async login(payload: LoginRequest): Promise<AuthResponse> {
        const response = await apiClient.post<AuthResponse>('/auth/login', payload);
        return response.data;
    },

    async getUserSettings(): Promise<UserSettingsResponse> {
        const response = await apiClient.get<UserSettingsResponse>('/auth/settings');
        return response.data;
    },

    async updateUserSettings(payload: UpdateUserSettingsRequest): Promise<UserSettingsResponse> {
        const response = await apiClient.patch<UserSettingsResponse>('/auth/settings', payload);
        return response.data;
    },

    async getNewsFeed(options?: NewsFeedQuery): Promise<NewsFeedResponse> {
        const params = buildNewsQueryParams(options);
        const query = params.toString();
        const response = await apiClient.get<NewsFeedResponse>(`/news/feed${query ? `?${query}` : ''}`);
        return response.data;
    },

    async getNewsArticle(articleId: number | string): Promise<NewsArticleDetail> {
        const response = await apiClient.get<NewsArticleDetail>(`/news/articles/${articleId}`);
        return response.data;
    },

    async refreshNewsArticleContent(articleId: number | string): Promise<NewsArticleDetail> {
        const response = await apiClient.post<NewsArticleDetail>(`/news/articles/${articleId}/refresh-content`);
        return response.data;
    },

    async summarizeNewsArticle(articleId: number | string, options?: { forceRefresh?: boolean }): Promise<NewsArticleSummaryResponse> {
        const params = new URLSearchParams();
        if (options?.forceRefresh) {
            params.set('force_refresh', 'true');
        }
        const query = params.toString();
        const response = await apiClient.post<NewsArticleSummaryResponse>(`/news/articles/${articleId}/summary${query ? `?${query}` : ''}`);
        return response.data;
    },

    async discoverNewsRss(payload: NewsRssDiscoveryRequest): Promise<NewsRssDiscoveryResponse> {
        const response = await apiClient.post<NewsRssDiscoveryResponse>('/news/rss/discover', payload);
        return response.data;
    },

    async validateNewsRss(payload: NewsRssSourceCreateRequest): Promise<NewsValidationResponse> {
        const response = await apiClient.post<NewsValidationResponse>('/news/rss/validate', payload);
        return response.data;
    },

    async validateNewsCrawl(payload: NewsCrawlSourceCreateRequest): Promise<NewsValidationResponse> {
        const response = await apiClient.post<NewsValidationResponse>('/news/crawl/validate', payload);
        return response.data;
    },

    async getNewsSources(): Promise<NewsSourcesResponse> {
        const response = await apiClient.get<NewsSourcesResponse>('/news/sources');
        return response.data;
    },

    async createNewsRssSource(payload: NewsRssSourceCreateRequest): Promise<NewsRssSource> {
        const response = await apiClient.post<NewsRssSource>('/news/sources/rss', payload);
        return response.data;
    },

    async createNewsCrawlSource(payload: NewsCrawlSourceCreateRequest): Promise<NewsCrawlSource> {
        const response = await apiClient.post<NewsCrawlSource>('/news/sources/crawl', payload);
        return response.data;
    },

    async updateNewsSource(
        sourceType: NewsSourceKind,
        sourceId: number,
        payload: NewsSourceUpdateRequest,
    ): Promise<NewsSourceSummary> {
        const response = await apiClient.patch<NewsSourceSummary>(`/news/sources/${sourceType}/${sourceId}`, payload);
        return response.data;
    },

    async deleteNewsSource(sourceType: NewsSourceKind, sourceId: number): Promise<void> {
        await apiClient.delete(`/news/sources/${sourceType}/${sourceId}`);
    },

    async getNewsPreferences(): Promise<NewsUserPreferences> {
        const response = await apiClient.get<NewsUserPreferences>('/news/preferences');
        return response.data;
    },

    async updateNewsPreferences(payload: NewsUserPreferencesUpdateRequest): Promise<NewsUserPreferences> {
        const response = await apiClient.patch<NewsUserPreferences>('/news/preferences', payload);
        return response.data;
    },

    async getNewsMonitoringOverview(): Promise<NewsMonitoringOverviewResponse> {
        const response = await apiClient.get<NewsMonitoringOverviewResponse>('/news/admin/overview');
        return response.data;
    },

    async getNewsMonitoringSources(): Promise<NewsSourcesResponse> {
        const response = await apiClient.get<NewsSourcesResponse>('/news/admin/sources');
        return response.data;
    },

    async deleteNewsMonitoringSource(sourceType: NewsSourceKind, sourceId: number): Promise<void> {
        await apiClient.delete(`/news/admin/sources/${sourceType}/${sourceId}`);
    },

    async getNewsMonitoringRuns(limit: number = 12): Promise<NewsMonitoringRunsResponse> {
        const response = await apiClient.get<NewsMonitoringRunsResponse>(`/news/admin/runs?limit=${limit}`);
        return response.data;
    },

    async getNewsAdminConfig(): Promise<NewsAdminConfig> {
        const response = await apiClient.get<NewsAdminConfig>('/news/admin/config');
        return response.data;
    },

    async updateNewsAdminConfig(payload: NewsAdminConfig): Promise<NewsAdminConfig> {
        const response = await apiClient.patch<NewsAdminConfig>('/news/admin/config', payload);
        return response.data;
    },

    async triggerNewsIngestion(): Promise<NewsMonitoringActionResponse> {
        const response = await apiClient.post<NewsMonitoringActionResponse>('/news/admin/ingest');
        return response.data;
    },

    async refreshNewsMonitoring(): Promise<NewsMonitoringActionResponse> {
        const response = await apiClient.post<NewsMonitoringActionResponse>('/news/admin/refresh');
        return response.data;
    },

    async repairNewsRssTitles(): Promise<NewsMonitoringActionResponse> {
        const response = await apiClient.post<NewsMonitoringActionResponse>('/news/admin/repair-rss-titles');
        return response.data;
    },

    async applyNewsDefaultPollIntervalToExistingSources(): Promise<NewsMonitoringActionResponse> {
        const response = await apiClient.post<NewsMonitoringActionResponse>('/news/admin/apply-default-poll-interval');
        return response.data;
    },

    /**
     * Fetch all available indices
     */
    async getIndices(): Promise<IndexListResponse> {
        const response = await apiClient.get<IndexListResponse>('/stocks/indices');
        return response.data;
    },

    /**
     * Fetch stocks for a given index
     * @param indexSymbol - Index symbol (e.g. 'VN30') to use with the generic endpoint
     */
    async getIndexStocks(indexSymbol: string, options?: StockListRangeParams): Promise<IndexStocksResponse> {
        const params = buildStockRangeParams(options);
        const query = params.toString();
        const response = await apiClient.get<IndexStocksResponse>(
            `/stocks/index/${indexSymbol}${query ? `?${query}` : ''}`
        );
        return response.data;
    },

    /**
     * Fetch all available industries
     */
    async getIndustries(): Promise<IndustryListResponse> {
        const response = await apiClient.get<IndustryListResponse>('/stocks/industries');
        return response.data;
    },

    /**
     * Fetch stocks for a given industry
     */
    async getIndustryStocks(industryName: string, options?: StockListRangeParams): Promise<IndustryStocksResponse> {
        const params = buildStockRangeParams(options);
        const query = params.toString();
        const response = await apiClient.get<IndustryStocksResponse>(
            `/stocks/industry/${encodeURIComponent(industryName)}${query ? `?${query}` : ''}`
        );
        return response.data;
    },

    /**
     * Fetch income statement for a specific stock
     */
    async getIncomeStatement(symbol: string): Promise<FinancialDataResponse> {
        const response = await apiClient.get<FinancialDataResponse>(`/stocks/finance/${symbol}/income-statement`);
        return response.data;
    },

    /**
     * Fetch balance sheet for a specific stock
     */
    async getBalanceSheet(symbol: string): Promise<FinancialDataResponse> {
        const response = await apiClient.get<FinancialDataResponse>(`/stocks/finance/${symbol}/balance-sheet`);
        return response.data;
    },

    /**
     * Fetch cash flow for a specific stock
     */
    async getCashFlow(symbol: string): Promise<FinancialDataResponse> {
        const response = await apiClient.get<FinancialDataResponse>(`/stocks/finance/${symbol}/cash-flow`);
        return response.data;
    },

    /**
     * Fetch financial ratios for a specific stock
     */
    async getFinancialRatios(symbol: string): Promise<FinancialDataResponse> {
        const response = await apiClient.get<FinancialDataResponse>(`/stocks/finance/${symbol}/ratios`);
        return response.data;
    },

    /**
     * Fetch company overview for a specific stock
     */
    async getCompanyOverview(symbol: string): Promise<FinancialDataResponse> {
        const response = await apiClient.get<FinancialDataResponse>(`/stocks/company/${symbol}/overview`);
        return response.data;
    },

    /**
     * Fetch major shareholders for a specific stock
     */
    async getShareholders(symbol: string): Promise<FinancialDataResponse> {
        const response = await apiClient.get<FinancialDataResponse>(`/stocks/company/${symbol}/shareholders`);
        return response.data;
    },

    /**
     * Fetch company officers for a specific stock
     */
    async getOfficers(symbol: string): Promise<FinancialDataResponse> {
        const response = await apiClient.get<FinancialDataResponse>(`/stocks/company/${symbol}/officers`);
        return response.data;
    },

    /**
     * Fetch subsidiaries for a specific stock
     */
    async getSubsidiaries(symbol: string): Promise<FinancialDataResponse> {
        const response = await apiClient.get<FinancialDataResponse>(`/stocks/company/${symbol}/subsidiaries`);
        return response.data;
    },

    /**
     * Fetch volume history for a specific stock
     * @param symbol Stock ticker symbol
     * @param days Number of days to fetch (default: 30)
     */
    async getVolumeHistory(
        symbol: string,
        days: number = 30,
        options?: HistoryRequestOptions,
    ): Promise<VolumeHistoryResponse> {
        const params = new URLSearchParams({ days: String(days) });
        if (options?.autoSync !== undefined) {
            params.set('auto_sync', String(options.autoSync));
        }
        const response = await apiClient.get<VolumeHistoryResponse>(
            `/stocks/history/${symbol}/volume?${params.toString()}`
        );
        return response.data;
    },

    /**
     * Fetch price history for a specific stock
     * @param symbol Stock ticker symbol
     * @param days Number of days to fetch (default: 30)
     */
    async getPriceHistory(
        symbol: string,
        days: number = 30,
        options?: HistoryRequestOptions,
    ): Promise<PriceHistoryResponse> {
        const params = new URLSearchParams({ days: String(days) });
        if (options?.autoSync !== undefined) {
            params.set('auto_sync', String(options.autoSync));
        }
        const response = await apiClient.get<PriceHistoryResponse>(
            `/stocks/history/${symbol}/price?${params.toString()}`
        );
        return response.data;
    },

    /**
     * Fetch full OHLCV history for a specific stock
     * @param symbol Stock ticker symbol
     */
    async getPriceHistoryOhlcv(symbol: string): Promise<OhlcvHistoryResponse> {
        const response = await apiClient.get<OhlcvHistoryResponse>(`/stocks/history/${symbol}/ohlcv`);
        return response.data;
    },

    /**
     * Fetch weekly prices for multiple stocks (for growth and risk/return charts)
     * @param symbols List of stock ticker symbols
     * @param startDate Inclusive start date (YYYY-MM-DD)
     * @param endDate Inclusive end date (YYYY-MM-DD)
     * @param includeBenchmarks Whether to include VNINDEX and VN30 benchmarks
     */
    async getStocksWeeklyPrices(
        symbols: string[],
        startDate: string,
        endDate: string,
        includeBenchmarks: boolean = true
    ): Promise<StocksWeeklyPricesResponse> {
        const response = await apiClient.post<StocksWeeklyPricesResponse>('/stocks/weekly-prices', {
            symbols,
            start_date: startDate,
            end_date: endDate,
            include_benchmarks: includeBenchmarks
        });
        return response.data;
    },

    /**
     * Fetch daily trading value series for multiple stocks.
     * @param symbols List of stock ticker symbols
     * @param startDate Inclusive start date (YYYY-MM-DD)
     * @param endDate Inclusive end date (YYYY-MM-DD)
     */
    async getStocksVolumeSeries(
        symbols: string[],
        startDate: string,
        endDate: string
    ): Promise<StocksVolumeSeriesResponse> {
        const response = await apiClient.post<StocksVolumeSeriesResponse>('/stocks/volume-series', {
            symbols,
            start_date: startDate,
            end_date: endDate,
        });
        return response.data;
    },

    /**
     * Fetch all available funds
     * @param fundType Optional filter by fund type (e.g., "STOCK", "BOND", "BALANCED")
     */
    async getFunds(fundType: string = ''): Promise<FundDataResponse> {
        const url = fundType ? `/funds/listing?fund_type=${fundType}` : '/funds/listing';
        const response = await apiClient.get<FundDataResponse>(url);
        return response.data;
    },

    /**
     * Fetch NAV history for a specific fund
     * @param symbol Fund symbol
     */
    async getFundNavReport(symbol: string): Promise<FundDataResponse> {
        const response = await apiClient.get<FundDataResponse>(`/funds/${symbol}/nav-report`);
        return response.data;
    },

    /**
     * Fetch top holdings for a specific fund
     * @param symbol Fund symbol
     */
    async getFundTopHolding(symbol: string): Promise<FundDataResponse> {
        const response = await apiClient.get<FundDataResponse>(`/funds/${symbol}/top-holding`);
        return response.data;
    },

    /**
     * Fetch industry allocation for a specific fund
     * @param symbol Fund symbol
     */
    async getFundIndustryHolding(symbol: string): Promise<FundDataResponse> {
        const response = await apiClient.get<FundDataResponse>(`/funds/${symbol}/industry-holding`);
        return response.data;
    },

    /**
     * Fetch asset allocation for a specific fund
     * @param symbol Fund symbol
     */
    async getFundAssetHolding(symbol: string): Promise<FundDataResponse> {
        const response = await apiClient.get<FundDataResponse>(`/funds/${symbol}/asset-holding`);
        return response.data;
    },

    /**
     * Fetch aggregated fund performance data for comparison charts
     * Includes normalized NAV, periodic returns, and risk metrics
     */
    async getFundPerformance(): Promise<FundPerformanceData> {
        const response = await apiClient.get<FundPerformanceData>('/funds/performance');
        return response.data;
    },

    /**
     * Fetch current background sync status
     */
    async getSyncStatus(): Promise<SyncStatusResponse> {
        const response = await apiClient.get<SyncStatusResponse>('/sync/status');
        return response.data;
    },

    async runHistorySync(
        forceRestart: boolean = false,
        symbols?: string[],
        indexSymbol?: string
    ): Promise<HistorySyncActionResponse> {
        const response = await apiClient.post<HistorySyncActionResponse>('/sync/history/run', {
            force_restart: forceRestart,
            symbols: symbols && symbols.length > 0 ? symbols : undefined,
            index_symbol: indexSymbol || undefined,
        });
        return response.data;
    },

    async runHistoryAudit(
        startDate: string,
        endDate: string,
        symbols?: string[],
        indexSymbol?: string,
        autoRepair: boolean = false
    ): Promise<HistoryAuditActionResponse> {
        const response = await apiClient.post<HistoryAuditActionResponse>('/sync/history/audit/run', {
            symbols: symbols && symbols.length > 0 ? symbols : undefined,
            index_symbol: indexSymbol || undefined,
            start_date: startDate,
            end_date: endDate,
            auto_repair: autoRepair,
        });
        return response.data;
    },

    async runHistoryRepairSync(
        symbols: string[] | undefined,
        startDate: string,
        endDate: string,
        indexSymbol?: string
    ): Promise<HistorySyncActionResponse> {
        const response = await apiClient.post<HistorySyncActionResponse>('/sync/history/repair/run', {
            symbols: symbols && symbols.length > 0 ? symbols : undefined,
            index_symbol: indexSymbol || undefined,
            start_date: startDate,
            end_date: endDate,
        });
        return response.data;
    },

    async runFinanceSync(
        forceRestart: boolean = false,
        symbols?: string[],
        indexSymbol?: string,
        quickSync: boolean = false
    ): Promise<HistorySyncActionResponse> {
        const response = await apiClient.post<HistorySyncActionResponse>('/sync/finance/run', {
            force_restart: forceRestart,
            symbols: symbols && symbols.length > 0 ? symbols : undefined,
            index_symbol: indexSymbol || undefined,
            quick_sync: quickSync,
        });
        return response.data;
    },

    async runCompanySync(
        forceRestart: boolean = false,
        symbols?: string[],
        indexSymbol?: string,
        quickSync: boolean = false
    ): Promise<HistorySyncActionResponse> {
        const response = await apiClient.post<HistorySyncActionResponse>('/sync/company/run', {
            force_restart: forceRestart,
            symbols: symbols && symbols.length > 0 ? symbols : undefined,
            index_symbol: indexSymbol || undefined,
            quick_sync: quickSync,
        });
        return response.data;
    },

    async getScheduledSyncJobs(): Promise<ScheduledSyncJobListResponse> {
        const response = await apiClient.get<ScheduledSyncJobListResponse>('/sync/scheduler/jobs');
        return response.data;
    },

    async createScheduledSyncJob(payload: ScheduledSyncJobCreateRequest): Promise<ScheduledSyncJob> {
        const response = await apiClient.post<ScheduledSyncJob>('/sync/scheduler/jobs', payload);
        return response.data;
    },

    async updateScheduledSyncJob(jobId: number, payload: ScheduledSyncJobUpdateRequest): Promise<ScheduledSyncJob> {
        const response = await apiClient.patch<ScheduledSyncJob>(`/sync/scheduler/jobs/${jobId}`, payload);
        return response.data;
    },

    async deleteScheduledSyncJob(jobId: number): Promise<void> {
        await apiClient.delete(`/sync/scheduler/jobs/${jobId}`);
    },

    async getScheduledSyncRuns(limit: number = 20): Promise<ScheduledSyncJobRunListResponse> {
        const response = await apiClient.get<ScheduledSyncJobRunListResponse>(`/sync/scheduler/runs?limit=${limit}`);
        return response.data;
    },

    // Portfolio positions
    async getPortfolioPositions(): Promise<PortfolioPositionsResponse> {
        const response = await apiClient.get<PortfolioPositionsResponse>('/portfolio/positions');
        return response.data;
    },

    async createPortfolioPosition(payload: PortfolioPositionCreateRequest): Promise<PortfolioPosition> {
        const response = await apiClient.post<PortfolioPosition>('/portfolio/positions', payload);
        return response.data;
    },

    async updatePortfolioPosition(positionId: number, payload: PortfolioPositionUpdateRequest): Promise<PortfolioPosition> {
        const response = await apiClient.patch<PortfolioPosition>(`/portfolio/positions/${positionId}`, payload);
        return response.data;
    },

    async deletePortfolioPosition(positionId: number): Promise<void> {
        await apiClient.delete(`/portfolio/positions/${positionId}`);
    },

    async getPortfolioImportBrokers(): Promise<PortfolioImportBroker[]> {
        const response = await apiClient.get<PortfolioImportBroker[]>('/portfolio/import/brokers');
        return response.data;
    },

    async importPortfolioPositions(formData: FormData): Promise<PortfolioImportResponse> {
        const response = await apiClient.post<PortfolioImportResponse>('/portfolio/import', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    },

    async exportPortfolioCsv(): Promise<PortfolioCsvExportResponse> {
        const response = await apiClient.get('/portfolio/export/csv', {
            responseType: 'blob',
        });
        const contentDisposition = response.headers['content-disposition'] as string | undefined;
        let filename = buildPortfolioCsvFallbackFilename();
        if (contentDisposition) {
            const utf8FilenameMatch = contentDisposition.match(/filename\*=UTF-8''([^;]+)/i);
            if (utf8FilenameMatch?.[1]) {
                filename = decodeURIComponent(utf8FilenameMatch[1]);
            } else {
                const filenameMatch = contentDisposition.match(/filename="?([^";]+)"?/i);
                if (filenameMatch?.[1]) {
                    filename = filenameMatch[1];
                }
            }
        }
        return {
            blob: response.data as Blob,
            filename,
        };
    },

    async freshImportPortfolioCsv(formData: FormData): Promise<PortfolioFreshImportResponse> {
        const response = await apiClient.post<PortfolioFreshImportResponse>('/portfolio/import/fresh', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    },

    async getTradingPositions(): Promise<TradingPositionsResponse> {
        const response = await apiClient.get<TradingPositionsResponse>('/trading/positions');
        return response.data;
    },

    async createTradingPosition(payload: TradingPositionCreateRequest): Promise<TradingPosition> {
        const response = await apiClient.post<TradingPosition>('/trading/positions', payload);
        return response.data;
    },

    async updateTradingPosition(positionId: number, payload: TradingPositionUpdateRequest): Promise<TradingPosition> {
        const response = await apiClient.patch<TradingPosition>(`/trading/positions/${positionId}`, payload);
        return response.data;
    },

    async deleteTradingPosition(positionId: number): Promise<void> {
        await apiClient.delete(`/trading/positions/${positionId}`);
    },

    async importTradingPositions(formData: FormData): Promise<TradingImportResponse> {
        const response = await apiClient.post<TradingImportResponse>('/trading/import', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    },

    async getStockQuotes(symbols: string[]): Promise<StockQuotesResponse> {
        const response = await apiClient.post<StockQuotesResponse>('/stocks/quotes', { symbols });
        return response.data;
    },

    // Bookmark groups
    async getBookmarkGroups(): Promise<BookmarkGroupsResponse> {
        const response = await apiClient.get<BookmarkGroupsResponse>('/bookmarks/groups');
        return response.data;
    },

    async createBookmarkGroup(payload: BookmarkGroupCreateRequest): Promise<BookmarkGroup> {
        const response = await apiClient.post<BookmarkGroup>('/bookmarks/groups', payload);
        return response.data;
    },

    async updateBookmarkGroup(groupId: number, payload: BookmarkGroupUpdateRequest): Promise<BookmarkGroup> {
        const response = await apiClient.patch<BookmarkGroup>(`/bookmarks/groups/${groupId}`, payload);
        return response.data;
    },

    async deleteBookmarkGroup(groupId: number): Promise<void> {
        await apiClient.delete(`/bookmarks/groups/${groupId}`);
    },

    async addBookmarkStock(groupId: number, ticker: string): Promise<BookmarkGroup> {
        const response = await apiClient.post<BookmarkGroup>(`/bookmarks/groups/${groupId}/stocks`, {
            ticker
        });
        return response.data;
    },

    async removeBookmarkStock(groupId: number, ticker: string): Promise<BookmarkGroup> {
        const response = await apiClient.delete<BookmarkGroup>(`/bookmarks/groups/${groupId}/stocks/${encodeURIComponent(ticker)}`);
        return response.data;
    },

    async getBookmarkGroupStocks(groupId: number, options?: StockListRangeParams): Promise<BookmarkGroupStocksResponse> {
        const params = buildStockRangeParams(options);
        const query = params.toString();
        const response = await apiClient.get<BookmarkGroupStocksResponse>(
            `/bookmarks/groups/${groupId}/stocks${query ? `?${query}` : ''}`
        );
        return response.data;
    },

    async getAppInfo(): Promise<AppInfoResponse> {
        const response = await apiClient.get<AppInfoResponse>('/info');
        return response.data;
    },
};

export default apiClient;
