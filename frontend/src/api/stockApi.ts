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
        const expiresAt = Date.now() + expiresInSeconds * 1000;
        window.localStorage.setItem(AUTH_TOKEN_KEY, token);
        window.localStorage.setItem(AUTH_USER_KEY, JSON.stringify(user));
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
            return JSON.parse(raw) as AuthUser;
        } catch {
            return null;
        }
    },
    setUser(user: AuthUser) {
        if (typeof window === 'undefined') {
            return;
        }
        window.localStorage.setItem(AUTH_USER_KEY, JSON.stringify(user));
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
    price_change_24h: number | null;
    price_change_1w: number | null;
    price_change_1m: number | null;
    price_change_6m?: number | null;
    price_change_1y: number | null;
    price_change_2y?: number | null;
    price_change_3y?: number | null;
    industry: string;  // ICB Level 2 industry classification
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
    data: any[];
    count: number;
}

export interface VolumeDataPoint {
    date: string;
    volume: number;
    value: number | null;
}

export interface VolumeHistoryResponse {
    symbol: string;
    company_name: string;
    data: VolumeDataPoint[];
    count: number;
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

// Fund data types
export interface FundDataResponse {
    symbol?: string;
    data: any[];
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

export interface PriceJobStatus {
    is_running: boolean;
    total_symbols: number;
    processed_symbols: number;
    success_symbols: number;
    failed_symbols: number;
    current_symbol: string | null;
    last_run_at: string | null;
    started_at: string | null;
    error: string | null;
    progress: number;
}

export interface PriceSyncStatus {
    sync: PriceJobStatus;
    audit: PriceJobStatus;
    repair: PriceJobStatus;
}

export interface SyncStatusResponse {
    fund_performance: SyncStatusItem;
    price_sync: PriceSyncStatus;
    is_rate_limited: boolean;
    rate_limit_reset_at: string | null;
}

export interface PriceSyncActionResponse {
    started: boolean;
    message: string;
    processed_symbols: number;
    success_symbols: number;
    failed_symbols: number;
    state?: string | null;
    start_date?: string | null;
    end_date?: string | null;
}

export interface PriceAuditSymbolResult {
    symbol: string;
    local_dates: number;
    upstream_dates: number;
    missing_dates: number;
    repaired_dates: number;
    missing_date_samples: string[];
    error?: string | null;
}

export interface PriceAuditActionResponse extends PriceSyncActionResponse {
    audited_symbols: number;
    symbols_with_gaps: number;
    total_missing_dates: number;
    total_repaired_dates: number;
    results: PriceAuditSymbolResult[];
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
    async getIndexStocks(indexSymbol: string): Promise<IndexStocksResponse> {
        const response = await apiClient.get<IndexStocksResponse>(`/stocks/index/${indexSymbol}`);
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
    async getIndustryStocks(industryName: string): Promise<IndustryStocksResponse> {
        const response = await apiClient.get<IndustryStocksResponse>(`/stocks/industry/${encodeURIComponent(industryName)}`);
        return response.data;
    },

    /**
     * Fetch income statement for a specific stock
     */
    async getIncomeStatement(symbol: string, period: string = 'quarter'): Promise<FinancialDataResponse> {
        const response = await apiClient.get<FinancialDataResponse>(`/stocks/finance/${symbol}/income-statement?period=${period}`);
        return response.data;
    },

    /**
     * Fetch balance sheet for a specific stock
     */
    async getBalanceSheet(symbol: string, period: string = 'quarter'): Promise<FinancialDataResponse> {
        const response = await apiClient.get<FinancialDataResponse>(`/stocks/finance/${symbol}/balance-sheet?period=${period}`);
        return response.data;
    },

    /**
     * Fetch cash flow for a specific stock
     */
    async getCashFlow(symbol: string, period: string = 'quarter'): Promise<FinancialDataResponse> {
        const response = await apiClient.get<FinancialDataResponse>(`/stocks/finance/${symbol}/cash-flow?period=${period}`);
        return response.data;
    },

    /**
     * Fetch financial ratios for a specific stock
     */
    async getFinancialRatios(symbol: string, period: string = 'quarter'): Promise<FinancialDataResponse> {
        const response = await apiClient.get<FinancialDataResponse>(`/stocks/finance/${symbol}/ratios?period=${period}`);
        return response.data;
    },

    /**
     * Fetch company overview for a specific stock
     */
    async getCompanyOverview(symbol: string, source: string = 'vci'): Promise<FinancialDataResponse> {
        const response = await apiClient.get<FinancialDataResponse>(`/stocks/company/${symbol}/overview?source=${encodeURIComponent(source)}`);
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
    async getVolumeHistory(symbol: string, days: number = 30): Promise<VolumeHistoryResponse> {
        const response = await apiClient.get<VolumeHistoryResponse>(`/stocks/history/${symbol}/volume?days=${days}`);
        return response.data;
    },

    /**
     * Fetch price history for a specific stock
     * @param symbol Stock ticker symbol
     * @param days Number of days to fetch (default: 30)
     */
    async getPriceHistory(symbol: string, days: number = 30): Promise<PriceHistoryResponse> {
        const response = await apiClient.get<PriceHistoryResponse>(`/stocks/history/${symbol}/price?days=${days}`);
        return response.data;
    },

    /**
     * Fetch weekly prices for multiple stocks (for growth chart)
     * @param symbols List of stock ticker symbols
     * @param startYear Starting year for the data
     * @param includeBenchmarks Whether to include VNINDEX and VN30 benchmarks
     */
    async getStocksWeeklyPrices(
        symbols: string[],
        startYear: number,
        includeBenchmarks: boolean = true
    ): Promise<StocksWeeklyPricesResponse> {
        const response = await apiClient.post<StocksWeeklyPricesResponse>('/stocks/weekly-prices', {
            symbols,
            start_year: startYear,
            include_benchmarks: includeBenchmarks
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

    async runPriceSync(
        forceRestart: boolean = false,
        symbols?: string[],
        indexSymbol?: string
    ): Promise<PriceSyncActionResponse> {
        const response = await apiClient.post<PriceSyncActionResponse>('/sync/prices/run', {
            force_restart: forceRestart,
            symbols: symbols && symbols.length > 0 ? symbols : undefined,
            index_symbol: indexSymbol || undefined,
        });
        return response.data;
    },

    async runPriceAudit(
        startDate: string,
        endDate: string,
        symbols?: string[],
        indexSymbol?: string,
        autoRepair: boolean = false
    ): Promise<PriceAuditActionResponse> {
        const response = await apiClient.post<PriceAuditActionResponse>('/sync/prices/audit/run', {
            symbols: symbols && symbols.length > 0 ? symbols : undefined,
            index_symbol: indexSymbol || undefined,
            start_date: startDate,
            end_date: endDate,
            auto_repair: autoRepair,
        });
        return response.data;
    },

    async runPriceRepairSync(
        symbols: string[],
        startDate: string,
        endDate: string
    ): Promise<PriceSyncActionResponse> {
        const response = await apiClient.post<PriceSyncActionResponse>('/sync/prices/repair/run', {
            symbols,
            start_date: startDate,
            end_date: endDate,
        });
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
                const filenameMatch = contentDisposition.match(/filename=\"?([^\";]+)\"?/i);
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

    async getBookmarkGroupStocks(groupId: number): Promise<BookmarkGroupStocksResponse> {
        const response = await apiClient.get<BookmarkGroupStocksResponse>(`/bookmarks/groups/${groupId}/stocks`);
        return response.data;
    },
};

export default apiClient;
