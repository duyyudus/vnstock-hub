import axios from 'axios';

const AUTH_TOKEN_KEY = 'vnstock_auth_token';
const AUTH_USER_KEY = 'vnstock_auth_user';

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

export const authStorage = {
    getToken() {
        return getStoredToken();
    },
    setToken(token: string) {
        if (typeof window === 'undefined') {
            return;
        }
        window.localStorage.setItem(AUTH_TOKEN_KEY, token);
    },
    clearToken() {
        if (typeof window === 'undefined') {
            return;
        }
        window.localStorage.removeItem(AUTH_TOKEN_KEY);
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
    },
    clearUser() {
        if (typeof window === 'undefined') {
            return;
        }
        window.localStorage.removeItem(AUTH_USER_KEY);
    },
    clearAll() {
        if (typeof window === 'undefined') {
            return;
        }
        window.localStorage.removeItem(AUTH_TOKEN_KEY);
        window.localStorage.removeItem(AUTH_USER_KEY);
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
    price_change_1y: number | null;
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

// Sync Status Types
export interface SyncStatusItem {
    is_syncing: boolean;
    last_sync: string | null;
    error: string | null;
    started_at: string | null;
}

export interface SyncStatusResponse {
    fund_performance: SyncStatusItem;
    is_rate_limited: boolean;
    rate_limit_reset_at: string | null;
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
};

export default apiClient;
