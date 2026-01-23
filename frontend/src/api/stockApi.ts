import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// Create axios instance with base configuration
const apiClient = axios.create({
    baseURL: `${API_BASE_URL}/api/v1`,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Stock data types
export interface Stock {
    ticker: string;
    price: number;
    market_cap: number;
    company_name: string;
    charter_capital: number;
    pe_ratio: number | null;
    accumulated_value: number | null;  // In billion VND
    price_change_24h: number | null;
    price_change_1w: number | null;
    price_change_1m: number | null;
    price_change_1y: number | null;
}

export interface VN100Response {
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

// Fund data types
export interface FundDataResponse {
    symbol?: string;
    data: any[];
    count: number;
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

    /**
     * Fetch all available indices
     */
    async getIndices(): Promise<IndexListResponse> {
        const response = await apiClient.get<IndexListResponse>('/stocks/indices');
        return response.data;
    },

    /**
     * Fetch VN-100 stocks data
     */
    async getVN100Stocks(): Promise<VN100Response> {
        const response = await apiClient.get<VN100Response>('/stocks/vn100');
        return response.data;
    },

    /**
     * Fetch stocks for a given index by endpoint or generic endpoint
     * @param indexSymbol - Index symbol (e.g. 'VN30') to use with generic endpoint, OR full endpoint path
     */
    async getIndexStocks(indexSymbol: string): Promise<VN100Response> {
        // If it starts with /, treat as direct endpoint (backward compatibility)
        if (indexSymbol.startsWith('/')) {
            const response = await apiClient.get<VN100Response>(indexSymbol);
            return response.data;
        }
        // Otherwise use the generic endpoint
        const response = await apiClient.get<VN100Response>(`/stocks/index/${indexSymbol}`);
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
    async getVolumeHistory(symbol: string, days: number = 30): Promise<VolumeHistoryResponse> {
        const response = await apiClient.get<VolumeHistoryResponse>(`/stocks/history/${symbol}/volume?days=${days}`);
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
};

export default apiClient;
