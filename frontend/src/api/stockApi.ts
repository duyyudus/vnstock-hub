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

// Stock API functions
export const stockApi = {
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
};

export default apiClient;
