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

// Stock API functions
export const stockApi = {
    /**
     * Fetch VN-100 stocks data
     */
    async getVN100Stocks(): Promise<VN100Response> {
        const response = await apiClient.get<VN100Response>('/stocks/vn100');
        return response.data;
    },
};

export default apiClient;
