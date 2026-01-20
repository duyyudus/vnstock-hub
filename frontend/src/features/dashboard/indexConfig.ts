/**
 * Index configuration for Vietnam stock market indices.
 * This file centralizes all index definitions to make adding new indices easy.
 */

export interface IndexConfig {
    /** Unique identifier (e.g., 'vn100', 'vn30') */
    id: string;
    /** Display name for dropdown (e.g., 'VN-100', 'VN-30') */
    label: string;
    /** Full title for the header section */
    title: string;
    /** Description text shown below the title */
    description: string;
    /** API endpoint path for fetching index stocks */
    apiEndpoint: string;
}

/**
 * Available indices that can be displayed.
 * Add new indices here to make them available in the dropdown.
 */
export const AVAILABLE_INDICES: IndexConfig[] = [
    {
        id: 'vn100',
        label: 'VN-100',
        title: 'VN-100 Stocks',
        description: 'Top 100 stocks by market capitalization on HOSE',
        apiEndpoint: '/stocks/vn100',
    },
    // Future indices can be added here:
    // {
    //   id: 'vn30',
    //   label: 'VN-30',
    //   title: 'VN-30 Stocks',
    //   description: 'Top 30 stocks by market capitalization and liquidity on HOSE',
    //   apiEndpoint: '/stocks/vn30',
    // },
    // {
    //   id: 'hnx30',
    //   label: 'HNX-30',
    //   title: 'HNX-30 Stocks',
    //   description: 'Top 30 stocks on Hanoi Stock Exchange',
    //   apiEndpoint: '/stocks/hnx30',
    // },
];

/** Default index to show when the page loads */
export const DEFAULT_INDEX_ID = 'vn100';

/**
 * Get an index configuration by its ID.
 * Falls back to the first available index if not found.
 */
export const getIndexById = (id: string): IndexConfig => {
    return AVAILABLE_INDICES.find((index) => index.id === id) || AVAILABLE_INDICES[0];
};
