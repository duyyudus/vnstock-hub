/**
 * Index configuration type definitions.
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
    /** API endpoint path or ID for fetching index stocks */
    apiEndpoint: string;
}