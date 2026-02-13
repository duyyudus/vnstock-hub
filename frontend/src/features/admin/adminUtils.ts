export const getErrorMessage = (error: unknown) => {
    if (typeof error === 'object' && error && 'response' in error) {
        const response = (error as { response?: { data?: { detail?: string } } }).response;
        if (response?.data?.detail) {
            return response.data.detail;
        }
    }
    if (error instanceof Error) {
        return error.message;
    }
    return 'Request failed.';
};

export const formatDateTime = (value: string | null | undefined) => {
    if (!value) {
        return '-';
    }
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) {
        return value;
    }
    return parsed.toLocaleString();
};

export const parseSymbolsInput = (value: string): string[] => {
    return value
        .split(/[\s,]+/)
        .map((symbol) => symbol.trim().toUpperCase())
        .filter(Boolean);
};
