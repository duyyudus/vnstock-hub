export type CsvRow = Record<string, unknown>;

export const DEFAULT_COMPANY_EXPORT_CATEGORY = 'company';
export const DEFAULT_FINANCE_EXPORT_CATEGORY = 'finance';

const CSV_ESCAPE_PATTERN = /[",\r\n]/;

const escapeCsvValue = (value: unknown) => {
    if (value === null || value === undefined) {
        return '';
    }

    const serialized = String(value);
    if (!CSV_ESCAPE_PATTERN.test(serialized)) {
        return serialized;
    }

    return `"${serialized.replace(/"/g, '""')}"`;
};

export const rowsToCsv = (
    rows: CsvRow[],
    transformValue?: (value: unknown, key: string, row: CsvRow) => unknown,
) => {
    if (rows.length === 0) {
        return '';
    }

    const headers: string[] = [];
    const seenHeaders = new Set<string>();

    rows.forEach((row) => {
        Object.keys(row).forEach((key) => {
            if (!seenHeaders.has(key)) {
                seenHeaders.add(key);
                headers.push(key);
            }
        });
    });

    if (headers.length === 0) {
        return '';
    }

    const lines = [headers.map((header) => escapeCsvValue(header)).join(',')];

    rows.forEach((row) => {
        lines.push(
            headers
                .map((header) => {
                    const rawValue = row[header];
                    const transformedValue = transformValue ? transformValue(rawValue, header, row) : rawValue;
                    return escapeCsvValue(transformedValue);
                })
                .join(','),
        );
    });

    return lines.join('\r\n');
};

export const transformFinanceCsvValue = (value: unknown, key: string) => {
    if (value === null || value === undefined) {
        return '';
    }
    if (typeof value !== 'number') {
        return value;
    }

    const lowerKey = key.toLowerCase();
    const isNonCurrencyMetric = lowerKey.includes('percent')
        || lowerKey.includes('quantity')
        || lowerKey.includes('volume')
        || lowerKey.includes('share');

    if (isNonCurrencyMetric) {
        return value;
    }

    if (Math.abs(value) > 1e6) {
        return Math.round(value / 1e6) / 1000;
    }

    return value;
};

export const sanitizePathSegment = (value: string | null | undefined, fallback: string) => {
    const normalized = (value ?? '')
        .trim()
        .replace(/[\\/]+/g, '_')
        .replace(/[^A-Za-z0-9._-]+/g, '_')
        .replace(/^[._-]+|[._-]+$/g, '');

    if (normalized) {
        return normalized;
    }

    const normalizedFallback = fallback
        .trim()
        .replace(/[\\/]+/g, '_')
        .replace(/[^A-Za-z0-9._-]+/g, '_')
        .replace(/^[._-]+|[._-]+$/g, '');

    return normalizedFallback || 'default';
};

export const resolveTickerFolder = (ticker: string) => {
    return sanitizePathSegment(ticker.toUpperCase(), 'TICKER');
};

export const resolveCompanyExportCategory = (value: string | null | undefined) => {
    return sanitizePathSegment(value, DEFAULT_COMPANY_EXPORT_CATEGORY);
};

export const resolveFinanceExportCategory = (value: string | null | undefined) => {
    return sanitizePathSegment(value, DEFAULT_FINANCE_EXPORT_CATEGORY);
};
