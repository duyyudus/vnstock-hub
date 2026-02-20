import { stockApi } from '../../../api/stockApi';
import type { AuthUser } from '../../../api/stockApi';
import { downloadBlobWithPreference } from '../../../utils/downloadFile';
import {
    resolveTickerFolder,
    rowsToCsv,
    transformFinanceCsvValue,
} from '../../../utils/exportCsv';

type CsvRow = Record<string, unknown>;
type ExportFetchResponse = { data: CsvRow[] };

export interface ExportDefinition {
    suffix: string;
    fetch: (ticker: string) => Promise<ExportFetchResponse>;
    prepareRows?: (rows: CsvRow[]) => CsvRow[];
    transformValue?: (value: unknown, key: string, row: CsvRow) => unknown;
}

export interface TickerExportResult {
    tickerFolder: string;
    total: number;
    successCount: number;
    failedCount: number;
    browserFallbackCount: number;
}

interface RunTickerExportDefinitionsInput {
    ticker: string;
    datasetName: 'company' | 'finance' | 'price_history';
    exportDefinitions: ExportDefinition[];
    category?: string;
    user: Pick<AuthUser, 'id' | 'download_folder'> | null;
}

export const COMPANY_EXPORT_DEFINITIONS: ExportDefinition[] = [
    { suffix: 'overview', fetch: stockApi.getCompanyOverview },
    { suffix: 'shareholders', fetch: stockApi.getShareholders },
    { suffix: 'officers', fetch: stockApi.getOfficers },
    { suffix: 'subsidiaries', fetch: stockApi.getSubsidiaries },
];

export const FINANCE_EXPORT_DEFINITIONS: ExportDefinition[] = [
    { suffix: 'income', fetch: stockApi.getIncomeStatement, transformValue: (value, key) => transformFinanceCsvValue(value, key) },
    { suffix: 'balance', fetch: stockApi.getBalanceSheet, transformValue: (value, key) => transformFinanceCsvValue(value, key) },
    { suffix: 'cashflow', fetch: stockApi.getCashFlow, transformValue: (value, key) => transformFinanceCsvValue(value, key) },
    { suffix: 'ratios', fetch: stockApi.getFinancialRatios, transformValue: (value, key) => transformFinanceCsvValue(value, key) },
];

const sortRowsByDateDesc = (rows: CsvRow[]) => {
    return [...rows].sort((a, b) => {
        const aDate = typeof a.date === 'string' ? a.date : '';
        const bDate = typeof b.date === 'string' ? b.date : '';
        return bDate.localeCompare(aDate);
    });
};

const PRICE_HISTORY_PRICE_KEYS = new Set(['open', 'high', 'low', 'close']);

const transformPriceHistoryCsvValue = (value: unknown, key: string) => {
    if (value === null || value === undefined) {
        return value;
    }
    if (typeof value !== 'number') {
        return value;
    }

    const lowerKey = key.toLowerCase();
    if (!PRICE_HISTORY_PRICE_KEYS.has(lowerKey)) {
        return value;
    }

    return Math.round(value * 1000);
};

export const PRICE_HISTORY_EXPORT_DEFINITIONS: ExportDefinition[] = [
    {
        suffix: 'price_history',
        fetch: stockApi.getPriceHistoryOhlcv,
        prepareRows: sortRowsByDateDesc,
        transformValue: (value, key) => transformPriceHistoryCsvValue(value, key),
    },
];

export const runTickerExportDefinitions = async (
    input: RunTickerExportDefinitionsInput,
): Promise<TickerExportResult> => {
    const tickerFolder = resolveTickerFolder(input.ticker);
    let successCount = 0;
    let failedCount = 0;
    let browserFallbackCount = 0;

    for (const definition of input.exportDefinitions) {
        try {
            const response = await definition.fetch(input.ticker);
            const rows = definition.prepareRows ? definition.prepareRows(response.data) : response.data;
            const csvContent = rowsToCsv(rows, definition.transformValue);
            const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8' });
            const subdirectories = [tickerFolder];
            if (input.datasetName !== 'price_history' && input.category) {
                subdirectories.push(input.category);
            }
            const result = await downloadBlobWithPreference({
                blob,
                filename: `${definition.suffix}.csv`,
                subdirectories,
                userId: input.user?.id,
                downloadFolder: input.user?.download_folder,
            });
            successCount += 1;
            if (result.mode === 'browser-default') {
                browserFallbackCount += 1;
            }
        } catch (error) {
            failedCount += 1;
            console.error(`Failed to export ${input.datasetName} ${definition.suffix} for ${input.ticker}:`, error);
        }
    }

    return {
        tickerFolder,
        total: input.exportDefinitions.length,
        successCount,
        failedCount,
        browserFallbackCount,
    };
};
