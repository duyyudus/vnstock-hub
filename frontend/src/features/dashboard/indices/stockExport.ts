import { stockApi } from '../../../api/stockApi';
import type { AuthUser, FinancialDataResponse } from '../../../api/stockApi';
import { downloadBlobWithPreference } from '../../../utils/downloadFile';
import {
    resolveTickerFolder,
    rowsToCsv,
    transformFinanceCsvValue,
} from '../../../utils/exportCsv';

type CsvRow = Record<string, unknown>;

export interface ExportDefinition {
    suffix: string;
    fetch: (ticker: string) => Promise<FinancialDataResponse>;
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
    datasetName: 'company' | 'finance';
    exportDefinitions: ExportDefinition[];
    category: string;
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
            const csvContent = rowsToCsv(response.data, definition.transformValue);
            const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8' });
            const result = await downloadBlobWithPreference({
                blob,
                filename: `${definition.suffix}.csv`,
                subdirectories: [tickerFolder, input.category],
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

