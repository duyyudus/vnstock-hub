import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { AuthWidget } from '../auth/AuthWidget';
import { useAuthUser } from '../auth/useAuthUser';
import {
    type IndexInfo,
    stockApi,
    type PriceAuditActionResponse,
    type PriceSyncActionResponse,
    type SyncStatusResponse,
} from '../../api/stockApi';
import { getErrorMessage, parseSymbolsInput } from './adminUtils';
import { CompanySyncTab } from './tabs/CompanySyncTab';
import { FinanceSyncTab } from './tabs/FinanceSyncTab';
import { PriceSyncTab } from './tabs/PriceSyncTab';
import { SettingsTab } from './tabs/SettingsTab';

const REFRESH_INTERVAL_MS = 5000;

type AdminTab = 'settings' | 'price' | 'finance' | 'company';
type TransientJobType = 'price' | 'finance' | 'company';

export const AdminPage: React.FC = () => {
    const user = useAuthUser();

    const [activeTab, setActiveTab] = useState<AdminTab>('settings');

    const [syncStatus, setSyncStatus] = useState<SyncStatusResponse | null>(null);
    const [loadingStatus, setLoadingStatus] = useState(false);
    const [statusError, setStatusError] = useState<string | null>(null);

    const [syncSymbols, setSyncSymbols] = useState('');
    const [syncIndexSymbol, setSyncIndexSymbol] = useState('');
    const [forceRestart, setForceRestart] = useState(false);

    const [indexOptions, setIndexOptions] = useState<IndexInfo[]>([]);
    const [auditSymbols, setAuditSymbols] = useState('');
    const [auditIndexSymbol, setAuditIndexSymbol] = useState('');
    const [auditStartDate, setAuditStartDate] = useState('');
    const [auditEndDate, setAuditEndDate] = useState('');
    const [auditAutoRepair, setAuditAutoRepair] = useState(false);
    const [auditResult, setAuditResult] = useState<PriceAuditActionResponse | null>(null);

    const [repairSymbols, setRepairSymbols] = useState('');
    const [repairStartDate, setRepairStartDate] = useState('');
    const [repairEndDate, setRepairEndDate] = useState('');

    const [financeSymbols, setFinanceSymbols] = useState('');
    const [financeIndexSymbol, setFinanceIndexSymbol] = useState('');
    const [financeForceRestart, setFinanceForceRestart] = useState(false);
    const [financeQuickSync, setFinanceQuickSync] = useState(false);
    const [companySymbols, setCompanySymbols] = useState('');
    const [companyIndexSymbol, setCompanyIndexSymbol] = useState('');
    const [companyForceRestart, setCompanyForceRestart] = useState(false);
    const [companyQuickSync, setCompanyQuickSync] = useState(false);

    const [syncRunning, setSyncRunning] = useState(false);
    const [auditRunning, setAuditRunning] = useState(false);
    const [repairRunning, setRepairRunning] = useState(false);
    const [financeRunning, setFinanceRunning] = useState(false);
    const [companyRunning, setCompanyRunning] = useState(false);
    const [actionMessage, setActionMessage] = useState<string | null>(null);
    const [isActionMessageTransient, setIsActionMessageTransient] = useState(false);
    const [transientJobType, setTransientJobType] = useState<TransientJobType | null>(null);
    const [hasSeenTransientRuntimeActive, setHasSeenTransientRuntimeActive] = useState(false);
    const [actionError, setActionError] = useState<string | null>(null);

    const canAccess = Boolean(user);

    const loadStatuses = useCallback(async () => {
        if (!canAccess) {
            return;
        }

        setLoadingStatus(true);
        setStatusError(null);

        try {
            const sync = await stockApi.getSyncStatus();
            setSyncStatus(sync);
        } catch (error) {
            setStatusError(getErrorMessage(error));
        } finally {
            setLoadingStatus(false);
        }
    }, [canAccess]);

    const loadIndexOptions = useCallback(async () => {
        if (!canAccess) {
            return;
        }
        try {
            const indices = await stockApi.getIndices();
            setIndexOptions(indices.indices);
        } catch {
            setIndexOptions([]);
        }
    }, [canAccess]);

    useEffect(() => {
        loadStatuses();
        loadIndexOptions();

        if (!canAccess) {
            return;
        }

        const intervalId = window.setInterval(() => {
            loadStatuses();
        }, REFRESH_INTERVAL_MS);

        return () => {
            window.clearInterval(intervalId);
        };
    }, [canAccess, loadIndexOptions, loadStatuses]);

    const runAction = async <T extends PriceSyncActionResponse>(
        fn: () => Promise<T>,
        setLoading: React.Dispatch<React.SetStateAction<boolean>>,
        onSuccess?: (result: T) => void,
        jobType?: TransientJobType,
    ) => {
        setLoading(true);
        setActionMessage(null);
        setIsActionMessageTransient(false);
        setTransientJobType(null);
        setHasSeenTransientRuntimeActive(false);
        setActionError(null);

        try {
            const result = await fn();
            const isTransient = result.state === 'running' && Boolean(jobType);
            setActionMessage(result.message);
            setIsActionMessageTransient(isTransient);
            setTransientJobType(isTransient ? jobType ?? null : null);
            setHasSeenTransientRuntimeActive(false);
            onSuccess?.(result);
            await loadStatuses();
        } catch (error) {
            setActionError(getErrorMessage(error));
        } finally {
            setLoading(false);
        }
    };

    const handleRunSync = async () => {
        const symbols = parseSymbolsInput(syncSymbols);
        await runAction(
            () => stockApi.runPriceSync(
                forceRestart,
                symbols.length > 0 ? symbols : undefined,
                syncIndexSymbol || undefined,
            ),
            setSyncRunning,
            undefined,
            'price',
        );
    };

    const handleRunAudit = async () => {
        if (!auditStartDate || !auditEndDate) {
            setActionError('Please provide both start date and end date for audit.');
            return;
        }

        const symbols = parseSymbolsInput(auditSymbols);

        await runAction(
            () => stockApi.runPriceAudit(
                auditStartDate,
                auditEndDate,
                symbols.length > 0 ? symbols : undefined,
                auditIndexSymbol || undefined,
                auditAutoRepair,
            ),
            setAuditRunning,
            (result) => setAuditResult(result as PriceAuditActionResponse),
        );
    };

    const handleRunRepair = async () => {
        const symbols = parseSymbolsInput(repairSymbols);

        if (symbols.length === 0) {
            setActionError('Please provide at least one symbol for repair.');
            return;
        }

        if (!repairStartDate || !repairEndDate) {
            setActionError('Please provide both start date and end date for repair.');
            return;
        }

        await runAction(
            () => stockApi.runPriceRepairSync(symbols, repairStartDate, repairEndDate),
            setRepairRunning,
        );
    };

    const handleRunFinanceSync = async () => {
        const symbols = parseSymbolsInput(financeSymbols);
        await runAction(
            () => stockApi.runFinanceSync(
                financeForceRestart,
                symbols.length > 0 ? symbols : undefined,
                financeIndexSymbol || undefined,
                financeQuickSync,
            ),
            setFinanceRunning,
            undefined,
            'finance',
        );
    };

    const handleRunCompanySync = async () => {
        const symbols = parseSymbolsInput(companySymbols);
        await runAction(
            () => stockApi.runCompanySync(
                companyForceRestart,
                symbols.length > 0 ? symbols : undefined,
                companyIndexSymbol || undefined,
                companyQuickSync,
            ),
            setCompanyRunning,
            undefined,
            'company',
        );
    };

    const runtimeSync = syncStatus?.price_sync.sync;
    const runtimeAudit = syncStatus?.price_sync.audit;
    const runtimeRepair = syncStatus?.price_sync.repair;
    const runtimeFinance = syncStatus?.finance_sync;
    const runtimeCompany = syncStatus?.company_sync;

    const runtimeSyncActive = Boolean(runtimeSync?.is_running);
    const runtimeAuditActive = Boolean(runtimeAudit?.is_running);
    const runtimeRepairActive = Boolean(runtimeRepair?.is_running);
    const runtimeFinanceActive = Boolean(runtimeFinance?.is_running);
    const runtimeCompanyActive = Boolean(runtimeCompany?.is_running);

    const syncActive = syncRunning || runtimeSyncActive;
    const auditActive = auditRunning || runtimeAuditActive;
    const repairActive = repairRunning || runtimeRepairActive;
    const financeActive = financeRunning || runtimeFinanceActive;
    const companyActive = companyRunning || runtimeCompanyActive;
    const anyJobActive = syncActive || auditActive || repairActive || financeActive || companyActive;
    const actionDisabled = loadingStatus || anyJobActive;

    useEffect(() => {
        if (!isActionMessageTransient || !transientJobType) {
            return;
        }
        const transientRuntimeActive = transientJobType === 'price'
            ? runtimeSyncActive
            : transientJobType === 'finance'
                ? runtimeFinanceActive
                : runtimeCompanyActive;

        if (transientRuntimeActive) {
            setHasSeenTransientRuntimeActive(true);
            return;
        }
        if (!hasSeenTransientRuntimeActive) {
            return;
        }
        setActionMessage(null);
        setIsActionMessageTransient(false);
        setTransientJobType(null);
        setHasSeenTransientRuntimeActive(false);
    }, [
        hasSeenTransientRuntimeActive,
        isActionMessageTransient,
        runtimeCompanyActive,
        runtimeFinanceActive,
        runtimeSyncActive,
        transientJobType,
    ]);

    const syncProgressPercent = useMemo(() => {
        const value = runtimeSync?.progress ?? 0;
        return Math.max(0, Math.min(100, Math.round(value * 100)));
    }, [runtimeSync?.progress]);

    const financeProgressPercent = useMemo(() => {
        const value = runtimeFinance?.progress ?? 0;
        return Math.max(0, Math.min(100, Math.round(value * 100)));
    }, [runtimeFinance?.progress]);

    const companyProgressPercent = useMemo(() => {
        const value = runtimeCompany?.progress ?? 0;
        return Math.max(0, Math.min(100, Math.round(value * 100)));
    }, [runtimeCompany?.progress]);

    if (!canAccess) {
        return (
            <div className="min-h-screen bg-base-300">
                <header className="navbar bg-base-100 shadow-lg px-4 md:px-6">
                    <div className="max-w-[64rem] mx-auto w-full flex items-center">
                        <div className="flex-1">
                            <a href="/" className="btn btn-ghost btn-sm">← Back to Dashboard</a>
                        </div>
                        <div className="flex-none">
                            <AuthWidget />
                        </div>
                    </div>
                </header>
                <main className="max-w-[64rem] mx-auto p-6">
                    <div className="alert alert-warning">
                        <span>Please sign in to access the admin page.</span>
                    </div>
                </main>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-base-300">
            <header className="navbar bg-base-100 shadow-lg px-4 md:px-6">
                <div className="max-w-[100rem] mx-auto w-full flex items-center gap-3">
                    <div className="flex-1 flex items-center gap-2">
                        <a href="/" className="btn btn-ghost btn-sm">← Dashboard</a>
                        <h1 className="text-xl font-bold">Control Panel</h1>
                    </div>
                    <AuthWidget />
                </div>
            </header>

            <main className="max-w-[100rem] mx-auto p-6 space-y-6">
                {statusError ? (
                    <div className="alert alert-error">
                        <span>Status load failed: {statusError}</span>
                    </div>
                ) : null}

                {actionMessage ? (
                    <div className="alert alert-success">
                        <span>{actionMessage}</span>
                    </div>
                ) : null}

                {actionError ? (
                    <div className="alert alert-error">
                        <span>{actionError}</span>
                    </div>
                ) : null}

                <div role="tablist" className="tabs tabs-boxed w-fit">
                    <button
                        role="tab"
                        className={`tab ${activeTab === 'settings' ? 'tab-active' : ''}`}
                        onClick={() => setActiveTab('settings')}
                    >
                        Settings
                    </button>
                    <button
                        role="tab"
                        className={`tab ${activeTab === 'price' ? 'tab-active' : ''}`}
                        onClick={() => setActiveTab('price')}
                    >
                        Price Sync
                    </button>
                    <button
                        role="tab"
                        className={`tab ${activeTab === 'finance' ? 'tab-active' : ''}`}
                        onClick={() => setActiveTab('finance')}
                    >
                        Finance Sync
                    </button>
                    <button
                        role="tab"
                        className={`tab ${activeTab === 'company' ? 'tab-active' : ''}`}
                        onClick={() => setActiveTab('company')}
                    >
                        Company Sync
                    </button>
                </div>

                {activeTab === 'settings' ? <SettingsTab /> : null}

                {activeTab === 'price' ? (
                    <PriceSyncTab
                        runtimeSync={runtimeSync}
                        runtimeAudit={runtimeAudit}
                        runtimeRepair={runtimeRepair}
                        syncProgressPercent={syncProgressPercent}
                        indexOptions={indexOptions}
                        syncIndexSymbol={syncIndexSymbol}
                        onSyncIndexSymbolChange={(value) => setSyncIndexSymbol(value)}
                        syncSymbols={syncSymbols}
                        onSyncSymbolsChange={(value) => setSyncSymbols(value)}
                        forceRestart={forceRestart}
                        onForceRestartChange={(checked) => setForceRestart(checked)}
                        onRunSync={handleRunSync}
                        syncActive={syncActive}
                        auditIndexSymbol={auditIndexSymbol}
                        onAuditIndexSymbolChange={(value) => setAuditIndexSymbol(value)}
                        auditSymbols={auditSymbols}
                        onAuditSymbolsChange={(value) => setAuditSymbols(value)}
                        auditStartDate={auditStartDate}
                        onAuditStartDateChange={(value) => setAuditStartDate(value)}
                        auditEndDate={auditEndDate}
                        onAuditEndDateChange={(value) => setAuditEndDate(value)}
                        auditAutoRepair={auditAutoRepair}
                        onAuditAutoRepairChange={(checked) => setAuditAutoRepair(checked)}
                        onRunAudit={handleRunAudit}
                        auditActive={auditActive}
                        repairSymbols={repairSymbols}
                        onRepairSymbolsChange={(value) => setRepairSymbols(value)}
                        repairStartDate={repairStartDate}
                        onRepairStartDateChange={(value) => setRepairStartDate(value)}
                        repairEndDate={repairEndDate}
                        onRepairEndDateChange={(value) => setRepairEndDate(value)}
                        onRunRepair={handleRunRepair}
                        repairActive={repairActive}
                        anyJobActive={anyJobActive}
                        actionDisabled={actionDisabled}
                        auditResult={auditResult}
                    />
                ) : null}

                {activeTab === 'finance' ? (
                    <FinanceSyncTab
                        runtimeFinance={runtimeFinance}
                        financeProgressPercent={financeProgressPercent}
                        indexOptions={indexOptions}
                        financeIndexSymbol={financeIndexSymbol}
                        onFinanceIndexSymbolChange={(value) => setFinanceIndexSymbol(value)}
                        financeSymbols={financeSymbols}
                        onFinanceSymbolsChange={(value) => setFinanceSymbols(value)}
                        financeForceRestart={financeForceRestart}
                        onFinanceForceRestartChange={(checked) => setFinanceForceRestart(checked)}
                        financeQuickSync={financeQuickSync}
                        onFinanceQuickSyncChange={(checked) => setFinanceQuickSync(checked)}
                        onRunFinanceSync={handleRunFinanceSync}
                        financeActive={financeActive}
                        anyJobActive={anyJobActive}
                        actionDisabled={actionDisabled}
                    />
                ) : null}

                {activeTab === 'company' ? (
                    <CompanySyncTab
                        runtimeCompany={runtimeCompany}
                        companyProgressPercent={companyProgressPercent}
                        indexOptions={indexOptions}
                        companyIndexSymbol={companyIndexSymbol}
                        onCompanyIndexSymbolChange={(value) => setCompanyIndexSymbol(value)}
                        companySymbols={companySymbols}
                        onCompanySymbolsChange={(value) => setCompanySymbols(value)}
                        companyForceRestart={companyForceRestart}
                        onCompanyForceRestartChange={(checked) => setCompanyForceRestart(checked)}
                        companyQuickSync={companyQuickSync}
                        onCompanyQuickSyncChange={(checked) => setCompanyQuickSync(checked)}
                        onRunCompanySync={handleRunCompanySync}
                        companyActive={companyActive}
                        anyJobActive={anyJobActive}
                        actionDisabled={actionDisabled}
                    />
                ) : null}
            </main>
        </div>
    );
};

export default AdminPage;
