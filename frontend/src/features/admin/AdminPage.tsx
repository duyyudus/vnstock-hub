import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { AuthWidget } from '../auth/AuthWidget';
import { useAuthUser } from '../auth/useAuthUser';
import {
    type IndexInfo,
    stockApi,
    type FundSyncCategory,
    type HistoryAuditActionResponse,
    type HistorySyncActionResponse,
    type SyncStatusResponse,
} from '../../api/stockApi';
import { formatDateInputValue, getErrorMessage, parseSymbolsInput, type SyncCollectionScope } from './adminUtils';
import { CompanySyncTab } from './tabs/CompanySyncTab';
import { FinanceSyncTab } from './tabs/FinanceSyncTab';
import { FundSyncTab } from './tabs/FundSyncTab';
import { HistorySyncTab } from './tabs/HistorySyncTab';
import { SchedulerTab } from './tabs/SchedulerTab';
import { SettingsTab } from './tabs/SettingsTab';

const REFRESH_INTERVAL_MS = 5000;

type AdminTab = 'settings' | 'price' | 'finance' | 'company' | 'fund' | 'scheduler';
type TransientJobType = 'price' | 'finance' | 'company' | 'fund';

export const AdminPage: React.FC = () => {
    const user = useAuthUser();

    const [activeTab, setActiveTab] = useState<AdminTab>('settings');

    const [syncStatus, setSyncStatus] = useState<SyncStatusResponse | null>(null);
    const [loadingStatus, setLoadingStatus] = useState(false);
    const [statusError, setStatusError] = useState<string | null>(null);

    const [syncSymbols, setSyncSymbols] = useState('');
    const [syncIndexSymbol, setSyncIndexSymbol] = useState('');
    const [syncCollectionScope, setSyncCollectionScope] = useState<SyncCollectionScope>('manual');
    const [forceRestart, setForceRestart] = useState(false);
    const [syncForceRefresh, setSyncForceRefresh] = useState(false);

    const [indexOptions, setIndexOptions] = useState<IndexInfo[]>([]);
    const [auditSymbols, setAuditSymbols] = useState('');
    const [auditIndexSymbol, setAuditIndexSymbol] = useState('');
    const [auditCollectionScope, setAuditCollectionScope] = useState<SyncCollectionScope>('manual');
    const [auditStartDate, setAuditStartDate] = useState('');
    const [auditEndDate, setAuditEndDate] = useState('');
    const [auditAutoRepair, setAuditAutoRepair] = useState(false);
    const [auditResult, setAuditResult] = useState<HistoryAuditActionResponse | null>(null);

    const [repairSymbols, setRepairSymbols] = useState('');
    const [repairIndexSymbol, setRepairIndexSymbol] = useState('');
    const [repairCollectionScope, setRepairCollectionScope] = useState<SyncCollectionScope>('manual');
    const [repairStartDate, setRepairStartDate] = useState('');
    const [repairEndDate, setRepairEndDate] = useState('');

    const [financeSymbols, setFinanceSymbols] = useState('');
    const [financeIndexSymbol, setFinanceIndexSymbol] = useState('');
    const [financeCollectionScope, setFinanceCollectionScope] = useState<SyncCollectionScope>('manual');
    const [financeForceRestart, setFinanceForceRestart] = useState(false);
    const [financeQuickSync, setFinanceQuickSync] = useState(false);
    const [financeForceRefresh, setFinanceForceRefresh] = useState(false);
    const [companySymbols, setCompanySymbols] = useState('');
    const [companyIndexSymbol, setCompanyIndexSymbol] = useState('');
    const [companyCollectionScope, setCompanyCollectionScope] = useState<SyncCollectionScope>('manual');
    const [companyForceRestart, setCompanyForceRestart] = useState(false);
    const [companyQuickSync, setCompanyQuickSync] = useState(false);
    const [companyForceRefresh, setCompanyForceRefresh] = useState(false);
    const [fundCategory, setFundCategory] = useState<FundSyncCategory>('ALL');
    const [portfolioCollectionSymbols, setPortfolioCollectionSymbols] = useState<string[]>([]);
    const [tradingCollectionSymbols, setTradingCollectionSymbols] = useState<string[]>([]);

    const [syncRunning, setSyncRunning] = useState(false);
    const [auditRunning, setAuditRunning] = useState(false);
    const [repairRunning, setRepairRunning] = useState(false);
    const [financeRunning, setFinanceRunning] = useState(false);
    const [companyRunning, setCompanyRunning] = useState(false);
    const [fundRunning, setFundRunning] = useState(false);
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

    const loadCollectionSymbols = useCallback(async () => {
        if (!canAccess) {
            setPortfolioCollectionSymbols([]);
            setTradingCollectionSymbols([]);
            return;
        }
        try {
            const [portfolioResponse, tradingResponse] = await Promise.all([
                stockApi.getPortfolioPositions(),
                stockApi.getTradingPositions(),
            ]);
            setPortfolioCollectionSymbols(Array.from(new Set(
                portfolioResponse.positions.map((position) => position.ticker.toUpperCase()),
            )));
            setTradingCollectionSymbols(Array.from(new Set(
                tradingResponse.positions.map((position) => position.ticker.toUpperCase()),
            )));
        } catch {
            setPortfolioCollectionSymbols([]);
            setTradingCollectionSymbols([]);
        }
    }, [canAccess]);

    useEffect(() => {
        loadStatuses();
        loadIndexOptions();
        loadCollectionSymbols();

        if (!canAccess) {
            return;
        }

        const intervalId = window.setInterval(() => {
            loadStatuses();
        }, REFRESH_INTERVAL_MS);

        return () => {
            window.clearInterval(intervalId);
        };
    }, [canAccess, loadCollectionSymbols, loadIndexOptions, loadStatuses]);

    const getCollectionSymbols = useCallback((scope: SyncCollectionScope) => {
        if (scope === 'portfolio') {
            return portfolioCollectionSymbols;
        }
        if (scope === 'trading') {
            return tradingCollectionSymbols;
        }
        return [];
    }, [portfolioCollectionSymbols, tradingCollectionSymbols]);

    const getCollectionLabel = useCallback((scope: SyncCollectionScope) => {
        if (scope === 'portfolio') {
            return 'portfolio holdings';
        }
        if (scope === 'trading') {
            return 'trading positions';
        }
        return 'manual symbols';
    }, []);

    const runAction = async <T extends HistorySyncActionResponse>(
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
        if (runtimeSyncActive) {
            await runAction(
                () => stockApi.cancelHistorySync(),
                setSyncRunning,
            );
            return;
        }

        const symbols = syncCollectionScope === 'manual'
            ? parseSymbolsInput(syncSymbols)
            : getCollectionSymbols(syncCollectionScope);
        if (syncCollectionScope !== 'manual' && symbols.length === 0) {
            setActionError(`No symbols found in ${getCollectionLabel(syncCollectionScope)}.`);
            return;
        }
        await runAction(
            () => stockApi.runHistorySync(
                forceRestart,
                symbols.length > 0 ? symbols : undefined,
                syncCollectionScope === 'manual' ? syncIndexSymbol || undefined : undefined,
                syncForceRefresh,
            ),
            setSyncRunning,
            undefined,
            'price',
        );
    };

    const handleRunAudit = async () => {
        if (runtimeAuditActive) {
            await runAction(
                () => stockApi.cancelHistoryAudit(),
                setAuditRunning,
            );
            return;
        }

        if (!auditStartDate) {
            setActionError('Please provide a start date for audit.');
            return;
        }

        const resolvedAuditEndDate = auditEndDate || formatDateInputValue();
        if (!auditEndDate) {
            setAuditEndDate(resolvedAuditEndDate);
        }

        const symbols = auditCollectionScope === 'manual'
            ? parseSymbolsInput(auditSymbols)
            : getCollectionSymbols(auditCollectionScope);
        if (auditCollectionScope !== 'manual' && symbols.length === 0) {
            setActionError(`No symbols found in ${getCollectionLabel(auditCollectionScope)}.`);
            return;
        }

        await runAction(
            () => stockApi.runHistoryAudit(
                auditStartDate,
                resolvedAuditEndDate,
                symbols.length > 0 ? symbols : undefined,
                auditCollectionScope === 'manual' ? auditIndexSymbol || undefined : undefined,
                auditAutoRepair,
            ),
            setAuditRunning,
            (result) => setAuditResult(result as HistoryAuditActionResponse),
        );
    };

    const handleRunRepair = async () => {
        if (runtimeRepairActive) {
            await runAction(
                () => stockApi.cancelHistoryRepairSync(),
                setRepairRunning,
            );
            return;
        }

        const symbols = repairCollectionScope === 'manual'
            ? parseSymbolsInput(repairSymbols)
            : getCollectionSymbols(repairCollectionScope);

        if (repairCollectionScope !== 'manual' && symbols.length === 0) {
            setActionError(`No symbols found in ${getCollectionLabel(repairCollectionScope)}.`);
            return;
        }

        if (repairCollectionScope === 'manual' && symbols.length === 0 && !repairIndexSymbol) {
            setActionError('Please provide at least one symbol or select an index scope for repair.');
            return;
        }

        if (!repairStartDate || !repairEndDate) {
            setActionError('Please provide both start date and end date for repair.');
            return;
        }

        await runAction(
            () => stockApi.runHistoryRepairSync(
                symbols.length > 0 ? symbols : undefined,
                repairStartDate,
                repairEndDate,
                repairCollectionScope === 'manual' ? repairIndexSymbol || undefined : undefined,
            ),
            setRepairRunning,
        );
    };

    const handleRunFinanceSync = async () => {
        if (runtimeFinanceActive) {
            await runAction(
                () => stockApi.cancelFinanceSync(),
                setFinanceRunning,
            );
            return;
        }

        const symbols = financeCollectionScope === 'manual'
            ? parseSymbolsInput(financeSymbols)
            : getCollectionSymbols(financeCollectionScope);
        if (financeCollectionScope !== 'manual' && symbols.length === 0) {
            setActionError(`No symbols found in ${getCollectionLabel(financeCollectionScope)}.`);
            return;
        }
        await runAction(
            () => stockApi.runFinanceSync(
                financeForceRestart,
                symbols.length > 0 ? symbols : undefined,
                financeCollectionScope === 'manual' ? financeIndexSymbol || undefined : undefined,
                financeQuickSync,
                financeForceRefresh,
            ),
            setFinanceRunning,
            undefined,
            'finance',
        );
    };

    const handleRunCompanySync = async () => {
        if (runtimeCompanyActive) {
            await runAction(
                () => stockApi.cancelCompanySync(),
                setCompanyRunning,
            );
            return;
        }

        const symbols = companyCollectionScope === 'manual'
            ? parseSymbolsInput(companySymbols)
            : getCollectionSymbols(companyCollectionScope);
        if (companyCollectionScope !== 'manual' && symbols.length === 0) {
            setActionError(`No symbols found in ${getCollectionLabel(companyCollectionScope)}.`);
            return;
        }
        await runAction(
            () => stockApi.runCompanySync(
                companyForceRestart,
                symbols.length > 0 ? symbols : undefined,
                companyCollectionScope === 'manual' ? companyIndexSymbol || undefined : undefined,
                companyQuickSync,
                companyForceRefresh,
            ),
            setCompanyRunning,
            undefined,
            'company',
        );
    };

    const handleRunFundSync = async () => {
        await runAction(
            () => stockApi.runFundSync(fundCategory),
            setFundRunning,
            undefined,
            'fund',
        );
    };

    const runtimeSync = syncStatus?.history_sync.sync;
    const runtimeAudit = syncStatus?.history_sync.audit;
    const runtimeRepair = syncStatus?.history_sync.repair;
    const runtimeFinance = syncStatus?.finance_sync;
    const runtimeCompany = syncStatus?.company_sync;
    const runtimeFund = syncStatus?.fund_performance;

    const runtimeSyncActive = Boolean(runtimeSync?.is_running);
    const runtimeAuditActive = Boolean(runtimeAudit?.is_running);
    const runtimeRepairActive = Boolean(runtimeRepair?.is_running);
    const runtimeFinanceActive = Boolean(runtimeFinance?.is_running);
    const runtimeCompanyActive = Boolean(runtimeCompany?.is_running);
    const runtimeFundActive = Boolean(runtimeFund?.is_syncing);

    const syncActive = syncRunning || runtimeSyncActive;
    const auditActive = auditRunning || runtimeAuditActive;
    const repairActive = repairRunning || runtimeRepairActive;
    const financeActive = financeRunning || runtimeFinanceActive;
    const companyActive = companyRunning || runtimeCompanyActive;
    const fundActive = fundRunning || runtimeFundActive;
    const anyJobActive = syncActive || auditActive || repairActive || financeActive || companyActive || fundActive;
    const syncActionDisabled = loadingStatus || (anyJobActive && !runtimeSyncActive);
    const auditActionDisabled = loadingStatus || (anyJobActive && !runtimeAuditActive);
    const repairActionDisabled = loadingStatus || (anyJobActive && !runtimeRepairActive);
    const financeActionDisabled = loadingStatus || (anyJobActive && !runtimeFinanceActive);
    const companyActionDisabled = loadingStatus || (anyJobActive && !runtimeCompanyActive);
    const fundActionDisabled = loadingStatus || anyJobActive;

    useEffect(() => {
        if (!isActionMessageTransient || !transientJobType) {
            return;
        }
        const transientRuntimeActive = transientJobType === 'price'
            ? runtimeSyncActive
            : transientJobType === 'finance'
                ? runtimeFinanceActive
                : transientJobType === 'company'
                    ? runtimeCompanyActive
                    : runtimeFundActive;

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
        runtimeFundActive,
        runtimeFinanceActive,
        runtimeSyncActive,
        transientJobType,
    ]);

    const syncProgressPercent = useMemo(() => {
        const value = runtimeSync?.progress ?? 0;
        return Math.max(0, Math.min(100, Math.round(value * 100)));
    }, [runtimeSync?.progress]);

    const auditProgressPercent = useMemo(() => {
        const value = runtimeAudit?.progress ?? 0;
        return Math.max(0, Math.min(100, Math.round(value * 100)));
    }, [runtimeAudit?.progress]);

    const repairProgressPercent = useMemo(() => {
        const value = runtimeRepair?.progress ?? 0;
        return Math.max(0, Math.min(100, Math.round(value * 100)));
    }, [runtimeRepair?.progress]);

    const financeProgressPercent = useMemo(() => {
        const value = runtimeFinance?.progress ?? 0;
        return Math.max(0, Math.min(100, Math.round(value * 100)));
    }, [runtimeFinance?.progress]);

    const companyProgressPercent = useMemo(() => {
        const value = runtimeCompany?.progress ?? 0;
        return Math.max(0, Math.min(100, Math.round(value * 100)));
    }, [runtimeCompany?.progress]);

    const fundProgressPercent = useMemo(() => {
        const value = runtimeFund?.progress ?? 0;
        return Math.max(0, Math.min(100, Math.round(value * 100)));
    }, [runtimeFund?.progress]);

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
                        History Sync
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
                    <button
                        role="tab"
                        className={`tab ${activeTab === 'fund' ? 'tab-active' : ''}`}
                        onClick={() => setActiveTab('fund')}
                    >
                        Fund Sync
                    </button>
                    <button
                        role="tab"
                        className={`tab ${activeTab === 'scheduler' ? 'tab-active' : ''}`}
                        onClick={() => setActiveTab('scheduler')}
                    >
                        Scheduler
                    </button>
                </div>

                {activeTab === 'settings' ? <SettingsTab /> : null}

                {activeTab === 'price' ? (
                    <HistorySyncTab
                        runtimeSync={runtimeSync}
                        runtimeAudit={runtimeAudit}
                        runtimeRepair={runtimeRepair}
                        syncProgressPercent={syncProgressPercent}
                        auditProgressPercent={auditProgressPercent}
                        repairProgressPercent={repairProgressPercent}
                        indexOptions={indexOptions}
                        syncIndexSymbol={syncIndexSymbol}
                        onSyncIndexSymbolChange={(value) => setSyncIndexSymbol(value)}
                        syncCollectionScope={syncCollectionScope}
                        onSyncCollectionScopeChange={setSyncCollectionScope}
                        syncSymbols={syncSymbols}
                        onSyncSymbolsChange={(value) => setSyncSymbols(value)}
                        forceRestart={forceRestart}
                        onForceRestartChange={(checked) => setForceRestart(checked)}
                        syncForceRefresh={syncForceRefresh}
                        onSyncForceRefreshChange={(checked) => setSyncForceRefresh(checked)}
                        onRunSync={handleRunSync}
                        syncActive={syncActive}
                        auditIndexSymbol={auditIndexSymbol}
                        onAuditIndexSymbolChange={(value) => setAuditIndexSymbol(value)}
                        auditCollectionScope={auditCollectionScope}
                        onAuditCollectionScopeChange={setAuditCollectionScope}
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
                        repairIndexSymbol={repairIndexSymbol}
                        onRepairIndexSymbolChange={(value) => setRepairIndexSymbol(value)}
                        repairCollectionScope={repairCollectionScope}
                        onRepairCollectionScopeChange={setRepairCollectionScope}
                        repairSymbols={repairSymbols}
                        onRepairSymbolsChange={(value) => setRepairSymbols(value)}
                        repairStartDate={repairStartDate}
                        onRepairStartDateChange={(value) => setRepairStartDate(value)}
                        repairEndDate={repairEndDate}
                        onRepairEndDateChange={(value) => setRepairEndDate(value)}
                        onRunRepair={handleRunRepair}
                        repairActive={repairActive}
                        anyJobActive={anyJobActive}
                        syncActionDisabled={syncActionDisabled}
                        auditActionDisabled={auditActionDisabled}
                        repairActionDisabled={repairActionDisabled}
                        syncCancelable={runtimeSyncActive}
                        auditCancelable={runtimeAuditActive}
                        repairCancelable={runtimeRepairActive}
                        auditResult={auditResult}
                        portfolioCollectionCount={portfolioCollectionSymbols.length}
                        tradingCollectionCount={tradingCollectionSymbols.length}
                    />
                ) : null}

                {activeTab === 'finance' ? (
                    <FinanceSyncTab
                        runtimeFinance={runtimeFinance}
                        financeProgressPercent={financeProgressPercent}
                        indexOptions={indexOptions}
                        financeIndexSymbol={financeIndexSymbol}
                        onFinanceIndexSymbolChange={(value) => setFinanceIndexSymbol(value)}
                        financeCollectionScope={financeCollectionScope}
                        onFinanceCollectionScopeChange={setFinanceCollectionScope}
                        financeSymbols={financeSymbols}
                        onFinanceSymbolsChange={(value) => setFinanceSymbols(value)}
                        financeForceRestart={financeForceRestart}
                        onFinanceForceRestartChange={(checked) => setFinanceForceRestart(checked)}
                        financeQuickSync={financeQuickSync}
                        onFinanceQuickSyncChange={(checked) => setFinanceQuickSync(checked)}
                        financeForceRefresh={financeForceRefresh}
                        onFinanceForceRefreshChange={(checked) => setFinanceForceRefresh(checked)}
                        onRunFinanceSync={handleRunFinanceSync}
                        financeActive={financeActive}
                        anyJobActive={anyJobActive}
                        actionDisabled={financeActionDisabled}
                        financeCancelable={runtimeFinanceActive}
                        portfolioCollectionCount={portfolioCollectionSymbols.length}
                        tradingCollectionCount={tradingCollectionSymbols.length}
                    />
                ) : null}

                {activeTab === 'company' ? (
                    <CompanySyncTab
                        runtimeCompany={runtimeCompany}
                        companyProgressPercent={companyProgressPercent}
                        indexOptions={indexOptions}
                        companyIndexSymbol={companyIndexSymbol}
                        onCompanyIndexSymbolChange={(value) => setCompanyIndexSymbol(value)}
                        companyCollectionScope={companyCollectionScope}
                        onCompanyCollectionScopeChange={setCompanyCollectionScope}
                        companySymbols={companySymbols}
                        onCompanySymbolsChange={(value) => setCompanySymbols(value)}
                        companyForceRestart={companyForceRestart}
                        onCompanyForceRestartChange={(checked) => setCompanyForceRestart(checked)}
                        companyQuickSync={companyQuickSync}
                        onCompanyQuickSyncChange={(checked) => setCompanyQuickSync(checked)}
                        companyForceRefresh={companyForceRefresh}
                        onCompanyForceRefreshChange={(checked) => setCompanyForceRefresh(checked)}
                        onRunCompanySync={handleRunCompanySync}
                        companyActive={companyActive}
                        anyJobActive={anyJobActive}
                        actionDisabled={companyActionDisabled}
                        companyCancelable={runtimeCompanyActive}
                        portfolioCollectionCount={portfolioCollectionSymbols.length}
                        tradingCollectionCount={tradingCollectionSymbols.length}
                    />
                ) : null}

                {activeTab === 'fund' ? (
                    <FundSyncTab
                        runtimeFund={runtimeFund}
                        fundProgressPercent={fundProgressPercent}
                        fundCategory={fundCategory}
                        onFundCategoryChange={setFundCategory}
                        onRunFundSync={handleRunFundSync}
                        fundActive={fundActive}
                        anyJobActive={anyJobActive}
                        actionDisabled={fundActionDisabled}
                    />
                ) : null}

                {activeTab === 'scheduler' ? (
                    <SchedulerTab indexOptions={indexOptions} />
                ) : null}

            </main>
        </div>
    );
};

export default AdminPage;
