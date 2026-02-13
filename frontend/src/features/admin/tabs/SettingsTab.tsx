import React, { useCallback, useEffect, useState } from 'react';
import { authStorage, stockApi } from '../../../api/stockApi';
import useAuthUser from '../../auth/useAuthUser';
import { getErrorMessage } from '../adminUtils';
import {
    clearStoredDirectoryHandle,
    getStoredDirectoryHandle,
    isDirectoryPickerSupported,
    pickAndStoreDirectoryHandle,
} from '../../../utils/downloadFolderPreference';
import {
    DEFAULT_COMPANY_EXPORT_CATEGORY,
    DEFAULT_FINANCE_EXPORT_CATEGORY,
    resolveCompanyExportCategory,
    resolveFinanceExportCategory,
} from '../../../utils/exportCsv';

export const SettingsTab: React.FC = () => {
    const user = useAuthUser();
    const userId = user?.id ?? null;
    const [downloadFolder, setDownloadFolder] = useState('');
    const [companyExportCategory, setCompanyExportCategory] = useState('');
    const [financeExportCategory, setFinanceExportCategory] = useState('');
    const [loading, setLoading] = useState(false);
    const [saving, setSaving] = useState(false);
    const [pickingFolder, setPickingFolder] = useState(false);
    const [localFolderLinked, setLocalFolderLinked] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState<string | null>(null);

    const directoryPickerSupported = isDirectoryPickerSupported();

    const syncCachedUserSettings = useCallback((
        targetUserId: number,
        payload: {
            downloadFolder?: string | null;
            companyExportCategory?: string | null;
            financeExportCategory?: string | null;
        },
    ) => {
        const cachedUser = authStorage.getUser();
        if (!cachedUser || cachedUser.id !== targetUserId) {
            return;
        }
        const nextDownloadFolder = payload.downloadFolder !== undefined
            ? payload.downloadFolder
            : cachedUser.download_folder;
        const nextCompanyExportCategory = payload.companyExportCategory !== undefined
            ? payload.companyExportCategory
            : cachedUser.company_export_category;
        const nextFinanceExportCategory = payload.financeExportCategory !== undefined
            ? payload.financeExportCategory
            : cachedUser.finance_export_category;
        if (
            cachedUser.download_folder === nextDownloadFolder
            && cachedUser.company_export_category === nextCompanyExportCategory
            && cachedUser.finance_export_category === nextFinanceExportCategory
        ) {
            return;
        }
        authStorage.setUser({
            ...cachedUser,
            download_folder: nextDownloadFolder,
            company_export_category: nextCompanyExportCategory,
            finance_export_category: nextFinanceExportCategory,
        });
    }, []);

    const refreshLocalFolderLink = useCallback(async (): Promise<boolean> => {
        if (!userId) {
            setLocalFolderLinked(false);
            return false;
        }
        const handle = await getStoredDirectoryHandle(userId);
        const isLinked = Boolean(handle);
        setLocalFolderLinked(isLinked);
        return isLinked;
    }, [userId]);

    useEffect(() => {
        let isActive = true;

        const run = async () => {
            if (!userId) {
                setDownloadFolder('');
                setCompanyExportCategory('');
                setFinanceExportCategory('');
                setLocalFolderLinked(false);
                return;
            }

            if (isActive) {
                setLoading(true);
                setError(null);
            }
            try {
                const response = await stockApi.getUserSettings();
                const folder = response.download_folder ?? '';
                if (!isActive) {
                    return;
                }
                setDownloadFolder(folder);
                setCompanyExportCategory(response.company_export_category ?? '');
                setFinanceExportCategory(response.finance_export_category ?? '');
                syncCachedUserSettings(userId, {
                    downloadFolder: response.download_folder,
                    companyExportCategory: response.company_export_category,
                    financeExportCategory: response.finance_export_category,
                });
            } catch (err) {
                if (isActive) {
                    setError(getErrorMessage(err));
                }
            } finally {
                if (isActive) {
                    setLoading(false);
                }
            }
        };

        void run();
        return () => {
            isActive = false;
        };
    }, [syncCachedUserSettings, userId]);

    useEffect(() => {
        void refreshLocalFolderLink();
    }, [refreshLocalFolderLink]);

    const handleSave = async () => {
        if (!user || !userId || saving) {
            return;
        }
        setSaving(true);
        setError(null);
        setSuccess(null);
        try {
            const normalizedFolder = downloadFolder.trim();
            const payloadFolder = normalizedFolder || null;
            const normalizedCompanyCategory = companyExportCategory.trim();
            const normalizedFinanceCategory = financeExportCategory.trim();
            const response = await stockApi.updateUserSettings({
                download_folder: payloadFolder,
                company_export_category: normalizedCompanyCategory || null,
                finance_export_category: normalizedFinanceCategory || null,
            });
            setDownloadFolder(response.download_folder ?? '');
            setCompanyExportCategory(response.company_export_category ?? '');
            setFinanceExportCategory(response.finance_export_category ?? '');
            const savedFolder = response.download_folder?.trim() ?? '';
            if (!response.download_folder) {
                await clearStoredDirectoryHandle(userId);
                setLocalFolderLinked(false);
            } else {
                await refreshLocalFolderLink();
            }
            syncCachedUserSettings(userId, {
                downloadFolder: savedFolder || null,
                companyExportCategory: response.company_export_category,
                financeExportCategory: response.finance_export_category,
            });
            setSuccess('Settings saved.');
        } catch (err) {
            setError(getErrorMessage(err));
        } finally {
            setSaving(false);
        }
    };

    const handleLinkFolder = async () => {
        if (!user || !userId || pickingFolder || !directoryPickerSupported) {
            return;
        }
        setPickingFolder(true);
        setError(null);
        try {
            const handle = await pickAndStoreDirectoryHandle(userId);
            if (!handle) {
                return;
            }
            const response = await stockApi.updateUserSettings({
                download_folder: handle.name,
            });
            setDownloadFolder(response.download_folder ?? handle.name);
            setCompanyExportCategory(response.company_export_category ?? '');
            setFinanceExportCategory(response.finance_export_category ?? '');
            setLocalFolderLinked(true);
            syncCachedUserSettings(userId, {
                downloadFolder: response.download_folder ?? handle.name,
                companyExportCategory: response.company_export_category,
                financeExportCategory: response.finance_export_category,
            });
            setSuccess('Folder linked locally and saved to account settings.');
        } catch (err) {
            setError(getErrorMessage(err));
        } finally {
            setPickingFolder(false);
        }
    };

    const handleClear = async () => {
        if (!user || !userId || saving) {
            return;
        }
        setSaving(true);
        setError(null);
        setSuccess(null);
        try {
            const response = await stockApi.updateUserSettings({
                download_folder: null,
            });
            await clearStoredDirectoryHandle(userId);
            setDownloadFolder(response.download_folder ?? '');
            setCompanyExportCategory(response.company_export_category ?? '');
            setFinanceExportCategory(response.finance_export_category ?? '');
            setLocalFolderLinked(false);
            syncCachedUserSettings(userId, {
                downloadFolder: response.download_folder,
                companyExportCategory: response.company_export_category,
                financeExportCategory: response.finance_export_category,
            });
            setSuccess('Download folder cleared. Browser default download location will be used.');
        } catch (err) {
            setError(getErrorMessage(err));
        } finally {
            setSaving(false);
        }
    };

    return (
        <section className="card bg-base-100 shadow-lg">
            <div className="card-body space-y-4">
                <h2 className="card-title">Personal Settings</h2>

                {!user ? (
                    <div className="alert alert-warning">
                        <span>Please sign in to update settings.</span>
                    </div>
                ) : null}

                {loading ? (
                    <div className="flex items-center gap-2 text-base-content/70">
                        <span className="loading loading-spinner loading-sm"></span>
                        <span>Loading settings...</span>
                    </div>
                ) : null}

                {error ? (
                    <div className="alert alert-error">
                        <span>{error}</span>
                    </div>
                ) : null}

                {success ? (
                    <div className="alert alert-success">
                        <span>{success}</span>
                    </div>
                ) : null}

                <label className="form-control">
                    <span className="label-text">Download Folder</span>
                    <div className="join w-full">
                        <input
                            type="text"
                            className="input input-bordered join-item w-full"
                            placeholder="Choose folder to link this browser profile"
                            value={downloadFolder}
                            readOnly
                            disabled={!user || loading || saving || pickingFolder}
                        />
                        <button
                            type="button"
                            className="btn btn-outline join-item"
                            onClick={handleLinkFolder}
                            disabled={!user || !directoryPickerSupported || loading || saving || pickingFolder}
                        >
                            {pickingFolder ? 'Choosing...' : 'Choose Folder'}
                        </button>
                    </div>
                    <span className="label-text-alt mt-1 text-base-content/60">
                        Choose Folder is the only way to set this value. The browser may only expose the selected folder name.
                    </span>
                </label>

                <label className="form-control">
                    <span className="label-text">Company Export Category</span>
                    <input
                        type="text"
                        className="input input-bordered"
                        placeholder={DEFAULT_COMPANY_EXPORT_CATEGORY}
                        value={companyExportCategory}
                        onChange={(event) => setCompanyExportCategory(event.target.value)}
                        disabled={!user || loading || saving}
                    />
                    <span className="label-text-alt mt-1 text-base-content/60">
                        Current export category: {resolveCompanyExportCategory(companyExportCategory)}
                    </span>
                </label>

                <label className="form-control">
                    <span className="label-text">Finance Export Category</span>
                    <input
                        type="text"
                        className="input input-bordered"
                        placeholder={DEFAULT_FINANCE_EXPORT_CATEGORY}
                        value={financeExportCategory}
                        onChange={(event) => setFinanceExportCategory(event.target.value)}
                        disabled={!user || loading || saving}
                    />
                    <span className="label-text-alt mt-1 text-base-content/60">
                        Current export category: {resolveFinanceExportCategory(financeExportCategory)}
                    </span>
                </label>

                {!directoryPickerSupported ? (
                    <div className="alert alert-warning">
                        <span>
                            Folder picker is unavailable in this browser. Custom folder save works in Chromium-based browsers
                            (for example Chrome/Edge). Downloads still work with browser default location.
                        </span>
                    </div>
                ) : null}

                {directoryPickerSupported && user && downloadFolder.trim() && !localFolderLinked ? (
                    <div className="alert alert-info">
                        <span>
                            Download folder label is saved, but this browser profile is not linked yet. Use Choose Folder above
                            to link it before exporting.
                        </span>
                    </div>
                ) : null}

                {directoryPickerSupported && localFolderLinked ? (
                    <div className="text-sm text-success">Local folder is linked for this browser profile.</div>
                ) : null}

                <div className="flex flex-wrap gap-2">
                    <button
                        type="button"
                        className="btn btn-primary"
                        onClick={handleSave}
                        disabled={!user || loading || saving}
                    >
                        {saving ? 'Saving...' : 'Save'}
                    </button>
                    <button
                        type="button"
                        className="btn btn-ghost"
                        onClick={handleClear}
                        disabled={!user || loading || saving}
                    >
                        Clear
                    </button>
                </div>
            </div>
        </section>
    );
};
