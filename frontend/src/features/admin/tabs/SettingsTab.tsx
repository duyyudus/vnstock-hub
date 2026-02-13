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

export const SettingsTab: React.FC = () => {
    const user = useAuthUser();
    const userId = user?.id ?? null;
    const [downloadFolder, setDownloadFolder] = useState('');
    const [loading, setLoading] = useState(false);
    const [saving, setSaving] = useState(false);
    const [pickingFolder, setPickingFolder] = useState(false);
    const [localFolderLinked, setLocalFolderLinked] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState<string | null>(null);

    const directoryPickerSupported = isDirectoryPickerSupported();

    const syncCachedUserFolder = useCallback((targetUserId: number, folder: string | null) => {
        const cachedUser = authStorage.getUser();
        if (!cachedUser || cachedUser.id !== targetUserId) {
            return;
        }
        if (cachedUser.download_folder === folder) {
            return;
        }
        authStorage.setUser({
            ...cachedUser,
            download_folder: folder,
        });
    }, []);

    const refreshLocalFolderLink = useCallback(async () => {
        if (!userId) {
            setLocalFolderLinked(false);
            return;
        }
        const handle = await getStoredDirectoryHandle(userId);
        setLocalFolderLinked(Boolean(handle));
    }, [userId]);

    useEffect(() => {
        let isActive = true;

        const run = async () => {
            if (!userId) {
                setDownloadFolder('');
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
                syncCachedUserFolder(userId, response.download_folder);
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
    }, [syncCachedUserFolder, userId]);

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
            const response = await stockApi.updateUserSettings({
                download_folder: payloadFolder,
            });
            setDownloadFolder(response.download_folder ?? '');
            if (!response.download_folder) {
                await clearStoredDirectoryHandle(userId);
                setLocalFolderLinked(false);
            } else {
                await refreshLocalFolderLink();
            }
            syncCachedUserFolder(userId, response.download_folder);
            setSuccess('Settings saved.');
        } catch (err) {
            setError(getErrorMessage(err));
        } finally {
            setSaving(false);
        }
    };

    const handleChooseFolder = async () => {
        if (!user || !userId || pickingFolder || !directoryPickerSupported) {
            return;
        }
        setPickingFolder(true);
        setError(null);
        setSuccess(null);
        try {
            const handle = await pickAndStoreDirectoryHandle(userId);
            if (!handle) {
                return;
            }
            setLocalFolderLinked(true);
            if (!downloadFolder.trim()) {
                setDownloadFolder(handle.name);
                setSuccess('Folder linked locally. Click Save to store this folder label in account settings.');
                return;
            }
            setSuccess('Folder linked locally for this browser profile.');
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
            setLocalFolderLinked(false);
            syncCachedUserFolder(userId, response.download_folder);
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
                    <input
                        type="text"
                        className="input input-bordered"
                        placeholder="Leave blank to use browser default download folder"
                        value={downloadFolder}
                        onChange={(event) => setDownloadFolder(event.target.value)}
                        disabled={!user || loading || saving}
                    />
                    <span className="label-text-alt mt-1 text-base-content/60">
                        Blank means browser default download location. Non-blank requires local folder linking on this browser.
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
                        <span>Download folder is set, but no local folder is linked on this browser yet.</span>
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
                        className="btn btn-outline"
                        onClick={handleChooseFolder}
                        disabled={!user || !directoryPickerSupported || loading || saving || pickingFolder}
                    >
                        {pickingFolder ? 'Choosing...' : 'Choose Folder'}
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
