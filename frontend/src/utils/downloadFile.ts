import { clearStoredDirectoryHandle, getStoredDirectoryHandle, isDirectoryPickerSupported } from './downloadFolderPreference';

export type DownloadMode = 'custom-folder' | 'browser-default';

interface DownloadBlobWithPreferenceInput {
    blob: Blob;
    filename: string;
    userId: number | null | undefined;
    downloadFolder: string | null | undefined;
}

const normalizeFilename = (value: string) => {
    const normalized = value.replace(/[\\/]+/g, '_').trim();
    return normalized || 'download.bin';
};

const downloadViaBrowser = (blob: Blob, filename: string): DownloadMode => {
    const blobUrl = window.URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = blobUrl;
    anchor.download = normalizeFilename(filename);
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    window.URL.revokeObjectURL(blobUrl);
    return 'browser-default';
};

const ensureReadWritePermission = async (handle: FileSystemDirectoryHandle): Promise<boolean> => {
    const descriptor = { mode: 'readwrite' as const };

    if (typeof handle.queryPermission === 'function') {
        const permission = await handle.queryPermission(descriptor);
        if (permission === 'granted') {
            return true;
        }
    }

    if (typeof handle.requestPermission === 'function') {
        const permission = await handle.requestPermission(descriptor);
        return permission === 'granted';
    }

    return false;
};

export const downloadBlobWithPreference = async (
    input: DownloadBlobWithPreferenceInput,
): Promise<{ mode: DownloadMode }> => {
    if (typeof window === 'undefined') {
        return { mode: 'browser-default' };
    }

    const normalizedDownloadFolder = input.downloadFolder?.trim() ?? '';
    if (!normalizedDownloadFolder || !input.userId || !isDirectoryPickerSupported()) {
        return { mode: downloadViaBrowser(input.blob, input.filename) };
    }

    const handle = await getStoredDirectoryHandle(input.userId);
    if (!handle) {
        return { mode: downloadViaBrowser(input.blob, input.filename) };
    }

    try {
        const hasPermission = await ensureReadWritePermission(handle);
        if (!hasPermission) {
            return { mode: downloadViaBrowser(input.blob, input.filename) };
        }

        const fileHandle = await handle.getFileHandle(normalizeFilename(input.filename), { create: true });
        const writable = await fileHandle.createWritable();
        await writable.write(input.blob);
        await writable.close();
        return { mode: 'custom-folder' };
    } catch {
        await clearStoredDirectoryHandle(input.userId);
        return { mode: downloadViaBrowser(input.blob, input.filename) };
    }
};
