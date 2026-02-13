const DOWNLOAD_PREF_DB_NAME = 'vnstock_download_preferences';
const DOWNLOAD_PREF_DB_VERSION = 1;
const DIRECTORY_HANDLE_STORE = 'directory_handles';

interface DirectoryHandleRecord {
    userId: number;
    handle: FileSystemDirectoryHandle;
    savedAt: number;
}

const openPreferencesDb = (): Promise<IDBDatabase> => {
    return new Promise((resolve, reject) => {
        if (typeof window === 'undefined' || !window.indexedDB) {
            reject(new Error('IndexedDB is not available.'));
            return;
        }

        const request = window.indexedDB.open(DOWNLOAD_PREF_DB_NAME, DOWNLOAD_PREF_DB_VERSION);

        request.onupgradeneeded = () => {
            const db = request.result;
            if (!db.objectStoreNames.contains(DIRECTORY_HANDLE_STORE)) {
                db.createObjectStore(DIRECTORY_HANDLE_STORE, { keyPath: 'userId' });
            }
        };

        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error ?? new Error('Failed to open IndexedDB.'));
    });
};

const runStoreRequest = <T>(
    mode: IDBTransactionMode,
    handler: (store: IDBObjectStore) => IDBRequest<T>,
): Promise<T> => {
    return openPreferencesDb().then((db) => new Promise((resolve, reject) => {
        const transaction = db.transaction(DIRECTORY_HANDLE_STORE, mode);
        const store = transaction.objectStore(DIRECTORY_HANDLE_STORE);
        const request = handler(store);

        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error ?? new Error('IndexedDB request failed.'));
        transaction.onerror = () => reject(transaction.error ?? new Error('IndexedDB transaction failed.'));
        transaction.oncomplete = () => {
            db.close();
        };
    }));
};

export const isDirectoryPickerSupported = () => {
    return typeof window !== 'undefined' && typeof window.showDirectoryPicker === 'function';
};

export const getStoredDirectoryHandle = async (userId: number): Promise<FileSystemDirectoryHandle | null> => {
    if (typeof window === 'undefined' || !window.indexedDB) {
        return null;
    }
    try {
        const record = await runStoreRequest<DirectoryHandleRecord | undefined>(
            'readonly',
            (store) => store.get(userId),
        );
        return record?.handle ?? null;
    } catch {
        return null;
    }
};

export const clearStoredDirectoryHandle = async (userId: number): Promise<void> => {
    if (typeof window === 'undefined' || !window.indexedDB) {
        return;
    }
    try {
        await runStoreRequest<undefined>('readwrite', (store) => store.delete(userId));
    } catch {
        // best-effort cleanup
    }
};

export const pickAndStoreDirectoryHandle = async (userId: number): Promise<FileSystemDirectoryHandle | null> => {
    if (!isDirectoryPickerSupported()) {
        return null;
    }

    try {
        const handle = await window.showDirectoryPicker?.({
            id: `vnstock-download-folder-${userId}`,
            mode: 'readwrite',
        });

        if (!handle) {
            return null;
        }

        const record: DirectoryHandleRecord = {
            userId,
            handle,
            savedAt: Date.now(),
        };
        await runStoreRequest<IDBValidKey>('readwrite', (store) => store.put(record));
        return handle;
    } catch (error) {
        if (error instanceof DOMException && error.name === 'AbortError') {
            return null;
        }
        throw error;
    }
};
