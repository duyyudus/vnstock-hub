import { useEffect, useState } from 'react';
import {
    applyThemePreference,
    getStoredThemePreference,
    normalizeThemePreference,
    storeThemePreference,
    SYSTEM_THEME_QUERY,
    THEME_STORAGE_KEY,
    type ThemePreference,
} from '../theme';

const THEME_OPTIONS: ReadonlyArray<{ value: ThemePreference; label: string }> = [
    { value: 'dark', label: 'Dark' },
    { value: 'light', label: 'Light' },
    { value: 'system', label: 'System' },
];

export function ThemeSelector() {
    const [preference, setPreference] = useState(getStoredThemePreference);

    useEffect(() => {
        const mediaQuery = window.matchMedia(SYSTEM_THEME_QUERY);
        applyThemePreference(preference, mediaQuery.matches);

        if (preference !== 'system') {
            return;
        }

        const handleSystemThemeChange = (event: MediaQueryListEvent) => {
            applyThemePreference('system', event.matches);
        };

        mediaQuery.addEventListener('change', handleSystemThemeChange);
        return () => {
            mediaQuery.removeEventListener('change', handleSystemThemeChange);
        };
    }, [preference]);

    useEffect(() => {
        const handleStorageChange = (event: StorageEvent) => {
            if (event.key === THEME_STORAGE_KEY) {
                setPreference(normalizeThemePreference(event.newValue));
            }
        };

        window.addEventListener('storage', handleStorageChange);
        return () => {
            window.removeEventListener('storage', handleStorageChange);
        };
    }, []);

    const handleChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
        const nextPreference = normalizeThemePreference(event.target.value);
        setPreference(nextPreference);
        storeThemePreference(nextPreference);
        applyThemePreference(nextPreference);
    };

    return (
        <div className="relative">
            <select
                className="select select-ghost select-sm appearance-none !bg-none pr-8"
                value={preference}
                onChange={handleChange}
                aria-label="Select theme"
            >
                {THEME_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                        {option.label}
                    </option>
                ))}
            </select>
            <svg
                className="pointer-events-none absolute right-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-base-content/70"
                viewBox="0 0 20 20"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.75"
                aria-hidden="true"
            >
                <path d="m6 8 4 4 4-4" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
        </div>
    );
}
