export const THEME_STORAGE_KEY = 'vnstock_theme';
export const SYSTEM_THEME_QUERY = '(prefers-color-scheme: dark)';

export type ThemePreference = 'dark' | 'light' | 'system';
export type ResolvedTheme = 'dark' | 'light';

export const normalizeThemePreference = (value: string | null): ThemePreference => {
    if (value === 'light' || value === 'system') {
        return value;
    }

    return 'dark';
};

export const getStoredThemePreference = (): ThemePreference => {
    if (typeof window === 'undefined') {
        return 'dark';
    }

    try {
        return normalizeThemePreference(window.localStorage.getItem(THEME_STORAGE_KEY));
    } catch {
        return 'dark';
    }
};

export const resolveTheme = (
    preference: ThemePreference,
    prefersDark = typeof window !== 'undefined'
        && window.matchMedia(SYSTEM_THEME_QUERY).matches,
): ResolvedTheme => {
    if (preference === 'system') {
        return prefersDark ? 'dark' : 'light';
    }

    return preference;
};

export const applyThemePreference = (
    preference: ThemePreference,
    prefersDark?: boolean,
): ResolvedTheme => {
    const theme = resolveTheme(preference, prefersDark);

    if (typeof document !== 'undefined') {
        document.documentElement.setAttribute('data-theme', theme);
    }

    return theme;
};

export const storeThemePreference = (preference: ThemePreference): void => {
    if (typeof window === 'undefined') {
        return;
    }

    try {
        window.localStorage.setItem(THEME_STORAGE_KEY, preference);
    } catch {
        // Applying the theme still works when storage is unavailable.
    }
};
