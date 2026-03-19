import { useState } from 'react';

const THEMES = [
    'light', 'dark', 'cupcake', 'bumblebee', 'emerald', 'corporate',
    'synthwave', 'retro', 'cyberpunk', 'valentine', 'halloween', 'garden',
    'forest', 'aqua', 'lofi', 'pastel', 'fantasy', 'wireframe', 'black',
    'luxury', 'dracula', 'cmyk', 'autumn', 'business', 'acid', 'lemonade',
    'night', 'coffee', 'winter', 'dim', 'nord', 'sunset',
] as const;
const STORAGE_KEY = 'vnstock_theme';

export function ThemeSelector() {
    const [theme, setTheme] = useState(
        () => localStorage.getItem(STORAGE_KEY) || 'dark'
    );

    const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const t = e.target.value;
        setTheme(t);
        document.documentElement.setAttribute('data-theme', t);
        localStorage.setItem(STORAGE_KEY, t);
    };

    return (
        <select
            className="select select-ghost select-sm"
            value={theme}
            onChange={handleChange}
            aria-label="Select theme"
        >
            {THEMES.map(t => (
                <option key={t} value={t}>{t}</option>
            ))}
        </select>
    );
}
