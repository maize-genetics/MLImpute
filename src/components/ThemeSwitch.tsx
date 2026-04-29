import { useState, useEffect } from 'react';
import Icon from '@mdi/react';
import { mdiWeatherSunny, mdiWeatherNight } from '@mdi/js';
import './ThemeSwitch.css';

export type Theme = 'light' | 'dark';

interface ThemeSwitchProps {
  onChange?: (theme: Theme) => void;
}

function ThemeSwitch({ onChange }: ThemeSwitchProps) {
  const [theme, setTheme] = useState<Theme>(() => {
    const saved = localStorage.getItem('theme') as Theme | null;
    if (saved === 'light' || saved === 'dark') return saved;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
    onChange?.(theme);
  }, [theme, onChange]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
  };

  const isDark = theme === 'dark';

  return (
    <button
      className={`theme-toggle ${isDark ? 'dark' : 'light'}`}
      onClick={toggleTheme}
      aria-label={`Switch to ${isDark ? 'light' : 'dark'} theme`}
      title={`Switch to ${isDark ? 'light' : 'dark'} theme`}
    >
      <span className="theme-toggle-track">
        <span className="theme-toggle-icon sun">
          <Icon path={mdiWeatherSunny} size={0.5} />
        </span>
        <span className="theme-toggle-icon moon">
          <Icon path={mdiWeatherNight} size={0.5} />
        </span>
        <span className="theme-toggle-thumb" />
      </span>
    </button>
  );
}

export default ThemeSwitch;
