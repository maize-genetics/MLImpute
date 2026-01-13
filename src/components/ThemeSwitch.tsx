import { useState, useEffect } from 'react';
import Icon from '@mdi/react';
import { mdiWeatherSunny, mdiWeatherNight, mdiMonitor } from '@mdi/js';
import './ThemeSwitch.css';

export type Theme = 'light' | 'dark' | 'system';

interface ThemeSwitchProps {
  onChange?: (theme: Theme) => void;
}

function ThemeSwitch({ onChange }: ThemeSwitchProps) {
  const [theme, setTheme] = useState<Theme>(() => {
    const saved = localStorage.getItem('theme') as Theme | null;
    return saved || 'system';
  });

  useEffect(() => {
    const root = document.documentElement;
    
    if (theme === 'system') {
      root.removeAttribute('data-theme');
    } else {
      root.setAttribute('data-theme', theme);
    }
    
    localStorage.setItem('theme', theme);
    onChange?.(theme);
  }, [theme, onChange]);

  const handleThemeChange = (newTheme: Theme) => {
    setTheme(newTheme);
  };

  return (
    <div className="theme-switch" role="radiogroup" aria-label="Theme selection">
      <button
        className={`theme-switch-option ${theme === 'light' ? 'active' : ''}`}
        onClick={() => handleThemeChange('light')}
        role="radio"
        aria-checked={theme === 'light'}
        aria-label="Light theme"
        title="Light theme"
      >
        <Icon path={mdiWeatherSunny} size={0.75} />
      </button>
      <button
        className={`theme-switch-option ${theme === 'system' ? 'active' : ''}`}
        onClick={() => handleThemeChange('system')}
        role="radio"
        aria-checked={theme === 'system'}
        aria-label="System theme"
        title="System theme"
      >
        <Icon path={mdiMonitor} size={0.75} />
      </button>
      <button
        className={`theme-switch-option ${theme === 'dark' ? 'active' : ''}`}
        onClick={() => handleThemeChange('dark')}
        role="radio"
        aria-checked={theme === 'dark'}
        aria-label="Dark theme"
        title="Dark theme"
      >
        <Icon path={mdiWeatherNight} size={0.75} />
      </button>
    </div>
  );
}

export default ThemeSwitch;
