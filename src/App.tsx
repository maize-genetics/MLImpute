import { useState } from "react";
import Icon from '@mdi/react';
import { mdiHome, mdiMapMarkerPath, mdiChartTimeline } from '@mdi/js';
import "./App.css";
import LandingPage from "./components/LandingPage";
import PS4GExplorer from "./components/PS4GExplorer";
import BEDExplorer from "./components/BEDExplorer";
import ThemeSwitch from "./components/ThemeSwitch";
import { isTauri } from "./platform";

type PageType = 'landing' | 'imputation' | 'ps4g' | 'bed';

function App() {
  const [activePage, setActivePage] = useState<PageType>('landing');

  return (
    <div className="app">
      <nav className="global-nav">
        <div className="nav-brand">
          <span className="brand-text">GRITS</span>
          <span className="brand-version">(v{__APP_VERSION__})</span>
        </div>
        <div className="nav-tabs">
          <button 
            className={`nav-tab ${activePage === 'landing' ? 'active' : ''}`}
            onClick={() => setActivePage('landing')}
          >
            <span className="nav-tab-icon"><Icon path={mdiHome} size={0.9} /></span>
            Home
          </button>
          {isTauri && (
            <button 
              className={`nav-tab ${activePage === 'imputation' ? 'active' : ''}`}
              onClick={() => setActivePage('imputation')}
            >
              <span className="nav-tab-icon"><Icon path={mdiMapMarkerPath} size={0.9} /></span>
              Imputation
            </button>
          )}
          <button 
            className={`nav-tab ${activePage === 'ps4g' ? 'active' : ''}`}
            onClick={() => setActivePage('ps4g')}
          >
            <span className="nav-tab-icon"><Icon path={mdiChartTimeline} size={0.9} /></span>
            PS4G Explorer
          </button>
          <button 
            className={`nav-tab ${activePage === 'bed' ? 'active' : ''}`}
            onClick={() => setActivePage('bed')}
          >
            <span className="nav-tab-icon"><Icon path={mdiChartTimeline} size={0.9} /></span>
            BED Explorer
          </button>
        </div>
        <div className="nav-spacer"></div>
        <ThemeSwitch />
      </nav>

      <div className="page-content">
        <div className="landing-page" style={{ display: activePage === 'landing' ? 'block' : 'none' }}>
          <LandingPage onNavigate={setActivePage} />
        </div>

        {isTauri && (
          <div className="imputation-page" style={{ display: activePage === 'imputation' ? 'flex' : 'none' }}>
            <ImputePage />
          </div>
        )}

        <div className="ps4g-page" style={{ display: activePage === 'ps4g' ? 'block' : 'none' }}>
          <PS4GExplorer />
        </div>

        <div className="bed-page" style={{ display: activePage === 'bed' ? 'block' : 'none' }}>
          <BEDExplorer />
        </div>
      </div>
    </div>
  );
}

// Lazy-loaded ImputePage (only available in Tauri)
function ImputePage() {
  const [Component, setComponent] = useState<React.FC | null>(null);

  if (!Component) {
    import("./components/ImputePage").then((mod) => {
      setComponent(() => mod.default);
    });
    return <div>Loading...</div>;
  }

  return <Component />;
}

export default App;
