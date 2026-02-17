import { useState } from "react";
import Icon from '@mdi/react';
import { mdiMapMarkerPath, mdiChartBoxOutline } from '@mdi/js';
import "./App.css";
import ImputePage from "./components/ImputePage";
import PS4GExplorer from "./components/PS4GExplorer";
import BEDExplorer from "./components/BEDExplorer";
import ThemeSwitch from "./components/ThemeSwitch";

type PageType = 'imputation' | 'ps4g' | 'bed';

function App() {
  const [activePage, setActivePage] = useState<PageType>('imputation');

  return (
    <div className="app">
      {/* Global Navigation Bar */}
      <nav className="global-nav">
        <div className="nav-brand">
          <span className="brand-text">MLImpute</span>
        </div>
        <div className="nav-tabs">
          <button 
            className={`nav-tab ${activePage === 'imputation' ? 'active' : ''}`}
            onClick={() => setActivePage('imputation')}
          >
            <span className="nav-tab-icon"><Icon path={mdiMapMarkerPath} size={0.9} /></span>
            Imputation
          </button>
          <button 
            className={`nav-tab ${activePage === 'ps4g' ? 'active' : ''}`}
            onClick={() => setActivePage('ps4g')}
          >
            <span className="nav-tab-icon"><Icon path={mdiChartBoxOutline} size={0.9} /></span>
            PS4G Explorer
          </button>
          <button 
            className={`nav-tab ${activePage === 'bed' ? 'active' : ''}`}
            onClick={() => setActivePage('bed')}
          >
            <span className="nav-tab-icon"><Icon path={mdiChartBoxOutline} size={0.9} /></span>
            BED Explorer
          </button>
        </div>
        <div className="nav-spacer"></div>
        <ThemeSwitch />
      </nav>

      {/* Page Content - Both pages are always rendered but hidden via CSS to preserve state */}
      <div className="page-content">
        {/* Imputation Page */}
        <div className="imputation-page" style={{ display: activePage === 'imputation' ? 'flex' : 'none' }}>
          <ImputePage />
        </div>

        {/* PS4G Explorer Page - Full Width */}
        <div className="ps4g-page" style={{ display: activePage === 'ps4g' ? 'block' : 'none' }}>
          <PS4GExplorer />
        </div>

        {/* BED Explorer Page - Full Width */}
        <div className="bed-page" style={{ display: activePage === 'bed' ? 'block' : 'none' }}>
          <BEDExplorer />
        </div>
      </div>
    </div>
  );
}

export default App;
