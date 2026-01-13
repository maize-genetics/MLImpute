import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import Icon from '@mdi/react';
import { mdiMapMarkerPath, mdiChartBoxOutline, mdiCheck, mdiClose } from '@mdi/js';
import "./App.css";
import ImputeSidebar from "./components/ImputeSidebar";
import PS4GExplorer from "./components/PS4GExplorer";
import InteractiveMatrix from "./components/InteractiveMatrix";
import ThemeSwitch from "./components/ThemeSwitch";
import { convertVisualizationToMatrix, validateVisualizationData } from "./utils/arrayUtils";

type PageType = 'imputation' | 'ps4g';

interface ImputeResult {
  success: boolean;
  message: string;
  output_file?: string;
  execution_time?: number;
  visualization_data?: string;
}

interface VisualizationData {
  status: string;
  matrix?: {
    data: string;
    shape: number[];
    dtype: string;
  };
  row_labels?: string[];
  col_labels?: string[];
  metadata?: any;
  error?: string;
}

function App() {
  const [activePage, setActivePage] = useState<PageType>('imputation');
  const [imputeResults, setImputeResults] = useState<ImputeResult | null>(null);
  const [visualizationData, setVisualizationData] = useState<VisualizationData | null>(null);
  const [sidebarWidth, setSidebarWidth] = useState<number>(360);
  const [isResizing, setIsResizing] = useState<boolean>(false);
  const resizeRef = useRef<HTMLDivElement>(null);

  const handleImputeResults = (results: ImputeResult) => {
    setImputeResults(results);
    
    // If imputation was successful and includes visualization data, process it
    if (results.success && results.visualization_data) {
      console.log('Raw visualization data:', results.visualization_data);
      try {
        const vizData = JSON.parse(results.visualization_data);
        console.log('Parsed visualization data:', vizData);
        const validation = validateVisualizationData(vizData);
        console.log('Validation result:', validation);
        
        if (validation.isValid) {
          setVisualizationData(vizData);
        } else {
          console.error('Invalid visualization data:', validation.errors);
          setVisualizationData({
            status: 'error',
            error: `Visualization data validation failed: ${validation.errors.join(', ')}`
          });
        }
      } catch (error) {
        console.error('Failed to parse visualization data:', error);
        console.error('Raw data that failed to parse:', results.visualization_data);
        setVisualizationData({
          status: 'error',
          error: 'Failed to parse visualization data from imputation results'
        });
      }
    } else {
      console.log('No visualization data in results:', results);
    }
  };

  const handleVisualizationData = (data: VisualizationData) => {
    setVisualizationData(data);
  };

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
  }, []);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isResizing) return;
    
    const newWidth = Math.max(200, Math.min(600, e.clientX));
    setSidebarWidth(newWidth);
  }, [isResizing]);

  const handleMouseUp = useCallback(() => {
    setIsResizing(false);
  }, []);


  // Add event listeners for mouse events
  useEffect(() => {
    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
    } else {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
  }, [isResizing, handleMouseMove, handleMouseUp]);

  // Prepare matrix data for D3Matrix component
  const matrixData = useMemo(() => {
    if (visualizationData && visualizationData.status === 'success') {
      // Use imputation results
      if (visualizationData.matrix) {
        return convertVisualizationToMatrix(visualizationData);
      }
    }
    
    // No data available - return null to show empty state
    return null;
  }, [visualizationData]);

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
        </div>
        <div className="nav-spacer"></div>
        <ThemeSwitch />
      </nav>

      {/* Page Content */}
      <div className="page-content">
        {activePage === 'imputation' ? (
          // Imputation Page - Sidebar + Visualization Layout
          <div className="imputation-layout">
            <div 
              className="sidebar-container"
              style={{ 
                width: `${sidebarWidth}px`,
                transition: isResizing ? 'none' : 'width 0.3s ease'
              }}
            >
              <ImputeSidebar 
                onResults={handleImputeResults} 
                onVisualizationData={handleVisualizationData}
              />
              <div 
                className={`resize-handle ${isResizing ? 'resizing' : ''}`}
                onMouseDown={handleMouseDown}
                ref={resizeRef}
              />
            </div>
            <main className="main-content">
              <div className="main-header">
                <h1>ML Imputation Visualization</h1>
                {imputeResults && (
                  <div className={`status-indicator ${imputeResults.success ? 'success' : 'error'}`}>
                    {imputeResults.success ? <><Icon path={mdiCheck} size={0.8} /> Imputation Complete</> : <><Icon path={mdiClose} size={0.8} /> Imputation Failed</>}
                    {imputeResults.execution_time && (
                      <span className="execution-time">
                        {imputeResults.execution_time.toFixed(2)}s
                      </span>
                    )}
                  </div>
                )}
              </div>
              <div className="visualization-container">
                {visualizationData && visualizationData.status === 'success' ? (
                  <div style={{ padding: '1rem', display: 'flex', flexDirection: 'column', height: '100%' }}>
                    <div style={{ marginBottom: '1rem', background: '#d4edda', padding: '0.75rem', borderRadius: '0.25rem', border: '1px solid #c3e6cb', flexShrink: 0 }}>
                      <h3 style={{ margin: '0 0 0.25rem 0', color: '#155724', fontSize: '1rem', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                        <Icon path={mdiCheck} size={0.9} /> Imputation Results Loaded!
                      </h3>
                      <div style={{ fontSize: '0.75rem', color: '#155724' }}>
                        {visualizationData.matrix && (
                          <>
                            <span>
                              <strong>Matrix:</strong> {visualizationData.matrix.shape.join(' × ')} 
                              <span style={{ marginLeft: '0.75rem' }}><strong>Type:</strong> {visualizationData.matrix.dtype}</span>
                            </span>
                            {matrixData && (
                              <span style={{ marginLeft: '0.75rem' }}>
                                <strong>Displayed:</strong> {matrixData.matrix.length} × {matrixData.matrix[0]?.length || 0}
                              </span>
                            )}
                          </>
                        )}
                      </div>
                    </div>
                    
                    {matrixData && (
                      <div style={{ 
                        background: '#fff', 
                        borderRadius: '0.25rem', 
                        border: '1px solid #dee2e6', 
                        flex: 1,
                        display: 'flex',
                        flexDirection: 'column',
                        minHeight: 0,
                        overflow: 'hidden'
                      }}>
                        <InteractiveMatrix
                          data={matrixData.matrix}
                          rowLabels={matrixData.rowLabels}
                          colLabels={matrixData.colLabels}
                          highlightData={matrixData.highlights}
                          margin={{ top: 140, right: 20, bottom: 40, left: 120 }}
                          maxVisibleRows={15}
                          maxVisibleCols={30}
                        />
                      </div>
                    )}
                  </div>
                ) : visualizationData && visualizationData.error ? (
                  <div style={{ padding: '2rem' }}>
                    <div style={{ background: '#f8d7da', padding: '1rem', borderRadius: '0.25rem', border: '1px solid #f5c6cb' }}>
                      <h4 style={{ color: '#721c24', margin: '0 0 0.5rem 0', display: 'flex', alignItems: 'center', gap: '0.25rem' }}><Icon path={mdiClose} size={0.9} /> Visualization Error</h4>
                      <p style={{ color: '#721c24', margin: '0', fontSize: '0.875rem' }}>{visualizationData.error}</p>
                    </div>
                  </div>
                ) : (
                  <div style={{ padding: '4rem 2rem', textAlign: 'center', color: '#6c757d' }}>
                    <h3>Visualization Area</h3>
                    <p>Select input files and run imputation to see the results visualized here.</p>
                    
                    {imputeResults && imputeResults.success && (
                      <div style={{ marginTop: '2rem', padding: '1rem', background: '#f8f9fa', borderRadius: '0.25rem' }}>
                        <h4>Imputation Completed Successfully!</h4>
                        {imputeResults.output_file && (
                          <p><strong>BED output file:</strong> {imputeResults.output_file}</p>
                        )}
                        {imputeResults.execution_time && (
                          <p><strong>Execution time:</strong> {imputeResults.execution_time.toFixed(2)} seconds</p>
                        )}
                        <p style={{ color: '#495057', fontSize: '0.875rem', marginTop: '1rem' }}>
                          The imputation data visualization should appear above. If not visible, check the console for any errors.
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </main>
          </div>
        ) : (
          // PS4G Explorer Page - Full Width
          <div className="ps4g-page">
            <PS4GExplorer />
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
