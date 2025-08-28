import { useState, useMemo } from "react";
import "./App.css";
import ImputeSidebar from "./components/ImputeSidebar";
import InteractiveMatrix from "./components/InteractiveMatrix";
import { convertVisualizationToMatrix, validateVisualizationData } from "./utils/arrayUtils";

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
  const [imputeResults, setImputeResults] = useState<ImputeResult | null>(null);
  const [visualizationData, setVisualizationData] = useState<VisualizationData | null>(null);

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
      <ImputeSidebar 
        onResults={handleImputeResults} 
        onVisualizationData={handleVisualizationData}
      />
      <main className="main-content">
        <div className="main-header">
          <h1>ML Imputation Visualization</h1>
          {imputeResults && (
            <div className={`status-indicator ${imputeResults.success ? 'success' : 'error'}`}>
              {imputeResults.success ? '✓ Imputation Complete' : '✗ Imputation Failed'}
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
                <h3 style={{ margin: '0 0 0.25rem 0', color: '#155724', fontSize: '1rem' }}>
                  ✓ Imputation Results Loaded!
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
                    margin={{ top: 120, right: 20, bottom: 80, left: 120 }}
                    maxVisibleRows={15}
                    maxVisibleCols={30}
                  />
                </div>
              )}
            </div>
          ) : visualizationData && visualizationData.error ? (
            <div style={{ padding: '2rem' }}>
              <div style={{ background: '#f8d7da', padding: '1rem', borderRadius: '0.25rem', border: '1px solid #f5c6cb' }}>
                <h4 style={{ color: '#721c24', margin: '0 0 0.5rem 0' }}>✗ Visualization Error</h4>
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
  );
}

export default App;