import { useState, useMemo } from "react";
import "./App.css";
import ImputeSidebar from "./components/ImputeSidebar";
import D3Matrix from "./components/D3Matrix";

interface ImputeResult {
  success: boolean;
  message: string;
  output_file?: string;
  execution_time?: number;
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
  demoMatrix?: {
    matrix: number[][];
    rowLabels: string[];
    colLabels: string[];
    highlights: Array<{col: string; row: string}>;
  };
}

function App() {
  const [imputeResults, setImputeResults] = useState<ImputeResult | null>(null);
  const [visualizationData, setVisualizationData] = useState<VisualizationData | null>(null);

  const handleImputeResults = (results: ImputeResult) => {
    setImputeResults(results);
  };

  const handleVisualizationData = (data: VisualizationData) => {
    setVisualizationData(data);
  };

  // Prepare matrix data for D3Matrix component using local generation
  const matrixData = useMemo(() => {
    if (visualizationData && visualizationData.status === 'success' && visualizationData.demoMatrix) {
      // Use the demo matrix data generated locally
      return visualizationData.demoMatrix;
    }
    
    // No fallback - return null to show empty state
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
            <div style={{ padding: '1rem' }}>
              <div style={{ marginBottom: '2rem', background: '#d4edda', padding: '1rem', borderRadius: '0.25rem', border: '1px solid #c3e6cb' }}>
                <h3 style={{ margin: '0 0 0.5rem 0', color: '#155724' }}>✓ Demo Data Generated Successfully!</h3>
                <div style={{ fontSize: '0.875rem', color: '#155724' }}>
                  {visualizationData.matrix && (
                    <p style={{ margin: '0' }}>
                      <strong>Original Matrix:</strong> {visualizationData.matrix.shape.join(' × ')} 
                      <span style={{ marginLeft: '1rem' }}><strong>Data Type:</strong> {visualizationData.matrix.dtype}</span>
                    </p>
                  )}
                  {matrixData && (
                    <p style={{ margin: '0.5rem 0 0 0' }}>
                      <strong>Displayed:</strong> {matrixData.matrix.length} samples × {matrixData.matrix[0]?.length || 0} positions (subsampled for performance)
                    </p>
                  )}
                </div>
              </div>
              
              {matrixData && (
                <div style={{ background: '#fff', borderRadius: '0.25rem', border: '1px solid #dee2e6', overflow: 'auto', padding: '2rem' }}>
                  <D3Matrix
                    data={matrixData.matrix}
                    rowLabels={matrixData.rowLabels}
                    colLabels={matrixData.colLabels}
                    highlightData={matrixData.highlights}
                    margin={{ top: 80, right: 100, bottom: 80, left: 120 }}
                    maxVisibleRows={15}
                    maxVisibleCols={30}
                    cellSize={60}
                  />
                </div>
              )}
            </div>
          ) : visualizationData && visualizationData.error ? (
            <div style={{ padding: '2rem' }}>
              <div style={{ background: '#f8d7da', padding: '1rem', borderRadius: '0.25rem', border: '1px solid #f5c6cb' }}>
                <h4 style={{ color: '#721c24', margin: '0 0 0.5rem 0' }}>✗ Demo Generation Failed</h4>
                <p style={{ color: '#721c24', margin: '0', fontSize: '0.875rem' }}>{visualizationData.error}</p>
              </div>
            </div>
          ) : (
            <div style={{ padding: '4rem 2rem', textAlign: 'center', color: '#6c757d' }}>
              <h3>Visualization Area</h3>
              <p>Click "Run Demo" to generate a sample visualization matrix, or select input files and run imputation to see results here.</p>
              
              {imputeResults && imputeResults.success && (
                <div style={{ marginTop: '2rem', padding: '1rem', background: '#f8f9fa', borderRadius: '0.25rem' }}>
                  <h4>Imputation Completed Successfully!</h4>
                  {imputeResults.output_file && (
                    <p><strong>Output file:</strong> {imputeResults.output_file}</p>
                  )}
                  {imputeResults.execution_time && (
                    <p><strong>Execution time:</strong> {imputeResults.execution_time.toFixed(2)} seconds</p>
                  )}
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