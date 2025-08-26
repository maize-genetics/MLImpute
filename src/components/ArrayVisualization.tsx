import React, { useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { decodeNumpyArray, validateVisualizationData, getArrayStats, type VisualizationData } from '../utils/arrayUtils';

interface ArrayVisualizationProps {
  className?: string;
}

const ArrayVisualization: React.FC<ArrayVisualizationProps> = ({ className }) => {
  const [data, setData] = useState<VisualizationData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [matrix, setMatrix] = useState<number[][] | null>(null);
  const [stats, setStats] = useState<ReturnType<typeof getArrayStats> | null>(null);

  const loadSampleData = async (rows = 10, cols = 20, seed = 42) => {
    setLoading(true);
    setError(null);

    try {
      const result = await invoke<VisualizationData>('get_sample_visualization_data', {
        rows,
        cols,
        seed,
      });

      const validation = validateVisualizationData(result);
      if (!validation.isValid) {
        throw new Error(`Validation failed: ${validation.errors.join(', ')}`);
      }

      setData(result);

      if (result.matrix) {
        const decodedMatrix = decodeNumpyArray(result.matrix);
        setMatrix(decodedMatrix);
        setStats(getArrayStats(decodedMatrix));
      }
    } catch (err) {
      console.error('Error in loadSampleData:', err);
      const errorMessage = err instanceof Error ? err.message : `Unknown error: ${JSON.stringify(err)}`;
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const renderMatrix = (matrix: number[][]) => {
    const maxDisplay = 10; // Limit display size for performance
    const displayRows = Math.min(matrix.length, maxDisplay);
    const displayCols = Math.min(matrix[0]?.length || 0, maxDisplay);

    return (
      <div className="matrix-container">
        <table style={{ fontSize: '12px', borderCollapse: 'collapse' }}>
          <tbody>
            {matrix.slice(0, displayRows).map((row, i) => (
              <tr key={i}>
                {row.slice(0, displayCols).map((value, j) => (
                  <td
                    key={j}
                    style={{
                      border: '1px solid #ddd',
                      padding: '2px 4px',
                      textAlign: 'center',
                      backgroundColor: `rgba(0, 100, 200, ${Math.abs(value)})`,
                    }}
                  >
                    {value.toFixed(3)}
                  </td>
                ))}
                {row.length > displayCols && <td>...</td>}
              </tr>
            ))}
            {matrix.length > displayRows && (
              <tr>
                <td colSpan={displayCols + 1}>...</td>
              </tr>
            )}
          </tbody>
        </table>
        {(matrix.length > displayRows || (matrix[0]?.length || 0) > displayCols) && (
          <p style={{ fontSize: '12px', color: '#666', marginTop: '8px' }}>
            Showing first {displayRows} × {displayCols} elements
          </p>
        )}
      </div>
    );
  };

  return (
    <div className={className}>
      <div style={{ marginBottom: '16px' }}>
        <h3>NumPy Array Visualization</h3>
        <div style={{ display: 'flex', gap: '8px', marginBottom: '12px' }}>
          <button onClick={() => loadSampleData(5, 10, 42)} disabled={loading}>
            Load Small (5×10)
          </button>
          <button onClick={() => loadSampleData(20, 50, 123)} disabled={loading}>
            Load Medium (20×50)
          </button>
          <button onClick={() => loadSampleData(100, 200, 456)} disabled={loading}>
            Load Large (100×200)
          </button>
          <button onClick={() => loadSampleData(100, 95000, 456)} disabled={loading}>
            Stress Test (100×95000)
          </button>
        </div>
      </div>

      {loading && <p>Loading data from Python backend...</p>}

      {error && (
        <div style={{ color: 'red', marginBottom: '16px' }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {data && (
        <div>
          <div style={{ marginBottom: '16px' }}>
            <h4>Response Info</h4>
            <p><strong>Status:</strong> {data.status}</p>
            {data.metadata && (
              <div>
                <strong>Metadata:</strong>
                <pre style={{ fontSize: '12px', background: '#f5f5f5', padding: '8px' }}>
                  {JSON.stringify(data.metadata, null, 2)}
                </pre>
              </div>
            )}
          </div>

          {stats && (
            <div style={{ marginBottom: '16px' }}>
              <h4>Array Statistics</h4>
              <p><strong>Shape:</strong> {stats.shape[0]} × {stats.shape[1]}</p>
              <p><strong>Range:</strong> {stats.min.toFixed(3)} to {stats.max.toFixed(3)}</p>
              <p><strong>Mean:</strong> {stats.mean.toFixed(3)}</p>
              {stats.hasNaN && <p style={{ color: 'orange' }}>Contains NaN values</p>}
              {stats.hasInfinite && <p style={{ color: 'orange' }}>Contains infinite values</p>}
            </div>
          )}

          {data.row_labels && (
            <div style={{ marginBottom: '16px' }}>
              <h4>Row Labels</h4>
              <p style={{ fontSize: '12px' }}>
                {data.row_labels.slice(0, 10).join(', ')}
                {data.row_labels.length > 10 && ` ... (${data.row_labels.length} total)`}
              </p>
            </div>
          )}

          {data.col_labels && (
            <div style={{ marginBottom: '16px' }}>
              <h4>Column Labels</h4>
              <p style={{ fontSize: '12px' }}>
                {data.col_labels.slice(0, 10).join(', ')}
                {data.col_labels.length > 10 && ` ... (${data.col_labels.length} total)`}
              </p>
            </div>
          )}

          {matrix && (
            <div>
              <h4>Matrix Preview</h4>
              {renderMatrix(matrix)}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ArrayVisualization;