import React, { useState, useEffect, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen, UnlistenFn } from '@tauri-apps/api/event';
import { open } from '@tauri-apps/plugin-dialog';
import Icon from '@mdi/react';
import { mdiRefresh, mdiClose, mdiCheck, mdiChartBoxOutline, mdiChartTimeline } from '@mdi/js';
import HeatmapViewer from './HeatmapViewer';
import './PS4GExplorer.css';

interface PS4GProgress {
  rows_processed: number;
  bytes_processed: number;
  total_bytes: number;
  percent: number;
}

interface GameteInfo {
  gamete: string;
  gamete_index: number;
  read_count: number;
  weight: number;
}

interface PS4GDataRow {
  gamete_set: number[];
  ref_contig: string;
  ref_pos_binned: number;
  count: number;
}

interface PS4GMetadata {
  version: string | null;
  command: string | null;
  total_unique_counts: number | null;
  gametes: GameteInfo[];
}

interface PS4GSummary {
  total_rows: number;
  unique_positions: number;
  chromosomes: string[];
  chromosome_counts: Record<string, number>;
  gamete_count: number;
  position_range: Record<string, [number, number]>;
}

interface PS4GParseResult {
  success: boolean;
  metadata: PS4GMetadata;
  summary: PS4GSummary;
  data_preview: PS4GDataRow[];
  error: string | null;
}

interface PS4GExplorerProps {
  onDataLoaded?: (result: PS4GParseResult) => void;
}

type GameteSortKey = 'index' | 'gamete' | 'read_count' | 'proportion';
type SortDirection = 'asc' | 'desc';

const PS4GExplorer: React.FC<PS4GExplorerProps> = ({ onDataLoaded }) => {
  const [filePath, setFilePath] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [parseResult, setParseResult] = useState<PS4GParseResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'summary' | 'gametes' | 'preview' | 'heatmap'>('summary');
  const [progress, setProgress] = useState<PS4GProgress | null>(null);
  const [gameteSortKey, setGameteSortKey] = useState<GameteSortKey>('index');
  const [gameteSortDir, setGameteSortDir] = useState<SortDirection>('asc');
  const unlistenRef = useRef<UnlistenFn | null>(null);

  // Set up event listener for progress updates
  useEffect(() => {
    const setupListener = async () => {
      unlistenRef.current = await listen<PS4GProgress>('ps4g-progress', (event) => {
        setProgress(event.payload);
      });
    };

    setupListener();

    return () => {
      if (unlistenRef.current) {
        unlistenRef.current();
      }
    };
  }, []);

  const selectFile = async () => {
    try {
      const selected = await open({
        title: 'Select PS4G File',
        multiple: false,
        filters: [
          { name: 'PS4G Files (*.ps4g, *.ps4g.txt, *_ps4g.txt)', extensions: ['ps4g', 'txt'] },
          { name: 'All Files', extensions: ['*'] }
        ]
      });
      
      if (selected && typeof selected === 'string') {
        setFilePath(selected);
        await loadPS4GFile(selected);
      }
    } catch (err) {
      console.error('Error selecting file:', err);
      setError(`Error opening file dialog: ${err}`);
    }
  };

  const loadPS4GFile = async (path: string) => {
    setIsLoading(true);
    setError(null);
    setParseResult(null);
    setProgress(null);

    try {
      const result = await invoke<PS4GParseResult>('parse_ps4g_file', { filePath: path });
      
      if (result.success) {
        setParseResult(result);
        if (onDataLoaded) {
          onDataLoaded(result);
        }
      } else {
        setError(result.error || 'Failed to parse PS4G file');
      }
    } catch (err) {
      console.error('Error parsing PS4G file:', err);
      setError(`Error parsing file: ${err}`);
    } finally {
      setIsLoading(false);
      setProgress(null);
    }
  };

  const formatNumber = (num: number): string => {
    return num.toLocaleString();
  };

  const getFileName = (path: string): string => {
    return path.split(/[/\\]/).pop() || path;
  };

  return (
    <div className="ps4g-explorer">
      <div className="explorer-header">
        <h2>PS4G File Explorer</h2>
        <p className="explorer-subtitle">Load and analyze PS4G haplotype files</p>
      </div>

      <div className="file-selector">
        <div className="file-input-group">
          <input
            type="text"
            value={filePath}
            onChange={(e) => setFilePath(e.target.value)}
            placeholder="Select a PS4G file..."
            readOnly
            className="file-path-input"
          />
          <button
            onClick={selectFile}
            disabled={isLoading}
            className="browse-button"
          >
            {isLoading ? 'Loading...' : 'Browse'}
          </button>
        </div>
        {filePath && (
          <button
            onClick={() => loadPS4GFile(filePath)}
            disabled={isLoading || !filePath}
            className="reload-button"
          >
            <Icon path={mdiRefresh} size={0.7} /> Reload
          </button>
        )}
      </div>

      {error && (
        <div className="error-message">
          <span className="error-icon"><Icon path={mdiClose} size={0.8} /></span>
          {error}
        </div>
      )}

      {isLoading && (
        <div className="loading-indicator">
          <div className="spinner"></div>
          <span>Parsing PS4G file...</span>
          {progress && (
            <div className="progress-container">
              <div className="progress-bar">
                <div 
                  className="progress-fill" 
                  style={{ width: `${Math.min(progress.percent, 100)}%` }}
                ></div>
              </div>
              <div className="progress-stats">
                <span className="progress-percent">{progress.percent.toFixed(1)}%</span>
                <span className="progress-rows">{formatNumber(progress.rows_processed)} rows processed</span>
              </div>
            </div>
          )}
        </div>
      )}

      {parseResult && (
        <div className="results-container">
          <div className="file-info-bar">
            <span className="file-name">{getFileName(filePath)}</span>
            {parseResult.metadata.version && (
              <span className="file-version">v{parseResult.metadata.version}</span>
            )}
            <span className="success-badge"><Icon path={mdiCheck} size={0.6} /> Loaded</span>
          </div>

          <div className="results-tabs">
            <button
              className={`tab-button ${activeTab === 'summary' ? 'active' : ''}`}
              onClick={() => setActiveTab('summary')}
            >
              Summary
            </button>
            <button
              className={`tab-button ${activeTab === 'gametes' ? 'active' : ''}`}
              onClick={() => setActiveTab('gametes')}
            >
              Gametes ({parseResult.summary.gamete_count})
            </button>
            <button
              className={`tab-button ${activeTab === 'preview' ? 'active' : ''}`}
              onClick={() => setActiveTab('preview')}
            >
              Data Preview
            </button>
            <button
              className={`tab-button ${activeTab === 'heatmap' ? 'active' : ''}`}
              onClick={() => setActiveTab('heatmap')}
            >
              <Icon path={mdiChartTimeline} size={0.7} style={{ marginRight: '0.375rem' }} />
              Heatmap
            </button>
          </div>

          <div className="tab-content">
            {activeTab === 'summary' && (
              <div className="summary-panel">
                <div className="stats-grid">
                  <div className="stat-card">
                    <div className="stat-value">{formatNumber(parseResult.summary.total_rows)}</div>
                    <div className="stat-label">Total Rows</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-value">{parseResult.summary.chromosomes.length}</div>
                    <div className="stat-label">Chromosomes</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-value">{parseResult.summary.gamete_count}</div>
                    <div className="stat-label">Gametes</div>
                  </div>
                  {parseResult.metadata.total_unique_counts && (
                    <div className="stat-card">
                      <div className="stat-value">{formatNumber(parseResult.metadata.total_unique_counts)}</div>
                      <div className="stat-label">Total Unique Counts</div>
                    </div>
                  )}
                </div>

                <div className="chromosome-breakdown">
                  <h4>Chromosome Distribution</h4>
                  <div className="chromosome-list">
                    <div className="chromosome-row chromosome-header">
                      <span className="chr-header-label">ID</span>
                      <span className="chr-header-label"></span>
                      <span className="chr-header-label">Count</span>
                      <span className="chr-header-label">Position (bp)</span>
                    </div>
                    {parseResult.summary.chromosomes.map(chr => {
                      const count = parseResult.summary.chromosome_counts[chr] || 0;
                      const range = parseResult.summary.position_range[chr];
                      const maxCount = Math.max(...Object.values(parseResult.summary.chromosome_counts));
                      const barWidth = (count / maxCount) * 100;
                      
                      return (
                        <div key={chr} className="chromosome-row">
                          <span className="chr-name">{chr}</span>
                          <div className="chr-bar-container">
                            <div className="chr-bar" style={{ width: `${barWidth}%` }}></div>
                          </div>
                          <span className="chr-count">{formatNumber(count)}</span>
                          {range && (
                            <span className="chr-range">
                              {formatNumber(range[0])} - {formatNumber(range[1])}
                            </span>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>

                {parseResult.metadata.command && (
                  <div className="command-section">
                    <h4>Generation Command</h4>
                    <code className="command-text">{parseResult.metadata.command}</code>
                  </div>
                )}
              </div>
            )}

            {activeTab === 'gametes' && (
              <div className="gametes-panel">
                <table className="gametes-table">
                  <thead>
                    <tr>
                      <th 
                        className={`sortable ${gameteSortKey === 'index' ? 'sorted' : ''}`}
                        onClick={() => {
                          if (gameteSortKey === 'index') {
                            setGameteSortDir(gameteSortDir === 'asc' ? 'desc' : 'asc');
                          } else {
                            setGameteSortKey('index');
                            setGameteSortDir('asc');
                          }
                        }}
                      >
                        Index {gameteSortKey === 'index' && (gameteSortDir === 'asc' ? '↑' : '↓')}
                      </th>
                      <th 
                        className={`sortable ${gameteSortKey === 'gamete' ? 'sorted' : ''}`}
                        onClick={() => {
                          if (gameteSortKey === 'gamete') {
                            setGameteSortDir(gameteSortDir === 'asc' ? 'desc' : 'asc');
                          } else {
                            setGameteSortKey('gamete');
                            setGameteSortDir('asc');
                          }
                        }}
                      >
                        Gamete {gameteSortKey === 'gamete' && (gameteSortDir === 'asc' ? '↑' : '↓')}
                      </th>
                      <th 
                        className={`sortable ${gameteSortKey === 'read_count' ? 'sorted' : ''}`}
                        onClick={() => {
                          if (gameteSortKey === 'read_count') {
                            setGameteSortDir(gameteSortDir === 'asc' ? 'desc' : 'asc');
                          } else {
                            setGameteSortKey('read_count');
                            setGameteSortDir('desc');
                          }
                        }}
                      >
                        Read Count {gameteSortKey === 'read_count' && (gameteSortDir === 'asc' ? '↑' : '↓')}
                      </th>
                      <th 
                        className={`sortable ${gameteSortKey === 'proportion' ? 'sorted' : ''}`}
                        onClick={() => {
                          if (gameteSortKey === 'proportion') {
                            setGameteSortDir(gameteSortDir === 'asc' ? 'desc' : 'asc');
                          } else {
                            setGameteSortKey('proportion');
                            setGameteSortDir('desc');
                          }
                        }}
                      >
                        Proportion {gameteSortKey === 'proportion' && (gameteSortDir === 'asc' ? '↑' : '↓')}
                      </th>
                      <th>Distribution</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(() => {
                      const totalReadCount = parseResult.metadata.gametes.reduce((sum, g) => sum + g.read_count, 0);
                      const sortedGametes = [...parseResult.metadata.gametes].sort((a, b) => {
                        let comparison = 0;
                        switch (gameteSortKey) {
                          case 'index':
                            comparison = a.gamete_index - b.gamete_index;
                            break;
                          case 'gamete':
                            comparison = a.gamete.localeCompare(b.gamete);
                            break;
                          case 'read_count':
                          case 'proportion':
                            comparison = a.read_count - b.read_count;
                            break;
                        }
                        return gameteSortDir === 'asc' ? comparison : -comparison;
                      });
                      
                      return sortedGametes.map(gamete => {
                          const proportion = totalReadCount > 0 ? gamete.read_count / totalReadCount : 0;
                          const barWidth = proportion * 100;
                          
                          return (
                            <tr key={gamete.gamete_index}>
                              <td className="index-cell">{gamete.gamete_index}</td>
                              <td className="gamete-name">{gamete.gamete}</td>
                              <td className="count-cell">{formatNumber(gamete.read_count)}</td>
                              <td className="weight-cell">{(proportion * 100).toFixed(2)}%</td>
                              <td className="bar-cell">
                                <div className="proportion-bar-container">
                                  <div className="proportion-bar" style={{ width: `${barWidth}%` }}></div>
                                </div>
                              </td>
                            </tr>
                          );
                        });
                    })()}
                  </tbody>
                </table>
              </div>
            )}

            {activeTab === 'preview' && (
              <div className="preview-panel">
                <div className="preview-info">
                  Showing first {parseResult.data_preview.length} of {formatNumber(parseResult.summary.total_rows)} rows
                </div>
                <div className="preview-table-container">
                  <table className="preview-table">
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>Gamete Set</th>
                        <th>Chromosome</th>
                        <th>Position</th>
                        <th>Count</th>
                      </tr>
                    </thead>
                    <tbody>
                      {parseResult.data_preview.map((row, idx) => (
                        <tr key={idx}>
                          <td className="row-number">{idx + 1}</td>
                          <td className="gamete-set">
                            <span className="gamete-tags">
                              {row.gamete_set.map(g => (
                                <span key={g} className="gamete-tag">{g}</span>
                              ))}
                            </span>
                          </td>
                          <td className="chromosome">{row.ref_contig}</td>
                          <td className="position">{formatNumber(row.ref_pos_binned)}</td>
                          <td className="count">{row.count}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {activeTab === 'heatmap' && (
              <div className="heatmap-panel">
                <HeatmapViewer
                  filePath={filePath}
                  metadata={parseResult.metadata}
                  summary={parseResult.summary}
                />
              </div>
            )}
          </div>
        </div>
      )}

      {!parseResult && !isLoading && !error && (
        <div className="empty-state">
          <div className="empty-icon"><Icon path={mdiChartBoxOutline} size={3} /></div>
          <h3>No File Loaded</h3>
          <p>Select a PS4G file to view its contents and statistics</p>
        </div>
      )}
    </div>
  );
};

export default PS4GExplorer;

