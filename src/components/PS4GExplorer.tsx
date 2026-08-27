import React, { useState, useRef, useMemo } from 'react';
import Icon from '@mdi/react';
import { mdiRefresh, mdiClose, mdiCheck, mdiFileTable, mdiChartTimeline, mdiLayersOutline } from '@mdi/js';
import HeatmapViewer from './HeatmapViewer';
import { getBackend } from '../platform';
import type { PS4GProgress, PS4GParseResult, FileHandle, FileInput } from '../platform';
import './PS4GExplorer.css';

interface PS4GExplorerProps {
  onDataLoaded?: (result: PS4GParseResult) => void;
}

type GameteSortKey = 'index' | 'gamete' | 'read_count' | 'pct_reads' | 'pct_hits';
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
  const [overlayModalOpen, setOverlayModalOpen] = useState<boolean>(false);
  const fileHandleRef = useRef<FileHandle | null>(null);
  const fileInputRef = useRef<FileInput | null>(null);

  const { totalReads, totalHits } = useMemo(() => {
    if (!parseResult) return { totalReads: 0, totalHits: 0 };
    return {
      totalReads: parseResult.summary.total_read_count,
      // Header-declared per-gamete counts. A read whose gameteSet names
      // several gametes is credited to each of them, so this is a hit
      // total, not a read total, and generally exceeds totalReads.
      totalHits: parseResult.metadata.gametes.reduce((sum, g) => sum + g.read_count, 0),
    };
  }, [parseResult]);

  const selectFile = async () => {
    try {
      const backend = await getBackend();
      const selected = await backend.openFile({
        title: 'Select PS4G File',
        filters: [
          { name: 'PS4G Files (*.ps4g, *.ps4g.txt, *_ps4g.txt)', extensions: ['ps4g', 'txt'] },
          { name: 'All Files', extensions: ['*'] }
        ]
      });

      if (selected) {
        const displayName = typeof selected === 'string'
          ? selected
          : (selected as File).name;
        setFilePath(displayName);
        fileInputRef.current = selected;
        await loadPS4GFile(selected);
      }
    } catch (err) {
      console.error('Error selecting file:', err);
      setError(`Error opening file dialog: ${err}`);
    }
  };

  const loadPS4GFile = async (file: FileInput) => {
    setIsLoading(true);
    setError(null);
    setParseResult(null);
    setProgress(null);

    try {
      const backend = await getBackend();
      const { result, handle } = await backend.parsePS4GFile(file, (p) => {
        setProgress(p);
      });

      fileHandleRef.current = handle;

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
        <h2>PS4G Explorer</h2>
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
              onClick={() => fileInputRef.current && loadPS4GFile(fileInputRef.current)}
              disabled={isLoading || !fileInputRef.current}
              className="reload-button"
            >
              <Icon path={mdiRefresh} size={0.7} /> Reload
            </button>
          )}
        </div>
        {parseResult && activeTab === 'heatmap' && (
          <button
            className="overlay-header-button"
            onClick={() => setOverlayModalOpen(true)}
            title="Load path overlay from .npy files"
          >
            <Icon path={mdiLayersOutline} size={0.7} />
            <span>Path Overlay</span>
          </button>
        )}
        <p className="explorer-subtitle">Load and analyze PS4G haplotype files</p>
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
            <div className="file-info-items">
              <span className="file-name">{getFileName(filePath)}</span>
              {parseResult.metadata.version && (
                <span className="file-version">v{parseResult.metadata.version}</span>
              )}
              <span className="success-badge"><Icon path={mdiCheck} size={0.45} /> Loaded</span>
            </div>
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
                  <div className="stat-card" title="Sum of the count column across all data rows. Each read is counted once, even when it matches several gametes.">
                    <div className="stat-value">{formatNumber(totalReads)}</div>
                    <div className="stat-label">Total Reads</div>
                  </div>
                  <div className="stat-card" title="Sum of the per-gamete read counts declared in the file header. A read matching several gametes is counted once per gamete, so this can exceed Total Reads.">
                    <div className="stat-value">{formatNumber(totalHits)}</div>
                    <div className="stat-label">Total Hits</div>
                  </div>
                  {parseResult.metadata.total_unique_counts != null && (
                    <div className="stat-card" title="Value declared by the file's own #TotalUniqueCounts header; may differ from Total Reads if the producer computed it differently.">
                      <div className="stat-value">{formatNumber(parseResult.metadata.total_unique_counts)}</div>
                      <div className="stat-label">TotalUniqueCounts (header)</div>
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
                <div className="gametes-caption">
                  <strong>% of Reads</strong> = read count / {formatNumber(totalReads)} total reads
                  (the sum of the data <code>count</code> column). A read matching several
                  gametes is credited to each, so this column need not sum to 100%.{' '}
                  <strong>% of Hits</strong> = read count / {formatNumber(totalHits)} total gamete
                  hits, and always sums to 100%.
                </div>
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
                        className={`sortable ${gameteSortKey === 'pct_reads' ? 'sorted' : ''}`}
                        onClick={() => {
                          if (gameteSortKey === 'pct_reads') {
                            setGameteSortDir(gameteSortDir === 'asc' ? 'desc' : 'asc');
                          } else {
                            setGameteSortKey('pct_reads');
                            setGameteSortDir('desc');
                          }
                        }}
                      >
                        % of Reads {gameteSortKey === 'pct_reads' && (gameteSortDir === 'asc' ? '↑' : '↓')}
                      </th>
                      <th
                        className={`sortable ${gameteSortKey === 'pct_hits' ? 'sorted' : ''}`}
                        onClick={() => {
                          if (gameteSortKey === 'pct_hits') {
                            setGameteSortDir(gameteSortDir === 'asc' ? 'desc' : 'asc');
                          } else {
                            setGameteSortKey('pct_hits');
                            setGameteSortDir('desc');
                          }
                        }}
                      >
                        % of Hits {gameteSortKey === 'pct_hits' && (gameteSortDir === 'asc' ? '↑' : '↓')}
                      </th>
                      <th>Distribution</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(() => {
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
                          case 'pct_reads':
                          case 'pct_hits':
                            // % of Reads and % of Hits each share a single
                            // denominator across all rows, so ordering by
                            // either is identical to ordering by read_count.
                            comparison = a.read_count - b.read_count;
                            break;
                        }
                        return gameteSortDir === 'asc' ? comparison : -comparison;
                      });

                      return sortedGametes.map(gamete => {
                          const pctReads = totalReads > 0 ? gamete.read_count / totalReads : 0;
                          const pctHits = totalHits > 0 ? gamete.read_count / totalHits : 0;
                          // The Distribution bar tracks % of Hits: it's the
                          // metric that always sums to 100%, so bars partition
                          // the row width honestly. % of Reads can exceed
                          // 100% for an individual gamete, which wouldn't.
                          const barWidth = Math.min(pctHits * 100, 100);

                          return (
                            <tr key={gamete.gamete_index}>
                              <td className="index-cell">{gamete.gamete_index}</td>
                              <td className="gamete-name">{gamete.gamete}</td>
                              <td className="count-cell">{formatNumber(gamete.read_count)}</td>
                              <td className="weight-cell">{(pctReads * 100).toFixed(2)}%</td>
                              <td className="weight-cell">{(pctHits * 100).toFixed(2)}%</td>
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
                  fileHandle={fileHandleRef.current}
                  metadata={parseResult.metadata}
                  summary={parseResult.summary}
                  overlayModalOpen={overlayModalOpen}
                  onOverlayModalClose={() => setOverlayModalOpen(false)}
                />
              </div>
            )}
          </div>
        </div>
      )}

      {!parseResult && !isLoading && !error && (
        <div className="empty-state">
          <div className="empty-icon"><Icon path={mdiFileTable} size={3} /></div>
          <h3>No File Loaded</h3>
          <p>Select a PS4G file to view its contents and statistics</p>
        </div>
      )}
    </div>
  );
};

export default PS4GExplorer;

