import React, { useState, useEffect, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen, UnlistenFn } from '@tauri-apps/api/event';
import { open } from '@tauri-apps/plugin-dialog';
import Icon from '@mdi/react';
import { mdiRefresh, mdiClose, mdiCheck, mdiChartBoxOutline } from '@mdi/js';
import BEDHeatmapViewer from './BEDHeatmapViewer';
import './BEDExplorer.css';

interface BEDProgress {
  rows_processed: number;
  bytes_processed: number;
  total_bytes: number;
  percent: number;
}

interface BEDDataRow {
  chrom: string;
  start: number;
  end: number;
  parent1: string;
  parent2: string;
}

interface ParentStats {
  parent_id: string;
  regions_as_parent1: number;
  regions_as_parent2: number;
  total_regions: number;
  coverage_bp_as_parent1: number;
  coverage_bp_as_parent2: number;
  total_coverage_bp: number;
  chromosome_count: number;
}

interface BEDSummary {
  total_rows: number;
  chromosomes: string[];
  chromosome_counts: Record<string, number>;
  position_range: Record<string, [number, number]>;
  total_coverage_bp: number;
  avg_region_size_bp: number;
  unique_parents: string[];
  unique_parent_pairs: number;
  parent_stats: ParentStats[];
}

interface BEDParseResult {
  success: boolean;
  summary: BEDSummary;
  data_preview: BEDDataRow[];
  error: string | null;
}

const BEDExplorer: React.FC = () => {
  const [filePath, setFilePath] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [parseResult, setParseResult] = useState<BEDParseResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'summary' | 'gametes' | 'preview' | 'visualizer'>('summary');
  const [progress, setProgress] = useState<BEDProgress | null>(null);
  const [gameteSortKey, setGameteSortKey] = useState<'parent_id' | 'total_regions' | 'as_parent1' | 'as_parent2' | 'total_coverage' | 'chromosomes' | 'proportion'>('total_regions');
  const [gameteSortDir, setGameteSortDir] = useState<'asc' | 'desc'>('desc');
  const unlistenRef = useRef<UnlistenFn | null>(null);

  useEffect(() => {
    const setupListener = async () => {
      unlistenRef.current = await listen<BEDProgress>('bed-progress', (event) => {
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
        title: 'Select BED File',
        multiple: false,
        filters: [
          { name: 'BED Files (*.bed)', extensions: ['bed'] },
          { name: 'All Files', extensions: ['*'] }
        ]
      });

      if (selected && typeof selected === 'string') {
        setFilePath(selected);
        await loadBEDFile(selected);
      }
    } catch (err) {
      console.error('Error selecting file:', err);
      setError(`Error opening file dialog: ${err}`);
    }
  };

  const loadBEDFile = async (path: string) => {
    setIsLoading(true);
    setError(null);
    setParseResult(null);
    setProgress(null);

    try {
      const result = await invoke<BEDParseResult>('parse_bed_file', { filePath: path });

      if (result.success) {
        setParseResult(result);
      } else {
        setError(result.error || 'Failed to parse BED file');
      }
    } catch (err) {
      console.error('Error parsing BED file:', err);
      setError(`Error parsing file: ${err}`);
    } finally {
      setIsLoading(false);
      setProgress(null);
    }
  };

  const formatNumber = (num: number): string => {
    return num.toLocaleString();
  };

  const formatBp = (bp: number): string => {
    if (bp >= 1_000_000_000) {
      return `${(bp / 1_000_000_000).toFixed(2)} Gb`;
    } else if (bp >= 1_000_000) {
      return `${(bp / 1_000_000).toFixed(2)} Mb`;
    } else if (bp >= 1_000) {
      return `${(bp / 1_000).toFixed(2)} kb`;
    }
    return `${formatNumber(bp)} bp`;
  };

  const getFileName = (path: string): string => {
    return path.split(/[/\\]/).pop() || path;
  };

  return (
    <div className="bed-explorer">
      <div className="explorer-header">
        <h2>BED Explorer</h2>
        <div className="file-selector">
          <div className="file-input-group">
            <input
              type="text"
              value={filePath}
              onChange={(e) => setFilePath(e.target.value)}
              placeholder="Select a BED file..."
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
              onClick={() => loadBEDFile(filePath)}
              disabled={isLoading || !filePath}
              className="reload-button"
            >
              <Icon path={mdiRefresh} size={0.7} /> Reload
            </button>
          )}
        </div>
        <p className="explorer-subtitle">Load and analyze imputation BED output files</p>
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
          <span>Parsing BED file...</span>
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
            <span className="file-version">BED</span>
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
              Gametes ({parseResult.summary.unique_parents.length})
            </button>
            <button
              className={`tab-button ${activeTab === 'preview' ? 'active' : ''}`}
              onClick={() => setActiveTab('preview')}
            >
              Data Preview
            </button>
            <button
              className={`tab-button ${activeTab === 'visualizer' ? 'active' : ''}`}
              onClick={() => setActiveTab('visualizer')}
            >
              Visualizer
            </button>
          </div>

          <div className="tab-content">
            {activeTab === 'summary' && (
              <div className="summary-panel">
                <div className="stats-grid">
                  <div className="stat-card">
                    <div className="stat-value">{formatNumber(parseResult.summary.total_rows)}</div>
                    <div className="stat-label">Total Regions</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-value">{parseResult.summary.chromosomes.length}</div>
                    <div className="stat-label">Chromosomes</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-value">{formatBp(parseResult.summary.total_coverage_bp)}</div>
                    <div className="stat-label">Genome Coverage</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-value">{formatBp(parseResult.summary.avg_region_size_bp)}</div>
                    <div className="stat-label">Avg Region Size</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-value">{parseResult.summary.unique_parents.length}</div>
                    <div className="stat-label">Unique Parents</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-value">{parseResult.summary.unique_parent_pairs}</div>
                    <div className="stat-label">Unique Parent Pairs</div>
                  </div>
                </div>

                <div className="chromosome-breakdown">
                  <h4>Chromosome Distribution</h4>
                  <div className="chromosome-list">
                    <div className="chromosome-row chromosome-header">
                      <span className="chr-header-label">ID</span>
                      <span className="chr-header-label"></span>
                      <span className="chr-header-label">Regions</span>
                      <span className="chr-header-label">Range (bp)</span>
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
              </div>
            )}

            {activeTab === 'gametes' && (
              <div className="gametes-panel">
                <div className="gametes-summary-cards">
                  <div className="stat-card">
                    <div className="stat-value">{parseResult.summary.unique_parents.length}</div>
                    <div className="stat-label">Unique Parent IDs</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-value">{parseResult.summary.unique_parent_pairs}</div>
                    <div className="stat-label">Unique Parent Pairs</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-value">
                      {parseResult.summary.parent_stats.length > 0
                        ? formatNumber(Math.round(
                            parseResult.summary.parent_stats.reduce((s, p) => s + p.total_regions, 0) /
                            parseResult.summary.parent_stats.length
                          ))
                        : '0'}
                    </div>
                    <div className="stat-label">Avg Regions / Parent</div>
                  </div>
                </div>
                <table className="gametes-table">
                  <thead>
                    <tr>
                      <th
                        className={`sortable ${gameteSortKey === 'parent_id' ? 'sorted' : ''}`}
                        onClick={() => {
                          if (gameteSortKey === 'parent_id') {
                            setGameteSortDir(gameteSortDir === 'asc' ? 'desc' : 'asc');
                          } else {
                            setGameteSortKey('parent_id');
                            setGameteSortDir('asc');
                          }
                        }}
                      >
                        Parent ID {gameteSortKey === 'parent_id' && (gameteSortDir === 'asc' ? '\u2191' : '\u2193')}
                      </th>
                      <th
                        className={`sortable ${gameteSortKey === 'total_regions' ? 'sorted' : ''}`}
                        onClick={() => {
                          if (gameteSortKey === 'total_regions') {
                            setGameteSortDir(gameteSortDir === 'asc' ? 'desc' : 'asc');
                          } else {
                            setGameteSortKey('total_regions');
                            setGameteSortDir('desc');
                          }
                        }}
                      >
                        Total Regions {gameteSortKey === 'total_regions' && (gameteSortDir === 'asc' ? '\u2191' : '\u2193')}
                      </th>
                      <th
                        className={`sortable ${gameteSortKey === 'as_parent1' ? 'sorted' : ''}`}
                        onClick={() => {
                          if (gameteSortKey === 'as_parent1') {
                            setGameteSortDir(gameteSortDir === 'asc' ? 'desc' : 'asc');
                          } else {
                            setGameteSortKey('as_parent1');
                            setGameteSortDir('desc');
                          }
                        }}
                      >
                        As Parent 1 {gameteSortKey === 'as_parent1' && (gameteSortDir === 'asc' ? '\u2191' : '\u2193')}
                      </th>
                      <th
                        className={`sortable ${gameteSortKey === 'as_parent2' ? 'sorted' : ''}`}
                        onClick={() => {
                          if (gameteSortKey === 'as_parent2') {
                            setGameteSortDir(gameteSortDir === 'asc' ? 'desc' : 'asc');
                          } else {
                            setGameteSortKey('as_parent2');
                            setGameteSortDir('desc');
                          }
                        }}
                      >
                        As Parent 2 {gameteSortKey === 'as_parent2' && (gameteSortDir === 'asc' ? '\u2191' : '\u2193')}
                      </th>
                      <th
                        className={`sortable ${gameteSortKey === 'total_coverage' ? 'sorted' : ''}`}
                        onClick={() => {
                          if (gameteSortKey === 'total_coverage') {
                            setGameteSortDir(gameteSortDir === 'asc' ? 'desc' : 'asc');
                          } else {
                            setGameteSortKey('total_coverage');
                            setGameteSortDir('desc');
                          }
                        }}
                      >
                        Total Coverage {gameteSortKey === 'total_coverage' && (gameteSortDir === 'asc' ? '\u2191' : '\u2193')}
                      </th>
                      <th
                        className={`sortable ${gameteSortKey === 'chromosomes' ? 'sorted' : ''}`}
                        onClick={() => {
                          if (gameteSortKey === 'chromosomes') {
                            setGameteSortDir(gameteSortDir === 'asc' ? 'desc' : 'asc');
                          } else {
                            setGameteSortKey('chromosomes');
                            setGameteSortDir('desc');
                          }
                        }}
                      >
                        Chromosomes {gameteSortKey === 'chromosomes' && (gameteSortDir === 'asc' ? '\u2191' : '\u2193')}
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
                        Proportion {gameteSortKey === 'proportion' && (gameteSortDir === 'asc' ? '\u2191' : '\u2193')}
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {(() => {
                      const totalAllRegions = parseResult.summary.parent_stats.reduce((s, p) => s + p.total_regions, 0);
                      const maxRegions = Math.max(...parseResult.summary.parent_stats.map(p => p.total_regions));
                      const sorted = [...parseResult.summary.parent_stats].sort((a, b) => {
                        let cmp = 0;
                        switch (gameteSortKey) {
                          case 'parent_id':
                            cmp = a.parent_id.localeCompare(b.parent_id, undefined, { numeric: true });
                            break;
                          case 'total_regions':
                            cmp = a.total_regions - b.total_regions;
                            break;
                          case 'as_parent1':
                            cmp = a.regions_as_parent1 - b.regions_as_parent1;
                            break;
                          case 'as_parent2':
                            cmp = a.regions_as_parent2 - b.regions_as_parent2;
                            break;
                          case 'total_coverage':
                            cmp = a.total_coverage_bp - b.total_coverage_bp;
                            break;
                          case 'chromosomes':
                            cmp = a.chromosome_count - b.chromosome_count;
                            break;
                          case 'proportion':
                            cmp = a.total_regions - b.total_regions;
                            break;
                        }
                        return gameteSortDir === 'asc' ? cmp : -cmp;
                      });

                      return sorted.map(ps => {
                        const proportion = totalAllRegions > 0 ? ps.total_regions / totalAllRegions : 0;
                        const barWidth = maxRegions > 0 ? (ps.total_regions / maxRegions) * 100 : 0;

                        return (
                          <tr key={ps.parent_id}>
                            <td className="gamete-name">{ps.parent_id}</td>
                            <td className="count-cell">{formatNumber(ps.total_regions)}</td>
                            <td className="count-cell">{formatNumber(ps.regions_as_parent1)}</td>
                            <td className="count-cell">{formatNumber(ps.regions_as_parent2)}</td>
                            <td className="count-cell">{formatBp(ps.total_coverage_bp)}</td>
                            <td className="count-cell">{ps.chromosome_count}</td>
                            <td className="bar-cell">
                              <div className="proportion-bar-container">
                                <div className="proportion-bar" style={{ width: `${barWidth}%` }}></div>
                              </div>
                              <span className="proportion-label">{(proportion * 100).toFixed(1)}%</span>
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
                        <th>Chrom</th>
                        <th>Start</th>
                        <th>End</th>
                        <th>Parent 1</th>
                        <th>Parent 2</th>
                      </tr>
                    </thead>
                    <tbody>
                      {parseResult.data_preview.map((row, idx) => (
                        <tr key={idx}>
                          <td className="row-number">{idx + 1}</td>
                          <td className="chromosome">{row.chrom}</td>
                          <td className="position">{formatNumber(row.start)}</td>
                          <td className="position">{formatNumber(row.end)}</td>
                          <td className="parent-cell">{row.parent1}</td>
                          <td className="parent-cell">{row.parent2}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {activeTab === 'visualizer' && (
              <div className="visualizer-panel">
                <BEDHeatmapViewer
                  filePath={filePath}
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
          <p>Select a BED file to view its contents and statistics</p>
        </div>
      )}
    </div>
  );
};

export default BEDExplorer;
