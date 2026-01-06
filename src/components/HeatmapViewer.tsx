import React, { useState, useEffect, useCallback, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen, UnlistenFn } from '@tauri-apps/api/event';
import Icon from '@mdi/react';
import { mdiChartBoxOutline, mdiAlertCircle, mdiChevronDown } from '@mdi/js';
import HeatmapCanvas from './HeatmapCanvas';
import HeatmapControls from './HeatmapControls';
import './HeatmapViewer.css';

interface GameteInfo {
  gamete: string;
  gamete_index: number;
  read_count: number;
  weight: number;
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

interface ChromosomeMatrixResult {
  success: boolean;
  chromosome: string;
  matrix: number[][];
  positions: number[];
  gamete_names: string[];
  num_gametes: number;
  num_positions: number;
  position_range: [number, number];
  error: string | null;
}

interface ChromosomeMatrixProgress {
  rows_processed: number;
  chromosome: string;
  percent: number;
}

interface HeatmapViewerProps {
  filePath: string;
  metadata: PS4GMetadata;
  summary: PS4GSummary;
}

const HeatmapViewer: React.FC<HeatmapViewerProps> = ({
  filePath,
  metadata: _metadata,
  summary,
}) => {
  // State
  const [selectedChromosome, setSelectedChromosome] = useState<string>(summary.chromosomes[0] || '');
  const [matrixData, setMatrixData] = useState<ChromosomeMatrixResult | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [loadProgress, setLoadProgress] = useState<ChromosomeMatrixProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  // View state
  const [zoomLevel, setZoomLevel] = useState<number>(1);
  const [scrollOffset, setScrollOffset] = useState<number>(0);
  const [cellWidthMultiplier, setCellWidthMultiplier] = useState<number>(1);
  const [showGridLines, setShowGridLines] = useState<boolean>(true);
  const [colorScheme, setColorScheme] = useState<'binary' | 'intensity'>('binary');
  
  const unlistenRef = useRef<UnlistenFn | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState<number>(800);

  // Set up progress event listener
  useEffect(() => {
    const setupListener = async () => {
      unlistenRef.current = await listen<ChromosomeMatrixProgress>('chromosome-matrix-progress', (event) => {
        setLoadProgress(event.payload);
      });
    };

    setupListener();

    return () => {
      if (unlistenRef.current) {
        unlistenRef.current();
      }
    };
  }, []);

  // Monitor container width for viewport calculation
  useEffect(() => {
    const updateWidth = () => {
      if (containerRef.current) {
        setContainerWidth(containerRef.current.getBoundingClientRect().width);
      }
    };

    updateWidth();
    const observer = new ResizeObserver(updateWidth);
    if (containerRef.current) {
      observer.observe(containerRef.current);
    }
    return () => observer.disconnect();
  }, []);

  // Load chromosome data when selection changes
  const loadChromosomeData = useCallback(async (chromosome: string) => {
    if (!chromosome || !filePath) return;

    setIsLoading(true);
    setError(null);
    setLoadProgress(null);

    try {
      const result = await invoke<ChromosomeMatrixResult>('get_chromosome_matrix', {
        filePath,
        chromosome,
      });

      if (result.success) {
        setMatrixData(result);
        // Reset view when loading new chromosome
        setZoomLevel(1);
        setScrollOffset(0);
      } else {
        setError(result.error || 'Failed to load chromosome data');
        setMatrixData(null);
      }
    } catch (err) {
      console.error('Error loading chromosome matrix:', err);
      setError(`Error loading data: ${err}`);
      setMatrixData(null);
    } finally {
      setIsLoading(false);
      setLoadProgress(null);
    }
  }, [filePath]);

  // Load data when chromosome selection changes
  useEffect(() => {
    if (selectedChromosome) {
      loadChromosomeData(selectedChromosome);
    }
  }, [selectedChromosome, loadChromosomeData]);

  // Handle chromosome selection
  const handleChromosomeChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedChromosome(e.target.value);
  }, []);

  // Reset view to defaults
  const handleResetView = useCallback(() => {
    setZoomLevel(1);
    setScrollOffset(0);
    setCellWidthMultiplier(1);
    setShowGridLines(true);
    setColorScheme('binary');
  }, []);

  // Calculate viewport width percentage
  const calculateViewportWidthPercent = useCallback((): number => {
    if (!matrixData) return 1;
    const baseCellSize = 12;
    const cellWidth = baseCellSize * zoomLevel * cellWidthMultiplier;
    const totalWidth = matrixData.num_positions * cellWidth;
    const labelMargin = 100;
    const padding = 20;
    const viewportWidth = Math.max(containerWidth - labelMargin - padding, 100);
    return Math.min(1, viewportWidth / totalWidth);
  }, [matrixData, zoomLevel, cellWidthMultiplier, containerWidth]);

  // Format count
  const formatNumber = (num: number): string => num.toLocaleString();

  return (
    <div className="heatmap-viewer" ref={containerRef}>
      {/* Header with chromosome selector */}
      <div className="heatmap-header">
        <div className="chromosome-selector">
          <label htmlFor="chromosome-select">Chromosome:</label>
          <div className="select-wrapper">
            <select
              id="chromosome-select"
              value={selectedChromosome}
              onChange={handleChromosomeChange}
              disabled={isLoading}
            >
              {summary.chromosomes.map(chr => (
                <option key={chr} value={chr}>
                  {chr} ({formatNumber(summary.chromosome_counts[chr] || 0)} positions)
                </option>
              ))}
            </select>
            <Icon path={mdiChevronDown} size={0.8} className="select-icon" />
          </div>
        </div>

        {matrixData && !isLoading && (
          <div className="matrix-info">
            <span className="info-item">
              <strong>{matrixData.num_gametes}</strong> gametes
            </span>
            <span className="info-divider">×</span>
            <span className="info-item">
              <strong>{formatNumber(matrixData.num_positions)}</strong> positions
            </span>
            <span className="info-item position-range">
              ({formatNumber(matrixData.position_range[0])} - {formatNumber(matrixData.position_range[1])} bp)
            </span>
          </div>
        )}
      </div>

      {/* Loading state */}
      {isLoading && (
        <div className="heatmap-loading">
          <div className="spinner"></div>
          <span>Loading {selectedChromosome}...</span>
          {loadProgress && (
            <div className="progress-container">
              <div className="progress-bar">
                <div 
                  className="progress-fill" 
                  style={{ width: `${Math.min(loadProgress.percent, 100)}%` }}
                ></div>
              </div>
              <div className="progress-stats">
                <span className="progress-percent">{loadProgress.percent.toFixed(1)}%</span>
                <span className="progress-rows">{formatNumber(loadProgress.rows_processed)} rows processed</span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Error state */}
      {error && !isLoading && (
        <div className="heatmap-error">
          <Icon path={mdiAlertCircle} size={1.5} />
          <span>{error}</span>
          <button onClick={() => loadChromosomeData(selectedChromosome)}>Retry</button>
        </div>
      )}

      {/* Heatmap content */}
      {matrixData && !isLoading && !error && (
        <>
          <HeatmapControls
            zoomLevel={zoomLevel}
            onZoomChange={setZoomLevel}
            cellWidthMultiplier={cellWidthMultiplier}
            onCellWidthChange={setCellWidthMultiplier}
            showGridLines={showGridLines}
            onToggleGridLines={() => setShowGridLines(prev => !prev)}
            colorScheme={colorScheme}
            onToggleColorScheme={() => setColorScheme(prev => prev === 'binary' ? 'intensity' : 'binary')}
            onResetView={handleResetView}
          />
          
          <div className="heatmap-canvas-wrapper">
            <HeatmapCanvas
              matrix={matrixData.matrix}
              positions={matrixData.positions}
              gameteNames={matrixData.gamete_names}
              zoomLevel={zoomLevel}
              scrollOffset={scrollOffset}
              onScrollChange={setScrollOffset}
              onZoomChange={setZoomLevel}
              cellWidthMultiplier={cellWidthMultiplier}
              showGridLines={showGridLines}
              colorScheme={colorScheme}
            />
          </div>

          <div className="heatmap-bottom-bar">
            <div className="bottom-position-slider">
              <div 
                className="bottom-slider-track"
                onMouseDown={(e) => {
                  const rect = e.currentTarget.getBoundingClientRect();
                  const viewportPercent = calculateViewportWidthPercent();
                  const thumbWidth = Math.max(viewportPercent * rect.width, 20);
                  const maxThumbPos = rect.width - thumbWidth;
                  const clickX = e.clientX - rect.left;
                  const newOffset = Math.max(0, Math.min(1, (clickX - thumbWidth / 2) / maxThumbPos));
                  setScrollOffset(newOffset);
                  
                  const handleDrag = (moveEvent: MouseEvent) => {
                    const moveX = moveEvent.clientX - rect.left;
                    const dragOffset = Math.max(0, Math.min(1, (moveX - thumbWidth / 2) / maxThumbPos));
                    setScrollOffset(dragOffset);
                  };
                  
                  const handleUp = () => {
                    window.removeEventListener('mousemove', handleDrag);
                    window.removeEventListener('mouseup', handleUp);
                  };
                  
                  window.addEventListener('mousemove', handleDrag);
                  window.addEventListener('mouseup', handleUp);
                }}
              >
                <div 
                  className="bottom-slider-thumb"
                  style={{
                    left: `${scrollOffset * (100 - Math.max(calculateViewportWidthPercent() * 100, 2))}%`,
                    width: `${Math.max(calculateViewportWidthPercent() * 100, 2)}%`,
                  }}
                />
              </div>
            </div>
            
            <div className="bottom-bar-info">
              <div className="legend-item">
                <span className="legend-color legend-no-data"></span>
                <span className="legend-label">No data</span>
              </div>
              <div className="legend-item">
                <span className="legend-color legend-has-data"></span>
                <span className="legend-label">Has reads</span>
              </div>
              <div className="legend-hint">
                Scroll: vertical | Shift+scroll: horizontal | Ctrl+scroll: zoom
              </div>
            </div>
          </div>
        </>
      )}

      {/* Empty state */}
      {!matrixData && !isLoading && !error && (
        <div className="heatmap-empty">
          <Icon path={mdiChartBoxOutline} size={3} />
          <h3>Select a Chromosome</h3>
          <p>Choose a chromosome from the dropdown to visualize its heatmap</p>
        </div>
      )}
    </div>
  );
};

export default HeatmapViewer;

