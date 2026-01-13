import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen, UnlistenFn } from '@tauri-apps/api/event';
import Icon from '@mdi/react';
import { mdiChartTimeline, mdiAlertCircle, mdiChevronDown } from '@mdi/js';
import HeatmapCanvas, { VisibleRange } from './HeatmapCanvas';
import HeatmapControls from './HeatmapControls';
import './HeatmapViewer.css';

/**
 * Natural sort comparison function for strings.
 * Handles numeric substrings so that "2" < "10" (unlike lexicographic sort).
 * For purely numeric IDs (e.g., "21", "0"), performs numeric comparison.
 */
function naturalSortCompare(a: string, b: string): number {
  // Split strings into chunks of digits and non-digits
  const splitRegex = /(\d+)/g;
  const aParts = a.split(splitRegex).filter(Boolean);
  const bParts = b.split(splitRegex).filter(Boolean);
  
  const maxLen = Math.max(aParts.length, bParts.length);
  
  for (let i = 0; i < maxLen; i++) {
    const aPart = aParts[i] || '';
    const bPart = bParts[i] || '';
    
    const aIsNum = /^\d+$/.test(aPart);
    const bIsNum = /^\d+$/.test(bPart);
    
    if (aIsNum && bIsNum) {
      // Compare as numbers
      const diff = parseInt(aPart, 10) - parseInt(bPart, 10);
      if (diff !== 0) return diff;
    } else if (aIsNum) {
      // Numbers come before letters
      return -1;
    } else if (bIsNum) {
      return 1;
    } else {
      // Compare as strings (case-insensitive)
      const cmp = aPart.toLowerCase().localeCompare(bPart.toLowerCase());
      if (cmp !== 0) return cmp;
    }
  }
  
  return 0;
}

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
  const [colorScheme, setColorScheme] = useState<'binary' | 'intensity'>('intensity');
  
  // Auto-scroll state
  const [isAutoScrolling, setIsAutoScrolling] = useState<boolean>(false);
  const [autoScrollSpeed, setAutoScrollSpeed] = useState<number>(0.5);
  const autoScrollRef = useRef<number | null>(null);
  
  const unlistenRef = useRef<UnlistenFn | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState<number>(800);
  const [visibleRange, setVisibleRange] = useState<VisibleRange | null>(null);

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

  // Auto-scroll effect
  useEffect(() => {
    if (isAutoScrolling) {
      const scrollStep = 0.0005 * autoScrollSpeed; // Base step adjusted by speed
      
      autoScrollRef.current = window.setInterval(() => {
        setScrollOffset(prev => {
          const next = prev + scrollStep;
          // Stop at the end and pause
          if (next >= 1) {
            setIsAutoScrolling(false);
            return 1;
          }
          return next;
        });
      }, 50); // Update every 50ms for smooth scrolling
    } else {
      if (autoScrollRef.current) {
        clearInterval(autoScrollRef.current);
        autoScrollRef.current = null;
      }
    }
    
    return () => {
      if (autoScrollRef.current) {
        clearInterval(autoScrollRef.current);
        autoScrollRef.current = null;
      }
    };
  }, [isAutoScrolling, autoScrollSpeed]);

  // Spacebar keyboard shortcut for play/pause auto-scroll
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Only trigger if spacebar is pressed and not in an input/textarea/select
      if (e.code === 'Space' && 
          !['INPUT', 'TEXTAREA', 'SELECT'].includes((e.target as HTMLElement).tagName)) {
        e.preventDefault();
        setIsAutoScrolling(prev => !prev);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

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
    setColorScheme('intensity');
    setIsAutoScrolling(false);
    setAutoScrollSpeed(0.5);
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

  // Handle visible range updates from the canvas
  const handleVisibleRangeChange = useCallback((range: VisibleRange) => {
    setVisibleRange(range);
  }, []);

  // Sort gamete names alphabetically (with natural sort for numeric IDs) and reorder matrix rows
  const sortedData = useMemo(() => {
    if (!matrixData) return null;
    
    const { gamete_names, matrix } = matrixData;
    
    // Create array of indices and sort by gamete name using natural sort
    const sortedIndices = gamete_names
      .map((name, index) => ({ name, index }))
      .sort((a, b) => naturalSortCompare(a.name, b.name))
      .map(item => item.index);
    
    // Reorder gamete names and matrix rows based on sorted indices
    const sortedGameteNames = sortedIndices.map(i => gamete_names[i]);
    const sortedMatrix = sortedIndices.map(i => matrix[i]);
    
    return {
      gameteNames: sortedGameteNames,
      matrix: sortedMatrix,
    };
  }, [matrixData]);

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
                  {chr} ({formatNumber(summary.chromosome_counts[chr] || 0)} observations)
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
            isAutoScrolling={isAutoScrolling}
            onToggleAutoScroll={() => setIsAutoScrolling(prev => !prev)}
            autoScrollSpeed={autoScrollSpeed}
            onAutoScrollSpeedChange={setAutoScrollSpeed}
          />
          
          <div className="heatmap-canvas-wrapper">
            <HeatmapCanvas
              matrix={sortedData?.matrix ?? matrixData.matrix}
              positions={matrixData.positions}
              gameteNames={sortedData?.gameteNames ?? matrixData.gamete_names}
              zoomLevel={zoomLevel}
              scrollOffset={scrollOffset}
              onScrollChange={setScrollOffset}
              onZoomChange={setZoomLevel}
              onVisibleRangeChange={handleVisibleRangeChange}
              cellWidthMultiplier={cellWidthMultiplier}
              showGridLines={showGridLines}
              colorScheme={colorScheme}
            />
          </div>

          <div className="heatmap-bottom-bar">
            {/* Visible range indicator */}
            {visibleRange && (
              <div className="visible-range-indicator">
                <span className="visible-range-label">Viewing:</span>
                <span className="visible-range-value">
                  {formatNumber(visibleRange.startPos)} - {formatNumber(visibleRange.endPos)} bp
                </span>
              </div>
            )}
            
            {/* Position scale above the slider */}
            <div className="position-scale">
              <span className="position-label">{formatNumber(matrixData.position_range[0])} bp</span>
              <span className="position-label">{formatNumber(Math.round((matrixData.position_range[0] + matrixData.position_range[1]) / 2))} bp</span>
              <span className="position-label">{formatNumber(matrixData.position_range[1])} bp</span>
            </div>
            
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
              {colorScheme === 'binary' ? (
                <div className="legend-item">
                  <span className="legend-color legend-has-data"></span>
                  <span className="legend-label">Has reads</span>
                </div>
              ) : (
                <div className="legend-item legend-gradient-item">
                  <span className="legend-label">Low</span>
                  <span className="legend-gradient"></span>
                  <span className="legend-label">High</span>
                </div>
              )}
              <div className="legend-hint">
                Scroll: vertical | Shift+scroll: horizontal | Ctrl+scroll: zoom | Space: play/pause
              </div>
            </div>
          </div>
        </>
      )}

      {/* Empty state */}
      {!matrixData && !isLoading && !error && (
        <div className="heatmap-empty">
          <Icon path={mdiChartTimeline} size={3} />
          <h3>Select a Chromosome</h3>
          <p>Choose a chromosome from the dropdown to visualize its heatmap</p>
        </div>
      )}
    </div>
  );
};

export default HeatmapViewer;

