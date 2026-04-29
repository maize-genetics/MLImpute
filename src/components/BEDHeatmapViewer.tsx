import React, { useState, useEffect, useCallback, useRef } from 'react';
import Icon from '@mdi/react';
import {
  mdiChartTimeline,
  mdiAlertCircle,
  mdiChevronDown,
  mdiDownload,
  mdiPlay,
  mdiPause,
  mdiEye,
  mdiEyeOff,
  mdiArrowExpandVertical,
  mdiKeyboard,
} from '@mdi/js';
import BEDHeatmapCanvas, { BEDVisibleRange, BEDHeatmapCanvasHandle } from './BEDHeatmapCanvas';
import HeatmapControls from './HeatmapControls';
import PositionSearch from './PositionSearch';
import ExportModal, { ExportSettings, PathOverlayInfo } from './ExportModal';
import { findNearestColumnIndex, calculateScrollOffset } from '../utils/positionSearch';
import { getBackend } from '../platform';
import type { FileHandle, BEDChromosomeMatrixResult, BEDMatrixProgress, BEDSummary } from '../platform';
import './HeatmapViewer.css';
import './BEDHeatmapViewer.css';

interface BEDHeatmapViewerProps {
  filePath: string;
  fileHandle: FileHandle | null;
  summary: BEDSummary;
}

const BEDHeatmapViewer: React.FC<BEDHeatmapViewerProps> = ({ filePath, fileHandle, summary }) => {
  // State
  const [selectedChromosome, setSelectedChromosome] = useState<string>(summary.chromosomes[0] || '');
  const [matrixData, setMatrixData] = useState<BEDChromosomeMatrixResult | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [loadProgress, setLoadProgress] = useState<BEDMatrixProgress | null>(null);
  const [error, setError] = useState<string | null>(null);

  // View state
  const [zoomLevel, setZoomLevel] = useState<number>(1);
  const [scrollOffset, setScrollOffset] = useState<number>(0);
  const [cellWidthMultiplier, setCellWidthMultiplier] = useState<number>(1);
  const [cellHeightMultiplier, setCellHeightMultiplier] = useState<number>(1);
  const [showGridLines, setShowGridLines] = useState<boolean>(true);
  const [showParent1Path, setShowParent1Path] = useState<boolean>(true);
  const [showParent2Path, setShowParent2Path] = useState<boolean>(true);
  const [showShortcuts, setShowShortcuts] = useState<boolean>(false);

  // Auto-scroll
  const [isAutoScrolling, setIsAutoScrolling] = useState<boolean>(false);
  const [autoScrollSpeed, setAutoScrollSpeed] = useState<number>(0.5);
  const autoScrollRef = useRef<number | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<BEDHeatmapCanvasHandle>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState<number>(800);
  const [visibleRange, setVisibleRange] = useState<BEDVisibleRange | null>(null);

  const [topControlsVisible, setTopControlsVisible] = useState<boolean>(true);

  // Export modal
  const [showExportModal, setShowExportModal] = useState<boolean>(false);

  // Container width
  useEffect(() => {
    const updateWidth = () => {
      if (containerRef.current) {
        setContainerWidth(containerRef.current.getBoundingClientRect().width);
      }
    };
    updateWidth();
    const observer = new ResizeObserver(updateWidth);
    if (containerRef.current) observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  // Load chromosome data
  const loadChromosomeData = useCallback(async (chromosome: string) => {
    if (!chromosome || !fileHandle) return;
    setIsLoading(true);
    setError(null);
    setLoadProgress(null);

    try {
      const backend = await getBackend();
      const result = await backend.getBEDChromosomeMatrix(fileHandle, chromosome, (p) => {
        setLoadProgress(p);
      });
      if (result.success) {
        setMatrixData(result);
        setZoomLevel(1);
        setScrollOffset(0);
      } else {
        setError(result.error || 'Failed to load chromosome data');
        setMatrixData(null);
      }
    } catch (err) {
      console.error('Error loading BED chromosome matrix:', err);
      setError(`Error loading data: ${err}`);
      setMatrixData(null);
    } finally {
      setIsLoading(false);
      setLoadProgress(null);
    }
  }, [fileHandle]);

  useEffect(() => {
    if (selectedChromosome) loadChromosomeData(selectedChromosome);
  }, [selectedChromosome, loadChromosomeData]);

  // Auto-scroll
  useEffect(() => {
    if (isAutoScrolling) {
      const scrollStep = 0.0005 * autoScrollSpeed;
      autoScrollRef.current = window.setInterval(() => {
        setScrollOffset(prev => {
          const next = prev + scrollStep;
          if (next >= 1) { setIsAutoScrolling(false); return 1; }
          return next;
        });
      }, 50);
    } else {
      if (autoScrollRef.current) { clearInterval(autoScrollRef.current); autoScrollRef.current = null; }
    }
    return () => { if (autoScrollRef.current) { clearInterval(autoScrollRef.current); autoScrollRef.current = null; } };
  }, [isAutoScrolling, autoScrollSpeed]);

  // Spacebar shortcut
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Space' && !['INPUT', 'TEXTAREA', 'SELECT'].includes((e.target as HTMLElement).tagName)) {
        if (!containerRef.current || containerRef.current.offsetParent === null) return;
        e.preventDefault();
        setIsAutoScrolling(prev => !prev);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const handleChromosomeChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedChromosome(e.target.value);
  }, []);

  const handleResetView = useCallback(() => {
    setZoomLevel(1);
    setScrollOffset(0);
    setCellWidthMultiplier(1);
    setCellHeightMultiplier(1);
    setShowGridLines(true);
    setShowParent1Path(true);
    setShowParent2Path(true);
    setIsAutoScrolling(false);
    setAutoScrollSpeed(0.5);
  }, []);

  const handleAutoFitHeight = useCallback(() => {
    if (!wrapperRef.current || !matrixData) return;
    const availableHeight = wrapperRef.current.clientHeight;
    const LABEL_MARGIN_TOP = 10;
    const PADDING = 10;
    const overhead = LABEL_MARGIN_TOP + PADDING * 2;
    const targetMatrixHeight = availableHeight - overhead;
    if (targetMatrixHeight <= 0) return;
    const numRows = matrixData.matrix.length;
    const baseCellSize = 12;
    const targetCellHeight = Math.floor(targetMatrixHeight / numRows);
    const newMultiplier = targetCellHeight / (baseCellSize * zoomLevel);
    setCellHeightMultiplier(Math.max(0.1, newMultiplier));
  }, [matrixData, zoomLevel]);

  const handlePositionSearch = useCallback((targetPos: number) => {
    if (!matrixData || matrixData.regions.length === 0) return;
    const idx = findNearestColumnIndex(targetPos, (i) => matrixData.regions[i].start, matrixData.regions.length);
    if (idx < 0) return;
    const offset = calculateScrollOffset(idx, matrixData.num_regions, zoomLevel, cellWidthMultiplier, containerWidth);
    if (offset !== null) setScrollOffset(offset);
  }, [matrixData, zoomLevel, cellWidthMultiplier, containerWidth]);

  const calculateViewportWidthPercent = useCallback((): number => {
    if (!matrixData) return 1;
    const baseCellSize = 12;
    const cw = baseCellSize * zoomLevel * cellWidthMultiplier;
    const totalWidth = matrixData.num_regions * cw;
    const labelMargin = 100;
    const padding = 20;
    const vw = Math.max(containerWidth - labelMargin - padding, 100);
    return Math.min(1, vw / totalWidth);
  }, [matrixData, zoomLevel, cellWidthMultiplier, containerWidth]);

  const handleVisibleRangeChange = useCallback((range: BEDVisibleRange) => {
    setVisibleRange(range);
  }, []);

  const getFileName = useCallback((path: string): string => {
    return path.split(/[/\\]/).pop() || path;
  }, []);

  const handleExportWithSettings = useCallback(async (settings: ExportSettings) => {
    setShowExportModal(false);
    if (canvasRef.current && selectedChromosome) {
      try {
        await canvasRef.current.exportToPng({
          fileId: getFileName(filePath),
          chromosome: selectedChromosome,
          title: settings.title,
          width: settings.width,
          height: settings.height,
          scale: settings.scale,
          includeCellValueLegend: settings.includeCellValueLegend,
          includePathLegend: settings.includePathLegend,
          pathVisibility: settings.pathVisibility,
          startPosition: settings.startPosition,
          endPosition: settings.endPosition,
        });
      } catch (err) {
        console.error('Export failed:', err);
      }
    }
  }, [filePath, selectedChromosome, getFileName]);

  const formatNumber = (num: number): string => num.toLocaleString();

  const showTopControls = topControlsVisible || !matrixData || isLoading || !!error;

  // Compute position range for the selected chromosome
  const positionRange = matrixData && matrixData.regions.length > 0
    ? [matrixData.regions[0].start, matrixData.regions[matrixData.regions.length - 1].end]
    : null;

  return (
    <div className="heatmap-viewer bed-heatmap-viewer" ref={containerRef}>
      {/* Header with chromosome selector */}
      {showTopControls && (
        <div className="heatmap-header">
          <div className="chromosome-selector">
            <label htmlFor="bed-chromosome-select">Chromosome:</label>
            <div className="select-wrapper">
              <select
                id="bed-chromosome-select"
                value={selectedChromosome}
                onChange={handleChromosomeChange}
                disabled={isLoading}
              >
                {summary.chromosomes.map(chr => (
                  <option key={chr} value={chr}>
                    {chr} ({formatNumber(summary.chromosome_counts[chr] || 0)} regions)
                  </option>
                ))}
              </select>
              <Icon path={mdiChevronDown} size={0.8} className="select-icon" />
            </div>
          </div>

          {matrixData && !isLoading && (
            <PositionSearch
              onSearch={handlePositionSearch}
              inputId="bed-position-search-input"
            />
          )}

          {matrixData && !isLoading && (
            <div className="matrix-info">
              <span className="info-item">
                <strong>{matrixData.num_parents}</strong> parents
              </span>
              <span className="info-divider">&times;</span>
              <span className="info-item">
                <strong>{formatNumber(matrixData.num_regions)}</strong> regions
              </span>
              {positionRange && (
                <span className="info-item position-range">
                  ({formatNumber(positionRange[0])} - {formatNumber(positionRange[1])} bp)
                </span>
              )}
            </div>
          )}
        </div>
      )}

      {/* Loading state */}
      {isLoading && (
        <div className="heatmap-loading">
          <div className="spinner"></div>
          <span>Loading {selectedChromosome}...</span>
          {loadProgress && (
            <div className="progress-container">
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${Math.min(loadProgress.percent, 100)}%` }}></div>
              </div>
              <div className="progress-stats">
                <span className="progress-percent">{loadProgress.percent.toFixed(1)}%</span>
                <span className="progress-rows">{formatNumber(loadProgress.rows_processed)} regions processed</span>
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
          {topControlsVisible && (
            <div className="bed-heatmap-controls-row">
              <HeatmapControls
                zoomLevel={zoomLevel}
                onZoomChange={setZoomLevel}
                cellWidthMultiplier={cellWidthMultiplier}
                onCellWidthChange={setCellWidthMultiplier}
                cellHeightMultiplier={cellHeightMultiplier}
                onCellHeightChange={setCellHeightMultiplier}
                showGridLines={showGridLines}
                onToggleGridLines={() => setShowGridLines(prev => !prev)}
                onResetView={handleResetView}
              />
              <div className="bed-auto-fit">
                <button
                  className="control-button auto-fit-button"
                  onClick={handleAutoFitHeight}
                  title="Fit heatmap height to view"
                >
                  <Icon path={mdiArrowExpandVertical} size={0.5} />
                  <span className="button-label">Fit Height</span>
                </button>
              </div>
            </div>
          )}

          <div className="heatmap-canvas-wrapper" ref={wrapperRef}>
            <BEDHeatmapCanvas
              ref={canvasRef}
              matrix={matrixData.matrix}
              regions={matrixData.regions}
              parentNames={matrixData.parent_names}
              parent1Path={matrixData.parent1_path}
              parent2Path={matrixData.parent2_path}
              showParent1Path={showParent1Path}
              showParent2Path={showParent2Path}
              zoomLevel={zoomLevel}
              scrollOffset={scrollOffset}
              onScrollChange={setScrollOffset}
              onZoomChange={setZoomLevel}
              onVisibleRangeChange={handleVisibleRangeChange}
              cellWidthMultiplier={cellWidthMultiplier}
              onCellWidthChange={setCellWidthMultiplier}
              cellHeightMultiplier={cellHeightMultiplier}
              showGridLines={showGridLines}
            />
          </div>

          <div className="heatmap-bottom-bar">
            <div className="bottom-bar-top-row">
              <div className="bottom-bar-section">
                <span className="section-label">Auto-scroll</span>
                <button
                  className={`section-button ${isAutoScrolling ? 'active' : ''}`}
                  onClick={() => setIsAutoScrolling(prev => !prev)}
                  title={isAutoScrolling ? 'Pause auto-scroll' : 'Start auto-scroll'}
                >
                  <Icon path={isAutoScrolling ? mdiPause : mdiPlay} size={0.8} />
                </button>
                <div className="speed-slider-container">
                  <input
                    type="range" min="0.1" max="2" step="0.1"
                    value={autoScrollSpeed}
                    onChange={(e) => setAutoScrollSpeed(parseFloat(e.target.value))}
                    className="speed-slider"
                    title="Scroll speed"
                  />
                  <span className="speed-value">{autoScrollSpeed.toFixed(1)}x</span>
                </div>
              </div>

              {visibleRange && matrixData.regions.length > 0 && (
                <div className="visible-range-indicator">
                  <span className="visible-range-label">Viewing:</span>
                  <span className="visible-range-value">
                    {formatNumber(matrixData.regions[visibleRange.startRegionIdx]?.start ?? 0)} - {formatNumber(matrixData.regions[visibleRange.endRegionIdx]?.end ?? 0)} bp
                  </span>
                </div>
              )}

              <div className="bottom-bar-right-group">
                <div className="bottom-bar-section">
                  <span className="section-label">Export</span>
                  <button
                    className="section-button with-label"
                    onClick={() => setShowExportModal(true)}
                    title="Export heatmap as PNG"
                  >
                    <Icon path={mdiDownload} size={0.7} />
                    <span>PNG</span>
                  </button>
                </div>
                <button
                  type="button"
                  className="heatmap-controls-toggle bottom-bar-toggle"
                  onClick={() => setTopControlsVisible(prev => !prev)}
                  title={topControlsVisible ? 'Hide header and tool bar' : 'Show header and tool bar'}
                >
                  <Icon path={topControlsVisible ? mdiEyeOff : mdiEye} size={0.85} />
                </button>
              </div>
            </div>

            {/* Position scale */}
            {positionRange && (
              <div className="position-scale">
                <span className="position-label">{formatNumber(positionRange[0])} bp</span>
                <span className="position-label">{formatNumber(Math.round((positionRange[0] + positionRange[1]) / 2))} bp</span>
                <span className="position-label">{formatNumber(positionRange[1])} bp</span>
              </div>
            )}

            {/* Tickmarks */}
            <div className="slider-tickmarks">
              {[0, 25, 50, 75, 100].map((percent) => (
                <div
                  key={percent}
                  className={`slider-tick ${percent === 0 || percent === 50 || percent === 100 ? 'major' : 'minor'}`}
                  style={{ left: `${percent}%` }}
                />
              ))}
            </div>

            {/* Scroll slider */}
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

            {/* Legend */}
            <div className="bottom-bar-info">
              <div className="legend-item">
                <span className="legend-color bed-legend-parent1"></span>
                <span className="legend-label">Parent 1</span>
              </div>
              <div className="legend-item">
                <span className="legend-color bed-legend-parent2"></span>
                <span className="legend-label">Parent 2</span>
              </div>
              <div className="legend-item">
                <span className="legend-color bed-legend-both"></span>
                <span className="legend-label">Both</span>
              </div>
              <div className="bed-path-toggle-inline">
                <span className="path-toggle-group-label">Paths</span>
                <div className="path-toggle-group">
                  <button
                    className={`control-button path-toggle-button ${showParent1Path ? 'active parent1' : ''}`}
                    onClick={() => setShowParent1Path(prev => !prev)}
                    title={showParent1Path ? 'Hide Parent 1 path' : 'Show Parent 1 path'}
                  >
                    <Icon path={showParent1Path ? mdiEye : mdiEyeOff} size={0.5} />
                    <span className="button-label">P1</span>
                  </button>
                  <button
                    className={`control-button path-toggle-button ${showParent2Path ? 'active parent2' : ''}`}
                    onClick={() => setShowParent2Path(prev => !prev)}
                    title={showParent2Path ? 'Hide Parent 2 path' : 'Show Parent 2 path'}
                  >
                    <Icon path={showParent2Path ? mdiEye : mdiEyeOff} size={0.5} />
                    <span className="button-label">P2</span>
                  </button>
                </div>
              </div>
              <div className="shortcuts-trigger-wrapper">
                <button
                  className="control-button shortcuts-trigger"
                  onClick={() => setShowShortcuts(prev => !prev)}
                  title="Keyboard & mouse shortcuts"
                >
                  <Icon path={mdiKeyboard} size={0.65} />
                </button>
                {showShortcuts && (
                  <div className="shortcuts-popup">
                    <div className="shortcuts-popup-header">
                      <span>Shortcuts</span>
                      <button className="shortcuts-popup-close" onClick={() => setShowShortcuts(false)}>&times;</button>
                    </div>
                    <div className="shortcuts-popup-body">
                      <div className="shortcut-row"><kbd>Scroll</kbd><span>Vertical pan</span></div>
                      <div className="shortcut-row"><kbd>Shift + Scroll</kbd><span>Horizontal pan</span></div>
                      <div className="shortcut-row"><kbd>Ctrl + Scroll</kbd><span>Zoom</span></div>
                      <div className="shortcut-row"><kbd>Ctrl + Shift + Scroll</kbd><span>Column width</span></div>
                      <div className="shortcut-row"><kbd>Space</kbd><span>Play / Pause auto-scroll</span></div>
                      <div className="shortcut-row"><kbd>Click + Drag</kbd><span>Pan horizontally</span></div>
                    </div>
                  </div>
                )}
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
          <p>Choose a chromosome from the dropdown to visualize parent assignments</p>
        </div>
      )}

      {showExportModal && matrixData && selectedChromosome && (() => {
        const fileId = getFileName(filePath);
        const firstRegion = matrixData.regions[0];
        const lastRegion = matrixData.regions[matrixData.regions.length - 1];
        const fullStart = firstRegion?.start ?? 0;
        const fullEnd = lastRegion?.end ?? 0;

        const visStart = visibleRange
          ? (matrixData.regions[visibleRange.startRegionIdx]?.start ?? fullStart)
          : fullStart;
        const visEnd = visibleRange
          ? (matrixData.regions[visibleRange.endRegionIdx]?.end ?? fullEnd)
          : fullEnd;

        const defaultTitle = `${fileId} | ${selectedChromosome} | ${visStart.toLocaleString()} - ${visEnd.toLocaleString()} bp`;

        const overlayInfos: PathOverlayInfo[] = [];
        if (matrixData.parent1_path?.length > 0) {
          overlayInfos.push({ label: 'Parent 1 Path', visible: showParent1Path });
        }
        if (matrixData.parent2_path?.length > 0) {
          overlayInfos.push({ label: 'Parent 2 Path', visible: showParent2Path });
        }

        return (
          <ExportModal
            defaultTitle={defaultTitle}
            defaultWidth={containerWidth}
            defaultHeight={400}
            visibleStartPos={visStart}
            visibleEndPos={visEnd}
            fullRangeStart={fullStart}
            fullRangeEnd={fullEnd}
            pathOverlays={overlayInfos}
            onExport={handleExportWithSettings}
            onClose={() => setShowExportModal(false)}
          />
        );
      })()}
    </div>
  );
};

export default BEDHeatmapViewer;
