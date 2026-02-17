import React, { useRef, useEffect, useCallback, useState, useMemo, forwardRef, useImperativeHandle } from 'react';
import { save } from '@tauri-apps/plugin-dialog';
import { writeFile } from '@tauri-apps/plugin-fs';

/** Visible range information reported by the canvas */
export interface BEDVisibleRange {
  startRegionIdx: number;
  endRegionIdx: number;
  startCol: number;
  endCol: number;
}

/** A genomic region from the BED file */
export interface BEDRegion {
  start: number;
  end: number;
}

/** Export options for PNG generation */
export interface BEDExportOptions {
  fileId: string;
  chromosome: string;
}

/** Methods exposed via ref */
export interface BEDHeatmapCanvasHandle {
  exportToPng: (options: BEDExportOptions) => Promise<void>;
}

interface BEDHeatmapCanvasProps {
  /** Matrix data: rows = parents, columns = regions. Values: 0=empty, 1=parent1, 2=parent2, 3=both */
  matrix: number[][];
  /** Region labels for x-axis */
  regions: BEDRegion[];
  /** Parent names for y-axis */
  parentNames: string[];
  /** For each column, the row index of parent1 */
  parent1Path: number[];
  /** For each column, the row index of parent2 */
  parent2Path: number[];
  /** Whether to show path lines */
  showPaths: boolean;
  /** Current zoom level (1 = 100%) */
  zoomLevel: number;
  /** Horizontal scroll offset (0-1) */
  scrollOffset: number;
  onScrollChange: (offset: number) => void;
  onZoomChange?: (zoom: number) => void;
  onVisibleRangeChange?: (range: BEDVisibleRange) => void;
  baseCellSize?: number;
  cellWidthMultiplier?: number;
  cellHeightMultiplier?: number;
  showGridLines?: boolean;
}

// Color constants matching FocusChart.ts
const COLOR_PARENT1 = '#FF6B35'; // Orange
const COLOR_PARENT2 = '#2E86AB'; // Blue
const COLOR_BOTH = '#9D4EDD';    // Purple
const COLOR_EMPTY = '#E8E8E8';   // Light grey

interface TooltipContentProps {
  mousePos: { x: number; y: number };
  containerSize: { width: number; height: number };
  parentNames: string[];
  regions: BEDRegion[];
  hoveredCell: { row: number; col: number; value: number };
}

const TooltipContent: React.FC<TooltipContentProps> = ({
  mousePos,
  containerSize,
  parentNames,
  regions,
  hoveredCell,
}) => {
  const tooltipRef = useRef<HTMLDivElement>(null);
  const [tooltipSize, setTooltipSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    if (tooltipRef.current) {
      const rect = tooltipRef.current.getBoundingClientRect();
      setTooltipSize({ width: rect.width, height: rect.height });
    }
  }, [hoveredCell]);

  const tooltipPosition = useMemo(() => {
    const offset = 15;
    const padding = 8;
    let left = mousePos.x + offset;
    let top = mousePos.y - 10;

    if (tooltipSize.width > 0 && left + tooltipSize.width + padding > containerSize.width) {
      left = mousePos.x - tooltipSize.width - offset;
    }
    if (tooltipSize.height > 0 && top + tooltipSize.height + padding > containerSize.height) {
      top = mousePos.y - tooltipSize.height - offset;
    }
    if (left < padding) left = padding;
    if (top < padding) top = padding;
    return { left, top };
  }, [mousePos, tooltipSize, containerSize]);

  const region = regions[hoveredCell.col];
  const parentName = parentNames[hoveredCell.row];
  const roleLabel = hoveredCell.value === 1 ? 'Parent 1' :
                    hoveredCell.value === 2 ? 'Parent 2' :
                    hoveredCell.value === 3 ? 'Both Parents' : 'Empty';

  return (
    <div
      ref={tooltipRef}
      className="heatmap-tooltip"
      style={{
        position: 'absolute',
        left: tooltipPosition.left,
        top: tooltipPosition.top,
        background: 'var(--md-sys-color-inverse-surface, #322f35)',
        color: 'var(--md-sys-color-inverse-on-surface, #f5eff7)',
        padding: '6px 10px',
        borderRadius: '4px',
        fontSize: '12px',
        fontFamily: '"Roboto", sans-serif',
        pointerEvents: 'none',
        zIndex: 100,
        boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
        whiteSpace: 'nowrap',
      }}
    >
      <div><strong>{parentName}</strong></div>
      {region && (
        <div>Region: {region.start.toLocaleString()} - {region.end.toLocaleString()} bp</div>
      )}
      <div>Role: {roleLabel}</div>
    </div>
  );
};

const BEDHeatmapCanvas = forwardRef<BEDHeatmapCanvasHandle, BEDHeatmapCanvasProps>(({
  matrix,
  regions,
  parentNames,
  parent1Path,
  parent2Path,
  showPaths,
  zoomLevel,
  scrollOffset,
  onScrollChange,
  onZoomChange,
  onVisibleRangeChange,
  baseCellSize = 12,
  cellWidthMultiplier = 1,
  cellHeightMultiplier = 1,
  showGridLines = true,
}, ref) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerSize, setContainerSize] = useState({ width: 800, height: 400 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStartX, setDragStartX] = useState(0);
  const [dragStartOffset, setDragStartOffset] = useState(0);
  const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number; value: number } | null>(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });

  const LABEL_MARGIN_LEFT = 100;
  const LABEL_MARGIN_TOP = 10;
  const PADDING = 10;

  const cellHeight = baseCellSize * zoomLevel * cellHeightMultiplier;
  const cellWidth = baseCellSize * zoomLevel * cellWidthMultiplier;
  const numRows = matrix.length;
  const numCols = regions.length;
  const totalMatrixWidth = numCols * cellWidth;
  const totalMatrixHeight = numRows * cellHeight;
  const viewportWidth = containerSize.width - LABEL_MARGIN_LEFT - PADDING * 2;

  const maxScrollOffset = Math.max(0, totalMatrixWidth - viewportWidth);
  const scrollX = scrollOffset * maxScrollOffset;
  const startCol = Math.floor(scrollX / cellWidth);
  const endCol = Math.min(numCols, Math.ceil((scrollX + viewportWidth) / cellWidth) + 1);

  // Monitor container size
  useEffect(() => {
    const updateSize = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        const requiredHeight = totalMatrixHeight + LABEL_MARGIN_TOP + PADDING * 2;
        setContainerSize({
          width: rect.width || 800,
          height: requiredHeight,
        });
      }
    };
    updateSize();
    const observer = new ResizeObserver(updateSize);
    if (containerRef.current) observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, [totalMatrixHeight]);

  // Report visible range changes
  useEffect(() => {
    if (onVisibleRangeChange && regions.length > 0) {
      const allDataVisible = maxScrollOffset === 0;
      if (allDataVisible) {
        onVisibleRangeChange({
          startRegionIdx: 0,
          endRegionIdx: regions.length - 1,
          startCol: 0,
          endCol: regions.length - 1,
        });
      } else {
        const lastVisibleCol = Math.min(endCol - 2, numCols - 2);
        onVisibleRangeChange({
          startRegionIdx: startCol,
          endRegionIdx: lastVisibleCol,
          startCol,
          endCol: lastVisibleCol,
        });
      }
    }
  }, [onVisibleRangeChange, startCol, endCol, numCols, regions, maxScrollOffset]);

  // Get cell color based on value
  const getCellColor = useCallback((value: number): string => {
    switch (value) {
      case 1: return COLOR_PARENT1;
      case 2: return COLOR_PARENT2;
      case 3: return COLOR_BOTH;
      default: return COLOR_EMPTY;
    }
  }, []);

  // Draw a parent path on the canvas
  const drawParentPath = useCallback((
    ctx: CanvasRenderingContext2D,
    pathData: number[],
    color: string,
    dashPattern: number[],
  ) => {
    if (pathData.length < 2) return;

    // Collect visible points
    const points: { x: number; y: number }[] = [];
    for (let col = startCol; col < endCol && col < pathData.length; col++) {
      const rowIdx = pathData[col];
      const x = LABEL_MARGIN_LEFT + PADDING + (col - startCol) * cellWidth + cellWidth / 2;
      const y = LABEL_MARGIN_TOP + PADDING + rowIdx * cellHeight + cellHeight / 2;
      points.push({ x, y });
    }

    if (points.length < 2) return;

    // Draw connecting line
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.setLineDash(dashPattern);
    ctx.globalAlpha = 0.85;
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i].x, points[i].y);
    }
    ctx.stroke();

    // Draw circles at each node
    ctx.setLineDash([]);
    ctx.globalAlpha = 0.9;
    for (const pt of points) {
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
    ctx.restore();
  }, [startCol, endCol, cellWidth, cellHeight, LABEL_MARGIN_LEFT, LABEL_MARGIN_TOP, PADDING]);

  // Export to PNG
  useImperativeHandle(ref, () => ({
    exportToPng: async (options: BEDExportOptions) => {
      try {
        const { fileId, chromosome } = options;
        const startRegion = regions[startCol] ?? regions[0];
        const lastVisibleCol = Math.min(endCol - 1, numCols - 1);
        const endRegion = regions[lastVisibleCol] ?? regions[regions.length - 1];
        const title = `${fileId} | ${chromosome} | ${startRegion.start.toLocaleString()} - ${endRegion.end.toLocaleString()} bp`;

        const TITLE_HEIGHT = 50;
        const LEGEND_HEIGHT = 50;
        const exportWidth = containerSize.width;
        const exportHeight = containerSize.height + TITLE_HEIGHT + LEGEND_HEIGHT;

        const exportCanvas = document.createElement('canvas');
        const exportScale = 3;
        exportCanvas.width = exportWidth * exportScale;
        exportCanvas.height = exportHeight * exportScale;

        const ctx = exportCanvas.getContext('2d');
        if (!ctx) return;
        ctx.scale(exportScale, exportScale);

        const surfaceColor = '#fef7ff';
        const onSurfaceColor = '#1d1b20';
        const outlineVariantColor = '#cac4d0';

        ctx.fillStyle = surfaceColor;
        ctx.fillRect(0, 0, exportWidth, exportHeight);

        // Title
        ctx.fillStyle = onSurfaceColor;
        ctx.font = 'bold 16px "Roboto", sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(title, exportWidth / 2, TITLE_HEIGHT / 2);

        ctx.save();
        ctx.translate(0, TITLE_HEIGHT);

        // Y-axis labels
        ctx.fillStyle = onSurfaceColor;
        ctx.font = `${Math.min(11, cellHeight * 0.8)}px "Roboto", sans-serif`;
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        for (let row = 0; row < numRows; row++) {
          const y = LABEL_MARGIN_TOP + PADDING + row * cellHeight + cellHeight / 2;
          if (y < containerSize.height) {
            const label = parentNames[row] || `Parent ${row}`;
            const truncated = label.length > 12 ? label.substring(0, 10) + '...' : label;
            ctx.fillText(truncated, LABEL_MARGIN_LEFT - 8, y);
          }
        }

        // White background behind cells
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(LABEL_MARGIN_LEFT + PADDING, LABEL_MARGIN_TOP + PADDING, viewportWidth, totalMatrixHeight);

        // Draw cells
        for (let row = 0; row < numRows; row++) {
          const y = LABEL_MARGIN_TOP + PADDING + row * cellHeight;
          if (y > containerSize.height) break;
          for (let col = startCol; col < endCol; col++) {
            const x = LABEL_MARGIN_LEFT + PADDING + (col - startCol) * cellWidth;
            if (x > containerSize.width - PADDING) break;
            const value = matrix[row]?.[col] ?? 0;
            ctx.fillStyle = getCellColor(value);
            ctx.fillRect(x, y, cellWidth - (showGridLines ? 1 : 0), cellHeight - (showGridLines ? 1 : 0));
          }
        }

        // Grid lines
        if (showGridLines && Math.min(cellWidth, cellHeight) >= 4) {
          ctx.strokeStyle = outlineVariantColor;
          ctx.lineWidth = 0.5;
          for (let col = startCol; col <= endCol; col++) {
            const x = LABEL_MARGIN_LEFT + PADDING + (col - startCol) * cellWidth;
            if (x <= containerSize.width - PADDING) {
              ctx.beginPath();
              ctx.moveTo(x, LABEL_MARGIN_TOP + PADDING);
              ctx.lineTo(x, LABEL_MARGIN_TOP + PADDING + totalMatrixHeight);
              ctx.stroke();
            }
          }
          for (let row = 0; row <= numRows; row++) {
            const y = LABEL_MARGIN_TOP + PADDING + row * cellHeight;
            if (y <= containerSize.height) {
              ctx.beginPath();
              ctx.moveTo(LABEL_MARGIN_LEFT + PADDING, y);
              ctx.lineTo(LABEL_MARGIN_LEFT + PADDING + viewportWidth, y);
              ctx.stroke();
            }
          }
        }

        // Paths
        if (showPaths) {
          drawParentPath(ctx, parent1Path, COLOR_PARENT1, [8, 4]);
          drawParentPath(ctx, parent2Path, COLOR_PARENT2, [4, 8]);
        }

        ctx.restore();

        // Legend
        const legendY = TITLE_HEIGHT + containerSize.height + LEGEND_HEIGHT / 2;
        ctx.font = '12px "Roboto", sans-serif';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        const legendStartX = exportWidth / 2 - 180;

        // Parent 1
        ctx.fillStyle = COLOR_PARENT1;
        ctx.fillRect(legendStartX, legendY - 8, 16, 16);
        ctx.fillStyle = onSurfaceColor;
        ctx.fillText('Parent 1', legendStartX + 24, legendY);

        // Parent 2
        ctx.fillStyle = COLOR_PARENT2;
        ctx.fillRect(legendStartX + 110, legendY - 8, 16, 16);
        ctx.fillStyle = onSurfaceColor;
        ctx.fillText('Parent 2', legendStartX + 134, legendY);

        // Both
        ctx.fillStyle = COLOR_BOTH;
        ctx.fillRect(legendStartX + 220, legendY - 8, 16, 16);
        ctx.fillStyle = onSurfaceColor;
        ctx.fillText('Both', legendStartX + 244, legendY);

        // Empty
        ctx.fillStyle = COLOR_EMPTY;
        ctx.fillRect(legendStartX + 300, legendY - 8, 16, 16);
        ctx.strokeStyle = outlineVariantColor;
        ctx.strokeRect(legendStartX + 300, legendY - 8, 16, 16);
        ctx.fillStyle = onSurfaceColor;
        ctx.fillText('Empty', legendStartX + 324, legendY);

        const sanitizedFileId = fileId.replace(/[^a-zA-Z0-9_-]/g, '_');
        const sanitizedChromosome = chromosome.replace(/[^a-zA-Z0-9_-]/g, '_');
        const suggestedFilename = `${sanitizedFileId}_${sanitizedChromosome}_heatmap.png`;

        const savePath = await save({
          title: 'Export BED Heatmap as PNG',
          defaultPath: suggestedFilename,
          filters: [{ name: 'PNG Image', extensions: ['png'] }],
        });
        if (!savePath) return;

        const blob = await new Promise<Blob | null>((resolve) => {
          exportCanvas.toBlob(resolve, 'image/png');
        });
        if (!blob) return;

        const arrayBuffer = await blob.arrayBuffer();
        const uint8Array = new Uint8Array(arrayBuffer);
        await writeFile(savePath, uint8Array);
      } catch (error) {
        console.error('Failed to export BED heatmap PNG:', error);
        throw error;
      }
    }
  }), [
    matrix, regions, parentNames, parent1Path, parent2Path, showPaths,
    containerSize, cellWidth, cellHeight, startCol, endCol, numRows, numCols,
    totalMatrixHeight, viewportWidth, showGridLines, getCellColor, drawParentPath,
  ]);

  // Draw the heatmap
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = containerSize.width * dpr;
    canvas.height = containerSize.height * dpr;
    ctx.scale(dpr, dpr);

    const computedStyle = getComputedStyle(document.documentElement);
    const surfaceColor = computedStyle.getPropertyValue('--md-sys-color-surface').trim() || '#fef7ff';
    const onSurfaceColor = computedStyle.getPropertyValue('--md-sys-color-on-surface').trim() || '#1d1b20';
    const outlineVariantColor = computedStyle.getPropertyValue('--md-sys-color-outline-variant').trim() || '#cac4d0';

    // Clear
    ctx.fillStyle = surfaceColor;
    ctx.fillRect(0, 0, containerSize.width, containerSize.height);

    // Y-axis labels
    ctx.fillStyle = onSurfaceColor;
    ctx.font = `${Math.min(11, cellHeight * 0.8)}px "Roboto", sans-serif`;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let row = 0; row < numRows; row++) {
      const y = LABEL_MARGIN_TOP + PADDING + row * cellHeight + cellHeight / 2;
      if (y < containerSize.height) {
        const label = parentNames[row] || `Parent ${row}`;
        const truncated = label.length > 12 ? label.substring(0, 10) + '...' : label;
        ctx.fillText(truncated, LABEL_MARGIN_LEFT - 8, y);
      }
    }

    // White background behind cells
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(LABEL_MARGIN_LEFT + PADDING, LABEL_MARGIN_TOP + PADDING, viewportWidth, totalMatrixHeight);

    // Draw cells
    for (let row = 0; row < numRows; row++) {
      const y = LABEL_MARGIN_TOP + PADDING + row * cellHeight;
      if (y > containerSize.height) break;
      for (let col = startCol; col < endCol; col++) {
        const x = LABEL_MARGIN_LEFT + PADDING + (col - startCol) * cellWidth;
        if (x > containerSize.width - PADDING) break;
        const value = matrix[row]?.[col] ?? 0;
        ctx.fillStyle = getCellColor(value);
        ctx.fillRect(x, y, cellWidth - (showGridLines ? 1 : 0), cellHeight - (showGridLines ? 1 : 0));
      }
    }

    // Grid lines
    if (showGridLines && Math.min(cellWidth, cellHeight) >= 4) {
      ctx.strokeStyle = outlineVariantColor;
      ctx.lineWidth = 0.5;
      for (let col = startCol; col <= endCol; col++) {
        const x = LABEL_MARGIN_LEFT + PADDING + (col - startCol) * cellWidth;
        if (x <= containerSize.width - PADDING) {
          ctx.beginPath();
          ctx.moveTo(x, LABEL_MARGIN_TOP + PADDING);
          ctx.lineTo(x, LABEL_MARGIN_TOP + PADDING + totalMatrixHeight);
          ctx.stroke();
        }
      }
      for (let row = 0; row <= numRows; row++) {
        const y = LABEL_MARGIN_TOP + PADDING + row * cellHeight;
        if (y <= containerSize.height) {
          ctx.beginPath();
          ctx.moveTo(LABEL_MARGIN_LEFT + PADDING, y);
          ctx.lineTo(LABEL_MARGIN_LEFT + PADDING + viewportWidth, y);
          ctx.stroke();
        }
      }
    }

    // Draw parent paths
    if (showPaths) {
      drawParentPath(ctx, parent1Path, COLOR_PARENT1, [8, 4]);
      drawParentPath(ctx, parent2Path, COLOR_PARENT2, [4, 8]);
    }

    // Hover highlight
    if (hoveredCell) {
      const { row, col } = hoveredCell;
      if (col >= startCol && col < endCol) {
        const x = LABEL_MARGIN_LEFT + PADDING + (col - startCol) * cellWidth;
        const y = LABEL_MARGIN_TOP + PADDING + row * cellHeight;
        const tertiaryColor = computedStyle.getPropertyValue('--md-sys-color-tertiary').trim() || '#7d5260';
        ctx.strokeStyle = tertiaryColor;
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, cellWidth, cellHeight);
      }
    }
  }, [matrix, regions, parentNames, parent1Path, parent2Path, showPaths,
      containerSize, cellWidth, cellHeight, startCol, endCol, numRows, numCols,
      totalMatrixHeight, viewportWidth, showGridLines, getCellColor, hoveredCell, drawParentPath]);

  // Mouse handlers
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button === 0) {
      setIsDragging(true);
      setDragStartX(e.clientX);
      setDragStartOffset(scrollOffset);
    }
  }, [scrollOffset]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setMousePos({ x, y });

    if (isDragging) {
      const deltaX = e.clientX - dragStartX;
      const deltaOffset = -deltaX / (maxScrollOffset || 1);
      const newOffset = Math.max(0, Math.min(1, dragStartOffset + deltaOffset));
      onScrollChange(newOffset);
    }

    const cellX = x - LABEL_MARGIN_LEFT - PADDING;
    const cellY = y - LABEL_MARGIN_TOP - PADDING;
    if (cellX >= 0 && cellY >= 0) {
      const col = startCol + Math.floor(cellX / cellWidth);
      const row = Math.floor(cellY / cellHeight);
      if (row >= 0 && row < numRows && col >= 0 && col < numCols) {
        const value = matrix[row]?.[col] ?? 0;
        setHoveredCell({ row, col, value });
      } else {
        setHoveredCell(null);
      }
    } else {
      setHoveredCell(null);
    }
  }, [isDragging, dragStartX, dragStartOffset, maxScrollOffset, onScrollChange,
      cellWidth, cellHeight, startCol, numRows, numCols, matrix]);

  const handleMouseUp = useCallback(() => { setIsDragging(false); }, []);
  const handleMouseLeave = useCallback(() => { setIsDragging(false); setHoveredCell(null); }, []);

  useEffect(() => {
    const handleGlobalMouseUp = () => setIsDragging(false);
    window.addEventListener('mouseup', handleGlobalMouseUp);
    return () => window.removeEventListener('mouseup', handleGlobalMouseUp);
  }, []);

  // Native wheel events
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const handleNativeWheel = (e: WheelEvent) => {
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        if (onZoomChange) {
          const delta = e.deltaY > 0 ? -0.1 : 0.1;
          onZoomChange(Math.max(0.25, Math.min(4, zoomLevel + delta)));
        }
      } else if (e.shiftKey) {
        e.preventDefault();
        const delta = e.deltaX !== 0 ? e.deltaX : e.deltaY;
        const scrollDelta = delta / (maxScrollOffset || 1);
        const newOffset = Math.max(0, Math.min(1, scrollOffset + scrollDelta * 0.1));
        onScrollChange(newOffset);
      }
    };
    canvas.addEventListener('wheel', handleNativeWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', handleNativeWheel);
  }, [scrollOffset, maxScrollOffset, zoomLevel, onScrollChange, onZoomChange]);

  const requiredHeight = totalMatrixHeight + LABEL_MARGIN_TOP + PADDING * 2;

  return (
    <div
      ref={containerRef}
      className="heatmap-canvas-container"
      style={{
        width: '100%',
        minHeight: Math.max(300, requiredHeight),
        height: requiredHeight,
        position: 'relative',
        cursor: isDragging ? 'grabbing' : 'grab',
      }}
    >
      <canvas
        ref={canvasRef}
        style={{ width: containerSize.width, height: containerSize.height }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
      />
      {hoveredCell && (
        <TooltipContent
          mousePos={mousePos}
          containerSize={containerSize}
          parentNames={parentNames}
          regions={regions}
          hoveredCell={hoveredCell}
        />
      )}
    </div>
  );
});

BEDHeatmapCanvas.displayName = 'BEDHeatmapCanvas';

export default BEDHeatmapCanvas;
