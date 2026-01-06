import React, { useRef, useEffect, useCallback, useState } from 'react';

interface HeatmapCanvasProps {
  /** Matrix data: rows = gametes, columns = positions. Values are counts (0 = no data) */
  matrix: number[][];
  /** Position labels for x-axis (binned positions) */
  positions: number[];
  /** Gamete names for y-axis */
  gameteNames: string[];
  /** Current zoom level (1 = 100%) */
  zoomLevel: number;
  /** Horizontal scroll offset (0-1 range, percentage of total width) */
  scrollOffset: number;
  /** Callback when scroll position changes via drag/wheel */
  onScrollChange: (offset: number) => void;
  /** Callback when zoom changes via wheel */
  onZoomChange?: (zoom: number) => void;
  /** Cell height at zoom level 1 */
  baseCellSize?: number;
  /** Cell width multiplier (relative to baseCellSize, default 1) */
  cellWidthMultiplier?: number;
  /** Whether to show grid lines */
  showGridLines?: boolean;
  /** Color scheme: 'binary' (gray/white) or 'intensity' (color gradient) */
  colorScheme?: 'binary' | 'intensity';
}

const HeatmapCanvas: React.FC<HeatmapCanvasProps> = ({
  matrix,
  positions,
  gameteNames,
  zoomLevel,
  scrollOffset,
  onScrollChange,
  onZoomChange,
  baseCellSize = 12,
  cellWidthMultiplier = 1,
  showGridLines = true,
  colorScheme = 'binary',
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerSize, setContainerSize] = useState({ width: 800, height: 400 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStartX, setDragStartX] = useState(0);
  const [dragStartOffset, setDragStartOffset] = useState(0);
  const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number; value: number } | null>(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });

  // Layout constants
  const LABEL_MARGIN_LEFT = 100; // Space for gamete labels
  const LABEL_MARGIN_TOP = 60;   // Space for position labels
  const PADDING = 10;

  // Calculate dimensions - separate width and height for cells
  const cellHeight = baseCellSize * zoomLevel;
  const cellWidth = baseCellSize * zoomLevel * cellWidthMultiplier;
  const numRows = matrix.length;
  const numCols = positions.length;
  const totalMatrixWidth = numCols * cellWidth;
  const totalMatrixHeight = numRows * cellHeight;
  const viewportWidth = containerSize.width - LABEL_MARGIN_LEFT - PADDING * 2;

  // Calculate visible range based on scroll offset
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
    if (containerRef.current) {
      observer.observe(containerRef.current);
    }
    return () => observer.disconnect();
  }, [totalMatrixHeight]);

  // Get color for cell value - primaryColor is passed in since CSS vars don't work in Canvas
  const getCellColor = useCallback((value: number, maxValue: number, primaryColor: string): string => {
    if (value === 0) {
      return '#ffffff'; // White for no data
    }
    
    if (colorScheme === 'binary') {
      return primaryColor;
    }
    
    // Intensity-based coloring - parse primary color for gradient
    // Default to purple if parsing fails
    const intensity = Math.min(value / maxValue, 1);
    const r = Math.round(103 + (255 - 103) * (1 - intensity));
    const g = Math.round(80 + (255 - 80) * (1 - intensity));
    const b = Math.round(164 + (255 - 164) * (1 - intensity));
    return `rgb(${r}, ${g}, ${b})`;
  }, [colorScheme]);

  // Get max value for intensity scaling
  const maxValue = React.useMemo(() => {
    let max = 1;
    for (const row of matrix) {
      for (const val of row) {
        if (val > max) max = val;
      }
    }
    return max;
  }, [matrix]);

  // Draw the heatmap
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size with device pixel ratio for crisp rendering
    const dpr = window.devicePixelRatio || 1;
    canvas.width = containerSize.width * dpr;
    canvas.height = containerSize.height * dpr;
    ctx.scale(dpr, dpr);

    // Get computed CSS variable colors
    const computedStyle = getComputedStyle(document.documentElement);
    const surfaceColor = computedStyle.getPropertyValue('--md-sys-color-surface').trim() || '#fef7ff';
    const onSurfaceColor = computedStyle.getPropertyValue('--md-sys-color-on-surface').trim() || '#1d1b20';
    const outlineVariantColor = computedStyle.getPropertyValue('--md-sys-color-outline-variant').trim() || '#cac4d0';
    const primaryColor = computedStyle.getPropertyValue('--md-sys-color-primary').trim() || '#6750a4';

    // Clear canvas with surface color
    ctx.fillStyle = surfaceColor;
    ctx.fillRect(0, 0, containerSize.width, containerSize.height);

    // Draw gamete labels (y-axis)
    ctx.fillStyle = onSurfaceColor;
    ctx.font = `${Math.min(11, cellHeight * 0.8)}px "Roboto", sans-serif`;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    for (let row = 0; row < numRows; row++) {
      const y = LABEL_MARGIN_TOP + PADDING + row * cellHeight + cellHeight / 2;
      if (y < containerSize.height) {
        const label = gameteNames[row] || `Gamete ${row}`;
        const truncatedLabel = label.length > 12 ? label.substring(0, 10) + '...' : label;
        ctx.fillText(truncatedLabel, LABEL_MARGIN_LEFT - 8, y);
      }
    }

    // Draw position labels (x-axis) - sparse labels for readability
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.font = `${Math.min(10, cellWidth * 0.7)}px "Roboto Mono", monospace`;
    
    const labelInterval = Math.max(1, Math.floor(50 / cellWidth)); // Show label every ~50px
    for (let col = startCol; col < endCol; col += labelInterval) {
      const x = LABEL_MARGIN_LEFT + PADDING + (col - startCol) * cellWidth + cellWidth / 2;
      if (x < containerSize.width - PADDING) {
        ctx.save();
        ctx.translate(x, LABEL_MARGIN_TOP - 8);
        ctx.rotate(-Math.PI / 4);
        const posLabel = formatPosition(positions[col]);
        ctx.fillText(posLabel, 0, 0);
        ctx.restore();
      }
    }

    // Draw cells
    for (let row = 0; row < numRows; row++) {
      const y = LABEL_MARGIN_TOP + PADDING + row * cellHeight;
      if (y > containerSize.height) break;

      for (let col = startCol; col < endCol; col++) {
        const x = LABEL_MARGIN_LEFT + PADDING + (col - startCol) * cellWidth;
        if (x > containerSize.width - PADDING) break;

        const value = matrix[row]?.[col] ?? 0;
        ctx.fillStyle = getCellColor(value, maxValue, primaryColor);
        ctx.fillRect(x, y, cellWidth - (showGridLines ? 1 : 0), cellHeight - (showGridLines ? 1 : 0));
      }
    }

    // Draw grid lines
    if (showGridLines && Math.min(cellWidth, cellHeight) >= 4) {
      ctx.strokeStyle = outlineVariantColor;
      ctx.lineWidth = 0.5;

      // Vertical lines
      for (let col = startCol; col <= endCol; col++) {
        const x = LABEL_MARGIN_LEFT + PADDING + (col - startCol) * cellWidth;
        if (x <= containerSize.width - PADDING) {
          ctx.beginPath();
          ctx.moveTo(x, LABEL_MARGIN_TOP + PADDING);
          ctx.lineTo(x, LABEL_MARGIN_TOP + PADDING + totalMatrixHeight);
          ctx.stroke();
        }
      }

      // Horizontal lines
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

    // Draw hover highlight
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

  }, [matrix, positions, gameteNames, containerSize, cellWidth, cellHeight, startCol, endCol, numRows, numCols, 
      totalMatrixHeight, viewportWidth, showGridLines, getCellColor, maxValue, hoveredCell]);

  // Format position for display
  const formatPosition = (pos: number): string => {
    if (pos >= 1000000) {
      return `${(pos / 1000000).toFixed(1)}M`;
    } else if (pos >= 1000) {
      return `${(pos / 1000).toFixed(1)}K`;
    }
    return pos.toString();
  };

  // Handle mouse down for drag scrolling
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button === 0) { // Left click
      setIsDragging(true);
      setDragStartX(e.clientX);
      setDragStartOffset(scrollOffset);
    }
  }, [scrollOffset]);

  // Handle mouse move for drag scrolling and hover
  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;

    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setMousePos({ x, y });

    // Update dragging
    if (isDragging) {
      const deltaX = e.clientX - dragStartX;
      const deltaOffset = -deltaX / (maxScrollOffset || 1);
      const newOffset = Math.max(0, Math.min(1, dragStartOffset + deltaOffset));
      onScrollChange(newOffset);
    }

    // Update hover cell
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

  // Handle mouse up
  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  // Handle mouse leave
  const handleMouseLeave = useCallback(() => {
    setIsDragging(false);
    setHoveredCell(null);
  }, []);

  // Global mouse up listener
  useEffect(() => {
    const handleGlobalMouseUp = () => setIsDragging(false);
    window.addEventListener('mouseup', handleGlobalMouseUp);
    return () => window.removeEventListener('mouseup', handleGlobalMouseUp);
  }, []);

  // Native wheel event listener for proper preventDefault support
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const handleNativeWheel = (e: WheelEvent) => {
      if (e.ctrlKey || e.metaKey) {
        // Zoom with Ctrl/Cmd + scroll
        e.preventDefault();
        if (onZoomChange) {
          const delta = e.deltaY > 0 ? -0.1 : 0.1;
          onZoomChange(Math.max(0.25, Math.min(4, zoomLevel + delta)));
        }
      } else if (e.shiftKey) {
        // Horizontal scroll with Shift + scroll
        // Note: Some browsers convert deltaY to deltaX when shift is held
        e.preventDefault();
        const delta = e.deltaX !== 0 ? e.deltaX : e.deltaY;
        const scrollDelta = delta / (maxScrollOffset || 1);
        const newOffset = Math.max(0, Math.min(1, scrollOffset + scrollDelta * 0.1));
        onScrollChange(newOffset);
      }
      // Normal scroll (no modifier) - let browser handle vertical scrolling
    };

    canvas.addEventListener('wheel', handleNativeWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', handleNativeWheel);
  }, [scrollOffset, maxScrollOffset, zoomLevel, onScrollChange, onZoomChange]);

  // Calculate the required height for the canvas
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
        style={{
          width: containerSize.width,
          height: containerSize.height,
        }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
      />
      
      {/* Tooltip */}
      {hoveredCell && (
        <div
          className="heatmap-tooltip"
          style={{
            position: 'absolute',
            left: mousePos.x + 15,
            top: mousePos.y - 10,
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
          <div><strong>{gameteNames[hoveredCell.row]}</strong></div>
          <div>Position: {positions[hoveredCell.col]?.toLocaleString()}</div>
          <div>Count: {hoveredCell.value}</div>
        </div>
      )}
    </div>
  );
};

export default HeatmapCanvas;

