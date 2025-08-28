import React, { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import { D3MatrixProps } from "./types";
import { createTooltip } from "./tooltip";
import { renderFocusChart } from "./FocusChart";
import { calculateResponsiveDimensions } from "./utils";

const D3Matrix: React.FC<D3MatrixProps> = ({
  data,
  rowLabels,
  colLabels,
  highlightData,
  cellSize = 15,
  margin = { top: 100, right: 5, bottom: 5, left: 80 },
  maxVisibleRows = 20,
  maxVisibleCols = 40,
}) => {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [containerSize, setContainerSize] = useState({ width: 800, height: 600 });

  // Create tooltip container
  useEffect(() => {
    createTooltip();
  }, []);

  // Monitor container size
  useEffect(() => {
    const updateContainerSize = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        setContainerSize({
          width: rect.width || 800,
          height: rect.height || 600,
        });
      }
    };

    updateContainerSize();
    
    const resizeObserver = new ResizeObserver(updateContainerSize);
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }

    return () => resizeObserver.disconnect();
  }, []);

  const { totalWidth, totalHeight, cellSize: responsiveCellSize } = calculateResponsiveDimensions(
    rowLabels.length,
    colLabels.length,
    containerSize.width,
    containerSize.height,
    margin,
    25, // minCellSize
    cellSize || 80 // maxCellSize - use provided cellSize as maximum or default to 80
  );

  // Draw main (focus) chart when data changes
  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    
    // Since we're now getting pre-filtered data, use full intervals
    const fullXInterval = { start: 0, end: colLabels.length };
    const fullYInterval = { start: 0, end: rowLabels.length };
    
    renderFocusChart(
      svg,
      data,
      rowLabels,
      colLabels,
      fullXInterval,
      fullYInterval,
      responsiveCellSize,
      margin,
      rowLabels.length,
      colLabels.length,
      highlightData
    );
  }, [data, rowLabels, colLabels, responsiveCellSize, margin, maxVisibleRows, maxVisibleCols, highlightData, containerSize, cellSize]);

  return (
    <div ref={containerRef} style={{ width: '100%', height: '100%', minHeight: '400px' }}>
      <svg ref={svgRef} width={totalWidth} height={totalHeight} />
    </div>
  );
};

export default D3Matrix;
