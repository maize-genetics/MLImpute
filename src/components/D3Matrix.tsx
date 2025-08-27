import React, { useEffect, useRef } from "react";
import * as d3 from "d3";
import { D3MatrixProps } from "./types";
import { createTooltip } from "./tooltip";
import { renderFocusChart } from "./FocusChart";
import { calculateDimensions } from "./utils";

const D3Matrix: React.FC<D3MatrixProps> = ({
  data,
  rowLabels,
  colLabels,
  highlightData,
  cellSize = 15,
  margin = { top: 20, right: 5, bottom: 5, left: 80 },
  maxVisibleRows = 20,
  maxVisibleCols = 40,
}) => {
  const svgRef = useRef<SVGSVGElement | null>(null);

  // Create tooltip container
  useEffect(() => {
    createTooltip();
  }, []);

  const { totalWidth, totalHeight } = calculateDimensions(
    rowLabels.length,
    colLabels.length,
    cellSize,
    margin
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
      cellSize,
      margin,
      rowLabels.length,
      colLabels.length,
      highlightData
    );
  }, [data, rowLabels, colLabels, cellSize, margin, maxVisibleRows, maxVisibleCols, highlightData]);


  return <svg ref={svgRef} width={totalWidth} height={totalHeight} />;
};

export default D3Matrix;
