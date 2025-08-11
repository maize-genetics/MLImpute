import React, { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import { D3MatrixProps, Interval } from "./types";
import { createTooltip } from "./tooltip";
import { renderFocusChart } from "./FocusChart";
import { renderContextCharts } from "./ContextCharts";
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
  contextSize = 50,
}) => {
  const svgRef = useRef<SVGSVGElement | null>(null);

  // Create tooltip container
  useEffect(() => {
    createTooltip();
  }, []);

  // Focus intervals
  const [xInterval, setXInterval] = useState<Interval>({
    start: 0,
    end: Math.min(colLabels.length, maxVisibleCols),
  });
  const [yInterval, setYInterval] = useState<Interval>({
    start: 0,
    end: Math.min(rowLabels.length, maxVisibleRows),
  });

  const { innerWidth, innerHeight, totalWidth, totalHeight } = calculateDimensions(
    maxVisibleRows,
    maxVisibleCols,
    cellSize,
    margin,
    contextSize
  );

  // Draw main (focus) chart when intervals change
  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    renderFocusChart(
      svg,
      data,
      rowLabels,
      colLabels,
      xInterval,
      yInterval,
      cellSize,
      margin,
      maxVisibleRows,
      maxVisibleCols,
      highlightData
    );
  }, [data, xInterval, yInterval, rowLabels, colLabels, cellSize, margin, maxVisibleRows, maxVisibleCols, highlightData]);

  // Draw contexts & brushes once
  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    renderContextCharts(
      svg,
      rowLabels,
      colLabels,
      xInterval,
      yInterval,
      setXInterval,
      setYInterval,
      margin,
      innerWidth,
      innerHeight,
      contextSize,
      maxVisibleRows,
      maxVisibleCols
    );
  }, [rowLabels, colLabels, innerWidth, innerHeight, contextSize, margin, maxVisibleRows, maxVisibleCols]);

  return <svg ref={svgRef} width={totalWidth} height={totalHeight} />;
};

export default D3Matrix;
