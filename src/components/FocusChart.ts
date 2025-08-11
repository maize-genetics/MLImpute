import * as d3 from "d3";
import { DataPoint, Interval, HighlightData } from "./types";
import { showTooltip, moveTooltip, hideTooltip } from "./tooltip";

export const renderFocusChart = (
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  data: number[][],
  rowLabels: string[],
  colLabels: string[],
  xInterval: Interval,
  yInterval: Interval,
  cellSize: number,
  margin: { top: number; right: number; bottom: number; left: number },
  maxVisibleRows: number,
  maxVisibleCols: number,
  highlightData?: HighlightData[]
) => {
  const neon = "#39FF14";
  const innerWidth = maxVisibleCols * cellSize;
  const innerHeight = maxVisibleRows * cellSize;

  svg.selectAll(".focus").remove();
  const focusG = svg
    .append("g")
    .attr("class", "focus")
    .attr("transform", `translate(${margin.left},${margin.top})`);

  // Slice data
  const focusCols = colLabels.slice(xInterval.start, xInterval.end);
  const focusRows = rowLabels.slice(yInterval.start, yInterval.end);
  const focusData = data
    .slice(yInterval.start, yInterval.end)
    .map((r) => r.slice(xInterval.start, xInterval.end));

  // Scales
  const xScale = d3
    .scaleBand<string>()
    .domain(focusCols)
    .range([0, innerWidth])
    .padding(0);
  const yScale = d3
    .scaleBand<string>()
    .domain(focusRows)
    .range([0, innerHeight])
    .padding(0);
  const colorScale = d3
    .scaleOrdinal<string>()
    .domain(["0", "1"])
    .range(["#d3d3d3", "#4a4a4a"]);

  const nCols = focusCols.length;
  const nRows = focusRows.length;

  // Grid lines
  renderGridLines(focusG, nRows, nCols, cellSize, innerWidth, innerHeight);

  // Cells
  const flat = focusData.flatMap((row, i) =>
    row.map((val, j) => ({
      row: focusRows[i],
      col: focusCols[j],
      value: val,
    }))
  );

  // Create highlight lookup for faster checking
  const highlightSet = new Set<string>();
  if (highlightData) {
    highlightData.forEach(h => highlightSet.add(`${h.row}:${h.col}`));
  }

  const cells = focusG
    .append("g")
    .selectAll("rect")
    .data(flat)
    .join("rect")
    .attr("x", (d: DataPoint) => xScale(d.col)!)
    .attr("y", (d: DataPoint) => yScale(d.row)!)
    .attr("width", innerWidth / nCols)
    .attr("height", innerHeight / nRows)
    .attr("fill", (d: DataPoint) => {
      const isHighlighted = highlightSet.has(`${d.row}:${d.col}`);
      if (isHighlighted) {
        return "#FF6B35"; // Bright orange for highlighted cells
      }
      return colorScale(String(d.value));
    })
    .attr("stroke", "#fff")
    .attr("stroke-width", 0.5);

  // Path connecting highlighted cells
  if (highlightData && highlightData.length > 0) {
    renderHighlightPath(focusG, highlightData, xScale, yScale, innerWidth, innerHeight, nCols, nRows, focusCols);
  }

  // Axes
  renderAxes(focusG, xScale, yScale, nCols, nRows);

  // Hover interactions
  addHoverInteractions(focusG, cells, xScale, yScale, innerWidth, innerHeight, nCols, nRows, neon);
};

const renderGridLines = (
  focusG: d3.Selection<SVGGElement, unknown, null, undefined>,
  nRows: number,
  nCols: number,
  cellSize: number,
  innerWidth: number,
  innerHeight: number
) => {
  // Horizontal lines
  focusG
    .append("g")
    .attr("stroke", "#fff")
    .attr("stroke-width", 0.5)
    .selectAll("line.h")
    .data(d3.range(nRows + 1))
    .join("line")
    .attr("x1", 0)
    .attr("y1", (d: number) => d * cellSize)
    .attr("x2", innerWidth)
    .attr("y2", (d: number) => d * cellSize);

  // Vertical lines
  focusG
    .append("g")
    .attr("stroke", "#fff")
    .attr("stroke-width", 0.5)
    .selectAll("line.v")
    .data(d3.range(nCols + 1))
    .join("line")
    .attr("x1", (d: number) => d * (innerWidth / nCols))
    .attr("y1", 0)
    .attr("x2", (d: number) => d * (innerWidth / nCols))
    .attr("y2", innerHeight);
};

const renderAxes = (
  focusG: d3.Selection<SVGGElement, unknown, null, undefined>,
  xScale: d3.ScaleBand<string>,
  yScale: d3.ScaleBand<string>,
  nCols: number,
  nRows: number
) => {
  // X axis
  focusG
    .append("g")
    .call(
      d3
        .axisTop(xScale)
        .tickSize(0)
        .tickValues(
          xScale
            .domain()
            .filter((_, i) => i % Math.ceil(nCols / 10) === 0)
        )
    )
    .selectAll("text")
    .attr("transform", "translate(5,-5) rotate(-90)")
    .style("text-anchor", "start")
    .style("font-size", "10px");

  // Y axis
  focusG
    .append("g")
    .call(
      d3
        .axisLeft(yScale)
        .tickSize(0)
        .tickValues(
          yScale
            .domain()
            .filter((_, i) => i % Math.ceil(nRows / 10) === 0)
        )
    )
    .selectAll("text")
    .style("font-size", "10px");
};

const addHoverInteractions = (
  focusG: d3.Selection<SVGGElement, unknown, null, undefined>,
  cells: d3.Selection<SVGRectElement, DataPoint, SVGGElement, unknown>,
  xScale: d3.ScaleBand<string>,
  yScale: d3.ScaleBand<string>,
  innerWidth: number,
  innerHeight: number,
  nCols: number,
  nRows: number,
  neon: string
) => {
  const tooltip = d3.select("body").select(".matrix-tooltip");
  const hoverLayer = focusG.append("g");

  cells
    .on("mouseover", function(_event, d) {
      const dataPoint = d as DataPoint;
      hoverLayer.selectAll("rect").remove();
      hoverLayer
        .append("rect")
        .attr("x", xScale(dataPoint.col)!)
        .attr("y", yScale(dataPoint.row)!)
        .attr("width", innerWidth / nCols)
        .attr("height", innerHeight / nRows)
        .attr("fill", "none")
        .attr("stroke", neon)
        .attr("stroke-width", 1.5)
        .style("opacity", 0.3)
        .transition()
        .duration(200)
        .style("opacity", 1);
      hoverLayer.raise();
      showTooltip(tooltip, dataPoint);
    })
    .on("mousemove", (event) => {
      moveTooltip(tooltip, event);
    })
    .on("mouseout", () => {
      hoverLayer
        .selectAll("rect")
        .transition()
        .duration(200)
        .style("opacity", 0)
        .remove();
      hideTooltip(tooltip);
    });
};

const renderHighlightPath = (
  focusG: d3.Selection<SVGGElement, unknown, null, undefined>,
  highlightData: HighlightData[],
  xScale: d3.ScaleBand<string>,
  yScale: d3.ScaleBand<string>,
  innerWidth: number,
  innerHeight: number,
  nCols: number,
  nRows: number,
  focusCols: string[]
) => {
  // Filter highlights to only include those visible in current focus
  const visibleHighlights = highlightData.filter(h => 
    focusCols.includes(h.col) && yScale.domain().includes(h.row)
  );

  // Sort by column order to ensure path follows column sequence
  visibleHighlights.sort((a, b) => {
    const aIndex = focusCols.indexOf(a.col);
    const bIndex = focusCols.indexOf(b.col);
    return aIndex - bIndex;
  });

  if (visibleHighlights.length < 2) return; // Need at least 2 points to draw a path

  // Calculate center coordinates for each highlighted cell
  const pathPoints = visibleHighlights.map(h => {
    const x = xScale(h.col)! + (innerWidth / nCols) / 2;
    const y = yScale(h.row)! + (innerHeight / nRows) / 2;
    return [x, y] as [number, number];
  });

  // Create line generator
  const line = d3.line()
    .x(d => d[0])
    .y(d => d[1])
    .curve(d3.curveLinear);

  // Remove existing path
  focusG.selectAll(".highlight-path").remove();

  // Draw path
  focusG
    .append("path")
    .datum(pathPoints)
    .attr("class", "highlight-path")
    .attr("d", line)
    .attr("fill", "none")
    .attr("stroke", "#FF0080") // Bright pink/magenta for the path
    .attr("stroke-width", 3)
    .attr("stroke-dasharray", "5,5")
    .style("opacity", 0.8);

  // Add circles at each point for better visibility
  focusG
    .selectAll(".highlight-point")
    .data(pathPoints)
    .join("circle")
    .attr("class", "highlight-point")
    .attr("cx", d => d[0])
    .attr("cy", d => d[1])
    .attr("r", 3)
    .attr("fill", "#FF0080")
    .attr("stroke", "#fff")
    .attr("stroke-width", 1)
    .style("opacity", 0.9);
};