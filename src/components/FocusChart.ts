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
  // Color scale removed - using highlight-based coloring instead

  const nCols = focusCols.length;
  const nRows = focusRows.length;

  // Grid lines
  renderGridLines(focusG, focusRows, focusCols, xScale, yScale, innerWidth, innerHeight);

  // Cells
  const flat = focusData.flatMap((row, i) =>
    row.map((val, j) => ({
      row: focusRows[i],
      col: focusCols[j],
      value: val,
    }))
  );

  // Create highlight lookup for faster checking with parent information
  const highlightMap = new Map<string, 'parent1' | 'parent2' | 'both'>();
  if (highlightData) {
    highlightData.forEach(h => {
      const key = `${h.row}:${h.col}`;
      const existing = highlightMap.get(key);
      if (existing && existing !== h.parent) {
        highlightMap.set(key, 'both');
      } else {
        highlightMap.set(key, h.parent || 'parent1');
      }
    });
  }

  const cells = focusG
    .append("g")
    .selectAll<SVGRectElement, DataPoint>("rect")
    .data(flat)
    .join("rect")
    .attr("x", (d: DataPoint) => xScale(d.col)!)
    .attr("y", (d: DataPoint) => yScale(d.row)!)
    .attr("width", xScale.bandwidth())
    .attr("height", yScale.bandwidth())
    .attr("fill", (d: DataPoint) => {
      const parent = highlightMap.get(`${d.row}:${d.col}`);
      if (parent === 'parent1') {
        return "#FF6B35"; // Orange for parent1
      } else if (parent === 'parent2') {
        return "#2E86AB"; // Blue for parent2
      } else if (parent === 'both') {
        return "#9D4EDD"; // Purple for overlap
      }
      return "#E8E8E8"; // Light grey for non-parent cells
    })
    .attr("stroke", "#fff")
    .attr("stroke-width", 0.5);

  // Paths connecting highlighted cells for each parent
  if (highlightData && highlightData.length > 0) {
    renderParentPaths(focusG, highlightData, xScale, yScale, innerWidth, innerHeight, nCols, nRows, focusCols);
  }

  // Axes
  renderAxes(focusG, xScale, yScale);

  // Hover interactions
  addHoverInteractions(focusG, cells, xScale, yScale, neon);
};

const renderGridLines = (
  focusG: d3.Selection<SVGGElement, unknown, null, undefined>,
  focusRows: string[],
  focusCols: string[],
  xScale: d3.ScaleBand<string>,
  yScale: d3.ScaleBand<string>,
  innerWidth: number,
  innerHeight: number
) => {
  // Horizontal lines - at the boundaries between rows
  const rowPositions = [0, ...focusRows.map(row => yScale(row)! + yScale.bandwidth()), innerHeight];
  focusG
    .append("g")
    .attr("stroke", "#fff")
    .attr("stroke-width", 0.5)
    .selectAll("line.h")
    .data(rowPositions)
    .join("line")
    .attr("x1", 0)
    .attr("y1", (d: number) => d)
    .attr("x2", innerWidth)
    .attr("y2", (d: number) => d);

  // Vertical lines - at the boundaries between columns
  const colPositions = [0, ...focusCols.map(col => xScale(col)! + xScale.bandwidth()), innerWidth];
  focusG
    .append("g")
    .attr("stroke", "#fff")
    .attr("stroke-width", 0.5)
    .selectAll("line.v")
    .data(colPositions)
    .join("line")
    .attr("x1", (d: number) => d)
    .attr("y1", 0)
    .attr("x2", (d: number) => d)
    .attr("y2", innerHeight);
};

const renderAxes = (
  focusG: d3.Selection<SVGGElement, unknown, null, undefined>,
  xScale: d3.ScaleBand<string>,
  yScale: d3.ScaleBand<string>
) => {
  // X axis - show all labels
  focusG
    .append("g")
    .attr("class", "axis x-axis")
    .call(
      d3
        .axisTop(xScale)
        .tickSize(0)
    )
    .selectAll("text")
    .attr("transform", "translate(5,-5) rotate(-90)")
    .style("text-anchor", "start")
    .style("font-size", "10px");

  // Y axis - show all labels
  focusG
    .append("g")
    .attr("class", "axis y-axis")
    .call(
      d3
        .axisLeft(yScale)
        .tickSize(0)
    )
    .selectAll("text")
    .style("font-size", "10px");
};

const addHoverInteractions = (
  focusG: d3.Selection<SVGGElement, unknown, null, undefined>,
  cells: d3.Selection<SVGRectElement, DataPoint, SVGGElement, unknown>,
  xScale: d3.ScaleBand<string>,
  yScale: d3.ScaleBand<string>,
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
        .attr("width", xScale.bandwidth())
        .attr("height", yScale.bandwidth())
        .attr("fill", "none")
        .attr("stroke", neon)
        .attr("stroke-width", 1.5)
        .style("opacity", 0.3)
        .transition()
        .duration(200)
        .style("opacity", 1);
      hoverLayer.raise();
      
      // Show and highlight corresponding axis labels
      const xAxis = focusG.select(".x-axis");
      const yAxis = focusG.select(".y-axis");
      
      // Find or create the specific row and column labels if they don't exist
      let colLabel = xAxis.selectAll<SVGTextElement, unknown>("text").filter(function() {
        return d3.select(this).text() === dataPoint.col;
      });
      
      let rowLabel = yAxis.selectAll<SVGTextElement, unknown>("text").filter(function() {
        return d3.select(this).text() === dataPoint.row;
      });
      
      // If column label doesn't exist (was filtered out), create it temporarily
      if (colLabel.empty()) {
        const colPosition = xScale(dataPoint.col)! + xScale.bandwidth()/2;
        colLabel = xAxis.append("text")
          .attr("class", "temp-label")
          .text(dataPoint.col)
          .attr("x", colPosition)
          .attr("y", -10)
          .attr("transform", `rotate(-90, ${colPosition}, -10)`)
          .style("text-anchor", "end")
          .style("font-size", "10px")
          .style("font-weight", "bold");
      }
      
      // If row label doesn't exist (was filtered out), create it temporarily  
      if (rowLabel.empty()) {
        const rowPosition = yScale(dataPoint.row)! + yScale.bandwidth()/2;
        rowLabel = yAxis.append("text")
          .attr("class", "temp-label")
          .text(dataPoint.row)
          .attr("x", -10)
          .attr("y", rowPosition)
          .attr("dy", "0.32em")
          .style("text-anchor", "end")
          .style("font-size", "10px")
          .style("font-weight", "bold");
      }
      
      // Make existing labels bold (temporary labels are already bold)
      if (!colLabel.classed("temp-label")) {
        colLabel.style("font-weight", "bold");
      }
      if (!rowLabel.classed("temp-label")) {
        rowLabel.style("font-weight", "bold");
      }
      
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
      
      // Reset axis labels to default styling and remove temporary labels
      focusG.selectAll(".axis text")
        .style("font-weight", "normal");
      
      // Remove temporary labels
      focusG.selectAll(".temp-label").remove();
      
      hideTooltip(tooltip);
    });
};

const renderParentPaths = (
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
  // Remove existing paths
  focusG.selectAll(".parent-path").remove();
  focusG.selectAll(".parent-point").remove();

  // Separate highlights by parent
  const parent1Highlights = highlightData.filter(h => h.parent === 'parent1');
  const parent2Highlights = highlightData.filter(h => h.parent === 'parent2');

  // Render path for parent1
  renderSingleParentPath(focusG, parent1Highlights, xScale, yScale, innerWidth, innerHeight, nCols, nRows, focusCols, {
    color: '#FF6B35',
    className: 'parent1-path',
    strokeWidth: 3,
    strokeDash: '8,4'
  });

  // Render path for parent2
  renderSingleParentPath(focusG, parent2Highlights, xScale, yScale, innerWidth, innerHeight, nCols, nRows, focusCols, {
    color: '#2E86AB', 
    className: 'parent2-path',
    strokeWidth: 3,
    strokeDash: '4,8'
  });
};

const renderSingleParentPath = (
  focusG: d3.Selection<SVGGElement, unknown, null, undefined>,
  highlights: HighlightData[],
  xScale: d3.ScaleBand<string>,
  yScale: d3.ScaleBand<string>,
  innerWidth: number,
  innerHeight: number,
  nCols: number,
  nRows: number,
  focusCols: string[],
  style: { color: string; className: string; strokeWidth: number; strokeDash: string }
) => {
  // Filter highlights to only include those visible in current focus
  const visibleHighlights = highlights.filter(h => 
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

  // Draw path
  focusG
    .append("path")
    .datum(pathPoints)
    .attr("class", `parent-path ${style.className}`)
    .attr("d", line)
    .attr("fill", "none")
    .attr("stroke", style.color)
    .attr("stroke-width", style.strokeWidth)
    .attr("stroke-dasharray", style.strokeDash)
    .style("opacity", 0.9);

  // Add circles at each point for better visibility
  focusG
    .selectAll(`.parent-point.${style.className}`)
    .data(pathPoints)
    .join("circle")
    .attr("class", `parent-point ${style.className}`)
    .attr("cx", d => d[0])
    .attr("cy", d => d[1])
    .attr("r", 4)
    .attr("fill", style.color)
    .attr("stroke", "#fff")
    .attr("stroke-width", 2)
    .style("opacity", 0.9);
};