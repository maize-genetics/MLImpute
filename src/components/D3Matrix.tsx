import React, { useEffect, useRef, useState } from "react";
import * as d3 from "d3";

interface D3MatrixProps {
  data: number[][];
  rowLabels: string[];
  colLabels: string[];
  cellSize?: number;
  margin?: { top: number; right: number; bottom: number; left: number };
  maxVisibleRows?: number;
  maxVisibleCols?: number;
  contextSize?: number;
}

interface DataPoint {
  row: string;
  col: string;
  value: number;
}

const D3Matrix: React.FC<D3MatrixProps> = (
  {
    data,
    rowLabels,
    colLabels,
    cellSize = 15,
    margin = { top: 20, right: 5, bottom: 5, left: 80 },
    maxVisibleRows = 20,
    maxVisibleCols = 40,
    contextSize = 50,
  }) => {
  const svgRef = useRef<SVGSVGElement | null>(null);

  // Create tooltip container
  useEffect(() => {
    d3.select("body").selectAll(".matrix-tooltip").remove();
    d3.select("body")
      .append("div")
      .attr("class", "matrix-tooltip")
      .style("position", "absolute")
      .style("pointer-events", "none")
      .style("background", "rgba(0,0,0,0.7)")
      .style("color", "#fff")
      .style("padding", "4px 8px")
      .style("border-radius", "4px")
      .style("font-size", "10px")
      .style("visibility", "hidden")
      .style("opacity", 0)
      .style("z-index", 1000);
  }, []);

  // Focus intervals
  const [xInterval, setXInterval] = useState({
    start: 0,
    end: Math.min(colLabels.length, maxVisibleCols),
  });
  const [yInterval, setYInterval] = useState({
    start: 0,
    end: Math.min(rowLabels.length, maxVisibleRows),
  });

  const rows = data.length;
  const cols = data[0]?.length || 0;
  const innerWidth = maxVisibleCols * cellSize;
  const innerHeight = maxVisibleRows * cellSize;
  const totalWidth = innerWidth + margin.left + margin.right + contextSize;
  const totalHeight = innerHeight + margin.top + margin.bottom + contextSize + 20;
  const neon = "#39FF14";

  // Draw main (focus) chart when intervals change
  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll(".focus").remove();
    const focusG = svg
      .append("g")
      .attr("class", "focus")
      .attr(
        "transform",
        `translate(${margin.left},${margin.top})`
      );

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

    // Cells
    const flat = focusData.flatMap((row, i) =>
      row.map((val, j) => ({
        row: focusRows[i],
        col: focusCols[j],
        value: val,
      }))
    );
    focusG
      .append("g")
      .selectAll("rect")
      .data(flat)
      .join("rect")
      .attr("x", (d: DataPoint) => xScale(d.col)!)
      .attr("y", (d: DataPoint) => yScale(d.row)!)
      .attr("width", innerWidth / nCols)
      .attr("height", innerHeight / nRows)
      .attr("fill", (d: DataPoint) => colorScale(String(d.value)))
      .attr("stroke", "#fff")
      .attr("stroke-width", 0.5);

    // Axes
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

    // Hover tooltip
    const tooltip = d3.select("body").select(".matrix-tooltip");
    const hoverLayer = focusG.append("g");
    focusG.selectAll("rect")
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
        tooltip
          .html(`Sample: ${dataPoint.row}<br/>Position: ${dataPoint.col}<br/>Value: ${dataPoint.value}`)
          .style("visibility", "visible")
          .transition()
          .duration(200)
          .style("opacity", 1)
          .style("z-index", 1001);
      })
      .on("mousemove", (event) => {
        tooltip
          .style("top", `${event.pageY + 10}px`)
          .style("left", `${event.pageX + 10}px`);
      })
      .on("mouseout", () => {
        hoverLayer
          .selectAll("rect")
          .transition()
          .duration(200)
          .style("opacity", 0)
          .remove();
        tooltip
          .transition()
          .duration(200)
          .style("opacity", 0)
          .on("end", () =>
            tooltip.style("visibility", "hidden")
          );
      });
  }, [data, xInterval, yInterval]);

  // Draw contexts & brushes once
  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll(".context").remove();

    // X context (bottom)
    const ctxX = svg
      .append("g")
      .attr("class", "context")
      .attr("transform", `translate(${margin.left},${margin.top + innerHeight + 20})`);
    const xCtxScale = d3
      .scaleBand<string>()
      .domain(colLabels)
      .range([0, innerWidth])
      .padding(0);
    ctxX
      .append("g")
      .selectAll("rect")
      .data(colLabels)
      .join("rect")
      .attr("x", (d: string) => xCtxScale(d)!)
      .attr("y", 0)
      .attr("width", xCtxScale.bandwidth())
      .attr("height", contextSize)
      .attr("fill", "#ccc");
    ctxX
      .append("g")
      .attr("transform", `translate(0,${contextSize})`)
      .call(
        d3
          .axisBottom(xCtxScale)
          .tickSize(0)
          .tickValues(
            xCtxScale
              .domain()
              .filter((_, i) => i % Math.ceil(cols / 20) === 0)
          )
      )
      .selectAll("text")
      .remove();
    const brushX = d3
      .brushX()
      .extent([[0, 0], [innerWidth, contextSize]])
      .on("brush end", (event) => {
        if (!event.selection) return;
        const [x0, x1] = event.selection;
        const s = Math.max(0, Math.floor((x0 / innerWidth) * cols));
        const e = Math.min(cols, Math.ceil((x1 / innerWidth) * cols));
        setXInterval({ start: s, end: e });
      });
    const bx = ctxX.append("g").attr("class", "brushX").call(brushX);
    // style brush selection
    bx.selectAll(".selection").attr("fill", "blue").attr("fill-opacity", 0.3).attr("stroke", "steelblue").attr("stroke-width", 1);
    // narrow brush handles
    bx.selectAll(".handle--n, .handle--s").attr("height", 4);
    // narrow brush handles
    bx.selectAll(".handle--w, .handle--e").attr("width", 4);
    bx.call(brushX.move, [
      (xInterval.start / cols) * innerWidth,
      (xInterval.end / cols) * innerWidth,
    ]);
    // border around X context
    ctxX.append("rect")
      .attr("x", 0)
      .attr("y", 0)
      .attr("width", innerWidth)
      .attr("height", contextSize)
      .attr("fill", "none")
      .attr("stroke", "#888")
      .attr("stroke-width", 1);


    // Y context (right)
    const ctxWidthY = contextSize;
    const ctxY = svg
      .append("g")
      .attr("class", "context")
      .attr(
        "transform",
        `translate(${margin.left + innerWidth + margin.right},${margin.top})`
      );
    const yCtxScale = d3
      .scaleBand<string>()
      .domain(rowLabels)
      .range([0, innerHeight])
      .padding(0);
    ctxY
      .append("g")
      .selectAll("rect")
      .data(rowLabels)
      .join("rect")
      .attr("y", (d: string) => yCtxScale(d)!)
      .attr("x", 0)
      .attr("height", yCtxScale.bandwidth())
      .attr("width", ctxWidthY)
      .attr("fill", "#ccc");
    ctxY
      .append("g")
      .attr("transform", `translate(${ctxWidthY},0)`)
      .call(
        d3
          .axisRight(yCtxScale)
          .tickSize(0)
          .tickValues(
            yCtxScale
              .domain()
              .filter((_, i) => i % Math.ceil(rows / 20) === 0)
          )
      )
      .selectAll("text")
      .remove();
    const brushY = d3
      .brushY()
      .extent([[0, 0], [ctxWidthY, innerHeight]])
      .on("brush end", (event) => {
        if (!event.selection) return;
        const [y0, y1] = event.selection;
        const s = Math.max(0, Math.floor((y0 / innerHeight) * rows));
        const e = Math.min(rows, Math.ceil((y1 / innerHeight) * rows));
        setYInterval({ start: s, end: e });
      });
    const by = ctxY.append("g").attr("class", "brushY").call(brushY);
    // style brush selection
    by.selectAll(".selection").attr("fill", "blue").attr("fill-opacity", 0.3).attr("stroke", "steelblue").attr("stroke-width", 1);
    by.call(brushY.move, [
      (yInterval.start / rows) * innerHeight,
      (yInterval.end / rows) * innerHeight,
    ]);
    // border around Y context
    ctxY.append("rect")
      .attr("x", 0)
      .attr("y", 0)
      .attr("width", ctxWidthY)
      .attr("height", innerHeight)
      .attr("fill", "none")
      .attr("stroke", "#888")
      .attr("stroke-width", 1);
  }, []);

  return <svg ref={svgRef} width={totalWidth} height={totalHeight} />;
};

export default D3Matrix;
