import * as d3 from "d3";
import { Interval } from "./types";

export const renderContextCharts = (
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  rowLabels: string[],
  colLabels: string[],
  xInterval: Interval,
  yInterval: Interval,
  setXInterval: (interval: Interval) => void,
  setYInterval: (interval: Interval) => void,
  margin: { top: number; right: number; bottom: number; left: number },
  innerWidth: number,
  innerHeight: number,
  contextSize: number,
  maxVisibleRows: number,
  maxVisibleCols: number
) => {
  const rows = rowLabels.length;
  const cols = colLabels.length;

  svg.selectAll(".context").remove();

  // X context (bottom)
  renderXContext(
    svg,
    colLabels,
    xInterval,
    setXInterval,
    margin,
    innerWidth,
    innerHeight,
    contextSize,
    cols
  );

  // Y context (right)
  renderYContext(
    svg,
    rowLabels,
    yInterval,
    setYInterval,
    margin,
    innerWidth,
    innerHeight,
    contextSize,
    rows
  );
};

const renderXContext = (
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  colLabels: string[],
  xInterval: Interval,
  setXInterval: (interval: Interval) => void,
  margin: { top: number; right: number; bottom: number; left: number },
  innerWidth: number,
  innerHeight: number,
  contextSize: number,
  cols: number
) => {
  const ctxX = svg
    .append("g")
    .attr("class", "context")
    .attr("transform", `translate(${margin.left},${margin.top + innerHeight + 20})`);

  const xCtxScale = d3
    .scaleBand<string>()
    .domain(colLabels)
    .range([0, innerWidth])
    .padding(0);

  // Background rectangles
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

  // Axis
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

  // Brush
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
  
  // Style brush
  bx.selectAll(".selection")
    .attr("fill", "blue")
    .attr("fill-opacity", 0.3)
    .attr("stroke", "steelblue")
    .attr("stroke-width", 1);
  
  bx.selectAll(".handle--n, .handle--s").attr("height", 4);
  bx.selectAll(".handle--w, .handle--e").attr("width", 4);
  
  bx.call(brushX.move, [
    (xInterval.start / cols) * innerWidth,
    (xInterval.end / cols) * innerWidth,
  ]);

  // Border
  ctxX
    .append("rect")
    .attr("x", 0)
    .attr("y", 0)
    .attr("width", innerWidth)
    .attr("height", contextSize)
    .attr("fill", "none")
    .attr("stroke", "#888")
    .attr("stroke-width", 1);
};

const renderYContext = (
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  rowLabels: string[],
  yInterval: Interval,
  setYInterval: (interval: Interval) => void,
  margin: { top: number; right: number; bottom: number; left: number },
  innerWidth: number,
  innerHeight: number,
  contextSize: number,
  rows: number
) => {
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

  // Background rectangles
  ctxY
    .append("g")
    .selectAll("rect")
    .data(rowLabels)
    .join("rect")
    .attr("y", (d: string) => yCtxScale(d)!)
    .attr("x", 0)
    .attr("height", yCtxScale.bandwidth())
    .attr("width", contextSize)
    .attr("fill", "#ccc");

  // Axis
  ctxY
    .append("g")
    .attr("transform", `translate(${contextSize},0)`)
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

  // Brush
  const brushY = d3
    .brushY()
    .extent([[0, 0], [contextSize, innerHeight]])
    .on("brush end", (event) => {
      if (!event.selection) return;
      const [y0, y1] = event.selection;
      const s = Math.max(0, Math.floor((y0 / innerHeight) * rows));
      const e = Math.min(rows, Math.ceil((y1 / innerHeight) * rows));
      setYInterval({ start: s, end: e });
    });

  const by = ctxY.append("g").attr("class", "brushY").call(brushY);
  
  // Style brush
  by.selectAll(".selection")
    .attr("fill", "blue")
    .attr("fill-opacity", 0.3)
    .attr("stroke", "steelblue")
    .attr("stroke-width", 1);
  
  by.call(brushY.move, [
    (yInterval.start / rows) * innerHeight,
    (yInterval.end / rows) * innerHeight,
  ]);

  // Border
  ctxY
    .append("rect")
    .attr("x", 0)
    .attr("y", 0)
    .attr("width", contextSize)
    .attr("height", innerHeight)
    .attr("fill", "none")
    .attr("stroke", "#888")
    .attr("stroke-width", 1);
};