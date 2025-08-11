import * as d3 from "d3";
import { DataPoint } from "./types";

export const createTooltip = () => {
  d3.select("body").selectAll(".matrix-tooltip").remove();
  return d3
    .select("body")
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
};

export const showTooltip = (tooltip: d3.Selection<HTMLDivElement, unknown, HTMLElement, any>, dataPoint: DataPoint) => {
  tooltip
    .html(`Sample: ${dataPoint.row}<br/>Position: ${dataPoint.col}<br/>Value: ${dataPoint.value}`)
    .style("visibility", "visible")
    .transition()
    .duration(200)
    .style("opacity", 1)
    .style("z-index", 1001);
};

export const moveTooltip = (tooltip: d3.Selection<HTMLDivElement, unknown, HTMLElement, any>, event: MouseEvent) => {
  tooltip
    .style("top", `${event.pageY + 10}px`)
    .style("left", `${event.pageX + 10}px`);
};

export const hideTooltip = (tooltip: d3.Selection<HTMLDivElement, unknown, HTMLElement, any>) => {
  tooltip
    .transition()
    .duration(200)
    .style("opacity", 0)
    .on("end", () => tooltip.style("visibility", "hidden"));
};