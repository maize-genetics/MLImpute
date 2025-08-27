import { Dimensions } from "./types";

export const calculateDimensions = (
  maxVisibleRows: number,
  maxVisibleCols: number,
  cellSize: number,
  margin: { top: number; right: number; bottom: number; left: number }
): Dimensions => {
  const innerWidth = maxVisibleCols * cellSize;
  const innerHeight = maxVisibleRows * cellSize;
  const totalWidth = innerWidth + margin.left + margin.right;
  const totalHeight = innerHeight + margin.top + margin.bottom;

  return {
    innerWidth,
    innerHeight,
    totalWidth,
    totalHeight,
  };
};

