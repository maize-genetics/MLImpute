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

export const calculateResponsiveCellSize = (
  maxVisibleRows: number,
  maxVisibleCols: number,
  containerWidth: number,
  containerHeight: number,
  margin: { top: number; right: number; bottom: number; left: number },
  minCellSize: number = 25,
  maxCellSize: number = 80
): number => {
  // Calculate available space for the matrix
  const availableWidth = containerWidth - margin.left - margin.right;
  const availableHeight = containerHeight - margin.top - margin.bottom;
  
  // Calculate cell size based on width and height constraints
  const cellSizeFromWidth = Math.floor(availableWidth / maxVisibleCols);
  const cellSizeFromHeight = Math.floor(availableHeight / maxVisibleRows);
  
  // Use the smaller dimension to ensure matrix fits in container
  const optimalCellSize = Math.min(cellSizeFromWidth, cellSizeFromHeight);
  
  // Clamp cell size within reasonable bounds
  return Math.max(minCellSize, Math.min(maxCellSize, optimalCellSize));
};

export const calculateResponsiveDimensions = (
  maxVisibleRows: number,
  maxVisibleCols: number,
  containerWidth: number,
  containerHeight: number,
  margin: { top: number; right: number; bottom: number; left: number },
  minCellSize?: number,
  maxCellSize?: number
): Dimensions & { cellSize: number } => {
  const cellSize = calculateResponsiveCellSize(
    maxVisibleRows,
    maxVisibleCols,
    containerWidth,
    containerHeight,
    margin,
    minCellSize,
    maxCellSize
  );
  
  const dimensions = calculateDimensions(
    maxVisibleRows,
    maxVisibleCols,
    cellSize,
    margin
  );
  
  return {
    ...dimensions,
    cellSize,
  };
};

