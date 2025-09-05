import { Dimensions } from "./types";

export const calculateDimensions = (
  actualRows: number,
  actualCols: number,
  cellSize: number,
  margin: { top: number; right: number; bottom: number; left: number }
): Dimensions => {
  const innerWidth = actualCols * cellSize;
  const innerHeight = actualRows * cellSize;
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
  actualRows: number,
  actualCols: number,
  containerWidth: number,
  containerHeight: number,
  margin: { top: number; right: number; bottom: number; left: number },
  minCellSize: number = 1,
  maxCellSize: number = 80
): number => {
  // Calculate available space for the matrix
  const availableWidth = containerWidth - margin.left - margin.right;
  const availableHeight = containerHeight - margin.top - margin.bottom;
  
  // Calculate cell size based on width and height constraints using actual data dimensions
  const cellSizeFromWidth = Math.floor(availableWidth / actualCols);
  const cellSizeFromHeight = Math.floor(availableHeight / actualRows);
  
  // Use the smaller dimension to ensure matrix fits in container
  const optimalCellSize = Math.min(cellSizeFromWidth, cellSizeFromHeight);
  
  // Clamp cell size within reasonable bounds
  // Use a smaller minimum (1px) to allow very dense matrices to still be visible
  return Math.max(minCellSize, Math.min(maxCellSize, optimalCellSize));
};

export const calculateResponsiveDimensions = (
  actualRows: number,
  actualCols: number,
  containerWidth: number,
  containerHeight: number,
  margin: { top: number; right: number; bottom: number; left: number },
  minCellSize?: number,
  maxCellSize?: number
): Dimensions & { cellSize: number } => {
  const cellSize = calculateResponsiveCellSize(
    actualRows,
    actualCols,
    containerWidth,
    containerHeight,
    margin,
    minCellSize,
    maxCellSize
  );
  
  const dimensions = calculateDimensions(
    actualRows,
    actualCols,
    cellSize,
    margin
  );
  
  return {
    ...dimensions,
    cellSize,
  };
};

