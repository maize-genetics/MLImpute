import { Dimensions, HighlightData } from "./types";

export const calculateDimensions = (
  maxVisibleRows: number,
  maxVisibleCols: number,
  cellSize: number,
  margin: { top: number; right: number; bottom: number; left: number },
  contextSize: number
): Dimensions => {
  const innerWidth = maxVisibleCols * cellSize;
  const innerHeight = maxVisibleRows * cellSize;
  const totalWidth = innerWidth + margin.left + margin.right + contextSize;
  const totalHeight = innerHeight + margin.top + margin.bottom + contextSize + 20;

  return {
    innerWidth,
    innerHeight,
    totalWidth,
    totalHeight,
  };
};

/**
 * Generates a random binary matrix and corresponding labels
 */
export function generateRandomMatrix(
  numRows: number,
  numCols: number
): { matrix: number[][]; rowLabels: string[]; colLabels: string[] } {
  const matrix = Array.from({ length: numRows }, () =>
    Array.from({ length: numCols }, () => (Math.random() < 0.25 ? 1 : 0))
  );
  const rowLabels = Array.from({ length: numRows }, (_, i) => `Sample ${i + 1}`);
  const colLabels = Array.from({ length: numCols }, (_, j) => `Pos${j + 1}`);
  return { matrix, rowLabels, colLabels };
}

/**
 * Generates random highlighted cells - one per column
 */
export function generateRandomHighlights(
  rowLabels: string[],
  colLabels: string[]
): HighlightData[] {
  return colLabels.map(col => ({
    col,
    row: rowLabels[Math.floor(Math.random() * rowLabels.length)]
  }));
}