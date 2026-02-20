/**
 * Parse position input, supporting shorthand with units (K, M, B).
 * Examples: "2.4M" → 2400000, "3K" → 3000, "3.14B" → 3140000000.
 * Also accepts plain numbers with or without commas (e.g. "2,400,000").
 * Returns the position as an integer, or null if the input is invalid.
 */
export function parsePositionInput(input: string): number | null {
  const trimmed = input.trim().replace(/,/g, '');
  if (!trimmed) return null;
  const match = trimmed.match(/^(\d+(?:\.\d+)?)\s*([KMB])?$/i);
  if (!match) return null;
  const num = parseFloat(match[1]);
  if (isNaN(num)) return null;
  const suffix = (match[2] || '').toUpperCase();
  const multipliers: Record<string, number> = { K: 1e3, M: 1e6, B: 1e9 };
  const mult = multipliers[suffix] ?? 1;
  return Math.round(num * mult);
}

/**
 * Binary search to find the column index whose position is nearest to targetPos.
 * 
 * @param targetPos - The genomic position to search for.
 * @param getPosition - Accessor that returns the position value for a given column index.
 * @param length - Total number of columns.
 * @returns The index of the nearest column, or -1 if length is 0.
 */
export function findNearestColumnIndex(
  targetPos: number,
  getPosition: (index: number) => number,
  length: number,
): number {
  if (length === 0) return -1;

  let left = 0;
  let right = length - 1;

  while (left < right) {
    const mid = Math.floor((left + right) / 2);
    if (getPosition(mid) < targetPos) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }

  if (left > 0) {
    const diffLeft = Math.abs(getPosition(left - 1) - targetPos);
    const diffRight = Math.abs(getPosition(left) - targetPos);
    if (diffLeft < diffRight) return left - 1;
  }

  return left;
}

const BASE_CELL_SIZE = 12;
const LABEL_MARGIN = 100;
const PADDING = 20;

/**
 * Compute a normalized scroll offset (0–1) that places the given column
 * at the left edge of the viewport.
 *
 * @returns The new scroll offset, or null if everything fits in the viewport.
 */
export function calculateScrollOffset(
  columnIndex: number,
  numColumns: number,
  zoomLevel: number,
  cellWidthMultiplier: number,
  containerWidth: number,
): number | null {
  const cellWidth = BASE_CELL_SIZE * zoomLevel * cellWidthMultiplier;
  const totalWidth = numColumns * cellWidth;
  const viewportWidth = Math.max(containerWidth - LABEL_MARGIN - PADDING, 100);
  const maxScrollOffset = Math.max(0, totalWidth - viewportWidth);

  if (maxScrollOffset === 0) return null;

  const targetPixelPos = columnIndex * cellWidth;
  return Math.max(0, Math.min(1, targetPixelPos / maxScrollOffset));
}
