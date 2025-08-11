export interface D3MatrixProps {
  data: number[][];
  rowLabels: string[];
  colLabels: string[];
  highlightData?: HighlightData[];
  cellSize?: number;
  margin?: { top: number; right: number; bottom: number; left: number };
  maxVisibleRows?: number;
  maxVisibleCols?: number;
  contextSize?: number;
}

export interface HighlightData {
  col: string;
  row: string;
}

export interface DataPoint {
  row: string;
  col: string;
  value: number;
}

export interface Interval {
  start: number;
  end: number;
}

export interface Dimensions {
  innerWidth: number;
  innerHeight: number;
  totalWidth: number;
  totalHeight: number;
}