import React, { useState, useEffect } from 'react';
import D3Matrix from './D3Matrix';
import RangeSliders from './RangeSliders';
import { Interval, HighlightData } from './types';

interface MatrixVisualizationProps {
  data: number[][];
  rowLabels: string[];
  colLabels: string[];
  highlightData?: HighlightData[];
  cellSize?: number;
  margin?: { top: number; right: number; bottom: number; left: number };
  maxVisibleRows?: number;
  maxVisibleCols?: number;
}

const MatrixVisualization: React.FC<MatrixVisualizationProps> = ({
  data,
  rowLabels,
  colLabels,
  highlightData,
  cellSize = 15,
  margin = { top: 100, right: 5, bottom: 5, left: 80 },
  maxVisibleRows = 20,
  maxVisibleCols = 40,
}) => {
  // State for intervals
  const [xInterval, setXInterval] = useState<Interval>({
    start: 0,
    end: Math.min(colLabels.length, maxVisibleCols),
  });
  
  const [yInterval, setYInterval] = useState<Interval>({
    start: 0,
    end: Math.min(rowLabels.length, maxVisibleRows),
  });

  // Reset intervals when data changes
  useEffect(() => {
    setXInterval({
      start: 0,
      end: Math.min(colLabels.length, maxVisibleCols),
    });
    setYInterval({
      start: 0,
      end: Math.min(rowLabels.length, maxVisibleRows),
    });
  }, [colLabels.length, rowLabels.length, maxVisibleCols, maxVisibleRows]);

  // Create focused data based on intervals
  const focusedData = data
    .slice(yInterval.start, yInterval.end)
    .map(row => row.slice(xInterval.start, xInterval.end));
  
  const focusedRowLabels = rowLabels.slice(yInterval.start, yInterval.end);
  const focusedColLabels = colLabels.slice(xInterval.start, xInterval.end);
  
  // Filter highlight data to only include visible cells
  const focusedHighlightData = highlightData?.filter(h => 
    focusedRowLabels.includes(h.row) && focusedColLabels.includes(h.col)
  );

  return (
    <div className="matrix-visualization">
      <RangeSliders
        xInterval={xInterval}
        yInterval={yInterval}
        maxCols={maxVisibleCols}
        maxRows={maxVisibleRows}
        onXIntervalChange={setXInterval}
        onYIntervalChange={setYInterval}
        colLabels={colLabels}
        rowLabels={rowLabels}
      />
      
      <div style={{ marginTop: '1rem', display: 'flex', justifyContent: 'center' }}>
        <D3Matrix
          data={focusedData}
          rowLabels={focusedRowLabels}
          colLabels={focusedColLabels}
          highlightData={focusedHighlightData}
          cellSize={cellSize}
          margin={margin}
          maxVisibleRows={Math.min(maxVisibleRows, focusedRowLabels.length)}
          maxVisibleCols={Math.min(maxVisibleCols, focusedColLabels.length)}
        />
      </div>
    </div>
  );
};

export default MatrixVisualization;