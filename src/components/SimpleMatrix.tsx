import React, { useState, useEffect, useCallback } from 'react';
import D3Matrix from './D3Matrix';
import RangeSliders from './RangeSliders';
import { Interval, HighlightData } from './types';

interface SimpleMatrixProps {
  data: number[][];
  rowLabels: string[];
  colLabels: string[];
  highlightData?: HighlightData[];
  cellSize?: number;
  margin?: { top: number; right: number; bottom: number; left: number };
  maxVisibleRows?: number;
  maxVisibleCols?: number;
}

const SimpleMatrix: React.FC<SimpleMatrixProps> = ({
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

  // Reset function
  const resetView = useCallback(() => {
    setXInterval({
      start: 0,
      end: Math.min(colLabels.length, maxVisibleCols),
    });
    setYInterval({
      start: 0,
      end: Math.min(rowLabels.length, maxVisibleRows),
    });
  }, [colLabels.length, maxVisibleCols, rowLabels.length, maxVisibleRows]);

  // Reset intervals when data changes
  useEffect(() => {
    resetView();
  }, [resetView]);

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
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', alignItems: 'center' }}>
      <div style={{ width: '100%', maxWidth: '800px' }}>
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
      </div>
      
      <div style={{ display: 'flex', justifyContent: 'center' }}>
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
      
      <div style={{ 
        background: '#f8f9fa', 
        padding: '0.75rem', 
        borderRadius: '0.25rem',
        border: '1px solid #dee2e6',
        fontSize: '0.875rem'
      }}>
        <div>
          <strong>Viewing:</strong> 
          Rows {yInterval.start + 1}-{yInterval.end} of {rowLabels.length} | 
          Cols {xInterval.start + 1}-{xInterval.end} of {colLabels.length}
        </div>
      </div>
    </div>
  );
};

export default SimpleMatrix;