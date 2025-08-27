import React, { useState } from 'react';
import RangeSliders from './RangeSliders';
import { Interval } from './types';
import './AdvancedControls.css';

interface AdvancedControlsProps {
  xInterval: Interval;
  yInterval: Interval;
  maxCols: number;
  maxRows: number;
  onXIntervalChange: (interval: Interval) => void;
  onYIntervalChange: (interval: Interval) => void;
  colLabels: string[];
  rowLabels: string[];
  onResetView: () => void;
}

const AdvancedControls: React.FC<AdvancedControlsProps> = ({
  xInterval,
  yInterval,
  maxCols,
  maxRows,
  onXIntervalChange,
  onYIntervalChange,
  colLabels,
  rowLabels,
  onResetView,
}) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="advanced-controls">
      <div className="advanced-controls-header">
        <button
          className={`dropdown-toggle ${isOpen ? 'open' : ''}`}
          onClick={() => setIsOpen(!isOpen)}
        >
          <span className="dropdown-icon">▶</span>
          Advanced Controls
        </button>
        
        <div className="controls-info">
          Rows {yInterval.start + 1}-{yInterval.end} of {rowLabels.length} | 
          Cols {xInterval.start + 1}-{xInterval.end} of {colLabels.length}
        </div>
        
        <button onClick={onResetView} className="reset-view-button">
          <span className="reset-icon">↻</span>
          Reset View
        </button>
      </div>

      {isOpen && (
        <div className="dropdown-content">
          <RangeSliders
            xInterval={xInterval}
            yInterval={yInterval}
            maxCols={maxCols}
            maxRows={maxRows}
            onXIntervalChange={onXIntervalChange}
            onYIntervalChange={onYIntervalChange}
            colLabels={colLabels}
            rowLabels={rowLabels}
          />
        </div>
      )}
    </div>
  );
};

export default AdvancedControls;