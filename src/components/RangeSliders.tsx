import React from 'react';
import { Interval } from './types';
import D3RangeSlider from './D3RangeSlider';
import './RangeSliders.css';

interface RangeSlidersProps {
  xInterval: Interval;
  yInterval: Interval;
  maxCols: number;
  maxRows: number;
  onXIntervalChange: (interval: Interval) => void;
  onYIntervalChange: (interval: Interval) => void;
  colLabels: string[];
  rowLabels: string[];
}

const RangeSliders: React.FC<RangeSlidersProps> = ({
  xInterval,
  yInterval,
  maxCols,
  maxRows,
  onXIntervalChange,
  onYIntervalChange,
  colLabels,
  rowLabels,
}) => {
  const resetXRange = () => {
    onXIntervalChange({ start: 0, end: Math.min(maxCols, colLabels.length) });
  };

  const resetYRange = () => {
    onYIntervalChange({ start: 0, end: Math.min(maxRows, rowLabels.length) });
  };

  return (
    <div className="range-sliders">
      <div className="range-section">
        <div className="range-header">
          <h4>Position Range</h4>
          <div className="range-info">
            <span>Showing {xInterval.end - xInterval.start} of {colLabels.length} columns</span>
            <button onClick={resetXRange} className="reset-button">
              Reset
            </button>
          </div>
        </div>
        <D3RangeSlider
          min={0}
          max={colLabels.length}
          range={xInterval}
          onChange={onXIntervalChange}
          width={500}
          height={80}
          className="x-range-slider"
        />
      </div>

      <div className="range-section">
        <div className="range-header">
          <h4>Sample Range</h4>
          <div className="range-info">
            <span>Showing {yInterval.end - yInterval.start} of {rowLabels.length} rows</span>
            <button onClick={resetYRange} className="reset-button">
              Reset
            </button>
          </div>
        </div>
        <D3RangeSlider
          min={0}
          max={rowLabels.length}
          range={yInterval}
          onChange={onYIntervalChange}
          width={500}
          height={80}
          className="y-range-slider"
        />
      </div>
    </div>
  );
};

export default RangeSliders;