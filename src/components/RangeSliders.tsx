import React from 'react';
import { Interval } from './types';
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
  const handleXStartChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const start = parseInt(e.target.value);
    // Only adjust end if start would exceed it
    if (start >= xInterval.end) {
      onXIntervalChange({ start, end: start + 1 });
    } else {
      onXIntervalChange({ start, end: xInterval.end });
    }
  };

  const handleXEndChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const end = parseInt(e.target.value);
    // Only adjust start if end would be less than or equal to it
    if (end <= xInterval.start) {
      onXIntervalChange({ start: end - 1, end });
    } else {
      onXIntervalChange({ start: xInterval.start, end });
    }
  };

  const handleYStartChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const start = parseInt(e.target.value);
    // Only adjust end if start would exceed it
    if (start >= yInterval.end) {
      onYIntervalChange({ start, end: start + 1 });
    } else {
      onYIntervalChange({ start, end: yInterval.end });
    }
  };

  const handleYEndChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const end = parseInt(e.target.value);
    // Only adjust start if end would be less than or equal to it
    if (end <= yInterval.start) {
      onYIntervalChange({ start: end - 1, end });
    } else {
      onYIntervalChange({ start: yInterval.start, end });
    }
  };

  const resetXRange = () => {
    onXIntervalChange({ start: 0, end: Math.min(maxCols, colLabels.length) });
  };

  const resetYRange = () => {
    onYIntervalChange({ start: 0, end: Math.min(maxRows, rowLabels.length) });
  };

  return (
    <div className="range-sliders">
      <div className="range-section">
        <h4>X-Axis Range (Columns)</h4>
        <div className="range-info">
          <span>Showing {xInterval.end - xInterval.start} of {colLabels.length} columns</span>
        </div>
        <div className="range-controls">
          <div className="range-input-group">
            <label>Start:</label>
            <input
              type="range"
              min={0}
              max={Math.max(0, colLabels.length - 1)}
              value={xInterval.start}
              onChange={handleXStartChange}
              className="range-slider"
            />
            <span className="range-value">{xInterval.start}</span>
          </div>
          <div className="range-input-group">
            <label>End:</label>
            <input
              type="range"
              min={1}
              max={Math.max(1, colLabels.length)}
              value={xInterval.end}
              onChange={handleXEndChange}
              className="range-slider"
            />
            <span className="range-value">{xInterval.end}</span>
          </div>
          <button onClick={resetXRange} className="reset-button">
            Reset
          </button>
        </div>
      </div>

      <div className="range-section">
        <h4>Y-Axis Range (Rows)</h4>
        <div className="range-info">
          <span>Showing {yInterval.end - yInterval.start} of {rowLabels.length} rows</span>
        </div>
        <div className="range-controls">
          <div className="range-input-group">
            <label>Start:</label>
            <input
              type="range"
              min={0}
              max={Math.max(0, rowLabels.length - 1)}
              value={yInterval.start}
              onChange={handleYStartChange}
              className="range-slider"
            />
            <span className="range-value">{yInterval.start}</span>
          </div>
          <div className="range-input-group">
            <label>End:</label>
            <input
              type="range"
              min={1}
              max={Math.max(1, rowLabels.length)}
              value={yInterval.end}
              onChange={handleYEndChange}
              className="range-slider"
            />
            <span className="range-value">{yInterval.end}</span>
          </div>
          <button onClick={resetYRange} className="reset-button">
            Reset
          </button>
        </div>
      </div>
    </div>
  );
};

export default RangeSliders;