import React, { useEffect, useRef, useState } from 'react';
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
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState(500);

  // Monitor container size to make sliders responsive
  useEffect(() => {
    const updateContainerWidth = () => {
      if (containerRef.current) {
        const width = containerRef.current.offsetWidth;
        // Reserve space for padding and margins
        const availableWidth = Math.max(300, width - 40); // minimum 300px, subtract 40px for padding
        setContainerWidth(availableWidth);
      }
    };

    updateContainerWidth();
    
    const resizeObserver = new ResizeObserver(updateContainerWidth);
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }

    return () => resizeObserver.disconnect();
  }, []);

  const resetXRange = () => {
    onXIntervalChange({ start: 0, end: Math.min(maxCols, colLabels.length) });
  };

  const resetYRange = () => {
    onYIntervalChange({ start: 0, end: Math.min(maxRows, rowLabels.length) });
  };

  return (
    <div ref={containerRef} className="range-sliders">
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
          width={containerWidth}
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
          width={containerWidth}
          height={80}
          className="y-range-slider"
        />
      </div>
    </div>
  );
};

export default RangeSliders;