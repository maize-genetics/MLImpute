import React, { useEffect, useRef } from 'react';
import { AxisBrush, AxisBrushConfig } from './AxisBrush';
import { Interval } from './types';

interface BrushableAxisProps {
  labels: string[];
  interval: Interval;
  onIntervalChange: (interval: Interval) => void;
  orientation: 'horizontal' | 'vertical';
  width?: number;
  height?: number;
  className?: string;
  snapToInteger?: boolean;
}

const BrushableAxis: React.FC<BrushableAxisProps> = ({
  labels,
  interval,
  onIntervalChange,
  orientation,
  width = orientation === 'horizontal' ? 400 : 60,
  height = orientation === 'horizontal' ? 60 : 400,
  className = '',
  snapToInteger = false,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const brushRef = useRef<AxisBrush | null>(null);

  useEffect(() => {
    if (!containerRef.current || labels.length === 0) return;

    try {
      const config: AxisBrushConfig = {
        width,
        height,
        margin: { top: 10, right: 10, bottom: 10, left: 10 },
        totalItems: labels.length,
        visibleItems: interval.end - interval.start,
        labels,
        interval,
        onIntervalChange,
        orientation,
        snapToInteger,
      };

      // Clean up existing brush
      if (brushRef.current) {
        brushRef.current.destroy();
      }

      // Create new brush
      brushRef.current = new AxisBrush(containerRef.current, config);
    } catch (error) {
      console.error('Error creating AxisBrush:', error);
    }

    return () => {
      if (brushRef.current) {
        try {
          brushRef.current.destroy();
          brushRef.current = null;
        } catch (error) {
          console.error('Error destroying AxisBrush:', error);
        }
      }
    };
  }, [labels.length, orientation, width, height, snapToInteger]);

  // Update brush when interval changes from external source
  useEffect(() => {
    if (brushRef.current) {
      try {
        brushRef.current.updateConfig({ interval });
      } catch (error) {
        console.error('Error updating brush interval:', error);
      }
    }
  }, [interval]);

  return (
    <div 
      ref={containerRef} 
      className={`brushable-axis ${className}`}
      style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
      }}
    />
  );
};

export default BrushableAxis;