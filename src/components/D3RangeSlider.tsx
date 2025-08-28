import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { Interval } from './types';

interface D3RangeSliderProps {
  min: number;
  max: number;
  range: Interval;
  onChange: (range: Interval) => void;
  width?: number;
  height?: number;
  label?: string;
  className?: string;
}

interface RangeSliderState {
  scale: d3.ScaleLinear<number, number>;
  brush: d3.BrushBehavior<unknown>;
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  brushG: d3.Selection<SVGGElement, unknown, null, undefined>;
  isUpdating: boolean;
}

class D3RangeSliderImpl {
  private container: HTMLElement;
  private props: D3RangeSliderProps;
  private state: RangeSliderState;
  private margin = { top: 10, right: 20, bottom: 40, left: 20 };

  constructor(container: HTMLElement, props: D3RangeSliderProps) {
    this.container = container;
    this.props = props;
    this.state = {} as RangeSliderState;
    this.init();
  }

  private init() {
    const { width = 400, height = 80 } = this.props;
    const innerWidth = width - this.margin.left - this.margin.right;
    
    // Clear existing content
    d3.select(this.container).selectAll('*').remove();
    
    // Create SVG
    this.state.svg = d3.select(this.container)
      .append('svg')
      .attr('width', width)
      .attr('height', height)
      .style('display', 'block');

    // Create main group
    const g = this.state.svg.append('g')
      .attr('transform', `translate(${this.margin.left}, ${this.margin.top})`);

    // Setup scale
    this.state.scale = d3.scaleLinear()
      .domain([this.props.min, this.props.max])
      .range([0, innerWidth])
      .clamp(true);

    // Create track background
    g.append('rect')
      .attr('class', 'track-background')
      .attr('x', 0)
      .attr('y', 20)
      .attr('width', innerWidth)
      .attr('height', 8)
      .attr('fill', '#e1e5e9')
      .attr('rx', 4);

    // Setup brush
    this.state.brush = d3.brushX()
      .extent([[0, 16], [innerWidth, 32]])
      .on('brush', this.handleBrush.bind(this))
      .on('end', this.handleBrushEnd.bind(this));

    // Add brush group
    this.state.brushG = g.append('g')
      .attr('class', 'brush')
      .call(this.state.brush);

    // Style the brush
    this.styleBrush();

    // Add axis
    const axis = d3.axisBottom(this.state.scale)
      .tickSize(-8)
      .tickPadding(8);

    g.append('g')
      .attr('class', 'axis')
      .attr('transform', 'translate(0, 40)')
      .call(axis);

    // Style axis
    g.select('.axis')
      .selectAll('text')
      .style('font-size', '11px')
      .style('fill', '#374151');
    
    g.select('.axis')
      .select('.domain')
      .style('stroke', '#d1d5db');

    g.select('.axis')
      .selectAll('.tick line')
      .style('stroke', '#d1d5db');

    // Add label if provided
    if (this.props.label) {
      g.append('text')
        .attr('class', 'range-label')
        .attr('x', innerWidth / 2)
        .attr('y', -5)
        .attr('text-anchor', 'middle')
        .style('font-size', '12px')
        .style('font-weight', '600')
        .style('fill', '#374151')
        .text(this.props.label);
    }

    // Set initial position
    this.updateBrushPosition();
  }

  private styleBrush() {
    // Style the selection area
    this.state.brushG.selectAll('.selection')
      .attr('fill', '#3b82f6')
      .attr('fill-opacity', 0.2)
      .attr('stroke', '#2563eb')
      .attr('stroke-width', 2)
      .attr('rx', 4);

    // Style the handles
    this.state.brushG.selectAll('.handle')
      .attr('fill', '#1f2937')
      .attr('stroke', '#ffffff')
      .attr('stroke-width', 2)
      .attr('width', 12)
      .attr('height', 20)
      .attr('rx', 2)
      .style('cursor', 'ew-resize');

    // Add hover effects
    this.state.brushG.selectAll('.handle')
      .on('mouseover', function() {
        d3.select(this).attr('fill', '#374151');
      })
      .on('mouseout', function() {
        d3.select(this).attr('fill', '#1f2937');
      });
  }

  private handleBrush(event: d3.D3BrushEvent<unknown>) {
    if (this.state.isUpdating) return;
    
    const selection = event.selection as [number, number] | null;
    if (!selection) return;

    const start = Math.max(this.props.min, Math.round(this.state.scale.invert(selection[0])));
    const end = Math.min(this.props.max, Math.round(this.state.scale.invert(selection[1])));

    // Ensure minimum range of 1
    const finalStart = start;
    const finalEnd = Math.max(start + 1, end);

    const newRange: Interval = {
      start: finalStart,
      end: finalEnd
    };

    this.props.onChange(newRange);
  }

  private handleBrushEnd(event: d3.D3BrushEvent<unknown>) {
    if (this.state.isUpdating) return;
    
    const selection = event.selection as [number, number] | null;
    if (!selection) return;

    // Snap to integer positions
    const start = Math.max(this.props.min, Math.round(this.state.scale.invert(selection[0])));
    const end = Math.min(this.props.max, Math.round(this.state.scale.invert(selection[1])));

    const finalStart = start;
    const finalEnd = Math.max(start + 1, end);

    // Update brush position to snapped values
    const snappedSelection: [number, number] = [
      this.state.scale(finalStart),
      this.state.scale(finalEnd)
    ];

    this.state.isUpdating = true;
    this.state.brushG.call(this.state.brush.move, snappedSelection);
    this.state.isUpdating = false;

    const newRange: Interval = {
      start: finalStart,
      end: finalEnd
    };

    this.props.onChange(newRange);
  }

  private updateBrushPosition() {
    const { range } = this.props;
    const selection: [number, number] = [
      this.state.scale(range.start),
      this.state.scale(range.end)
    ];
    
    this.state.isUpdating = true;
    this.state.brushG.call(this.state.brush.move, selection);
    this.state.isUpdating = false;
  }

  public updateRange(newRange: Interval) {
    if (this.state.isUpdating) return;
    this.props = { ...this.props, range: newRange };
    this.updateBrushPosition();
  }

  public destroy() {
    if (this.state.svg) {
      this.state.svg.remove();
    }
  }
}

const D3RangeSlider: React.FC<D3RangeSliderProps> = (props) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const sliderRef = useRef<D3RangeSliderImpl | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    // Clean up existing slider
    if (sliderRef.current) {
      sliderRef.current.destroy();
    }

    // Create new slider
    sliderRef.current = new D3RangeSliderImpl(containerRef.current, props);

    return () => {
      if (sliderRef.current) {
        sliderRef.current.destroy();
        sliderRef.current = null;
      }
    };
  }, [props.min, props.max, props.width, props.height, props.label]);

  // Update range when it changes externally
  useEffect(() => {
    if (sliderRef.current) {
      sliderRef.current.updateRange(props.range);
    }
  }, [props.range]);

  return (
    <div 
      ref={containerRef} 
      className={`d3-range-slider ${props.className || ''}`}
      style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        margin: '10px 0'
      }}
    />
  );
};

export default D3RangeSlider;