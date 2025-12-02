import * as d3 from 'd3';
import { Interval } from './types';

export interface AxisBrushConfig {
  width: number;
  height: number;
  margin: { top: number; right: number; bottom: number; left: number };
  totalItems: number;
  visibleItems: number;
  labels: string[];
  interval: Interval;
  onIntervalChange: (interval: Interval) => void;
  orientation: 'horizontal' | 'vertical';
  snapToInteger?: boolean;
}

export class AxisBrush {
  private svg!: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private brush!: d3.BrushBehavior<unknown>;
  private scale!: d3.ScaleLinear<number, number>;
  private config: AxisBrushConfig;
  private brushG!: d3.Selection<SVGGElement, unknown, null, undefined>;
  
  constructor(container: HTMLElement, config: AxisBrushConfig) {
    this.config = config;
    this.setupSVG(container);
    this.setupScale();
    this.setupBrush();
    this.render();
  }

  private setupSVG(container: HTMLElement) {
    // Clear existing SVG
    d3.select(container).selectAll('svg').remove();
    
    this.svg = d3.select(container)
      .append('svg')
      .attr('width', this.config.width)
      .attr('height', this.config.height)
      .style('display', 'block')
      .style('user-select', 'none');
  }

  private setupScale() {
    const { margin, totalItems, orientation, width, height } = this.config;
    
    if (orientation === 'horizontal') {
      const innerWidth = width - margin.left - margin.right;
      this.scale = d3.scaleLinear()
        .domain([0, totalItems])
        .range([0, innerWidth]);
    } else {
      const innerHeight = height - margin.top - margin.bottom;
      this.scale = d3.scaleLinear()
        .domain([0, totalItems])
        .range([0, innerHeight]);
    }
  }

  private setupBrush() {
    const { margin, orientation, width, height } = this.config;
    
    if (orientation === 'horizontal') {
      const innerWidth = width - margin.left - margin.right;
      this.brush = d3.brushX()
        .extent([[0, 0], [innerWidth, 40]])
        .on('brush', this.handleBrush.bind(this))
        .on('end', this.handleBrushEnd.bind(this));
    } else {
      const innerHeight = height - margin.top - margin.bottom;
      this.brush = d3.brushY()
        .extent([[0, 0], [40, innerHeight]])
        .on('brush', this.handleBrush.bind(this))
        .on('end', this.handleBrushEnd.bind(this));
    }
  }

  private handleBrush(event: d3.D3BrushEvent<unknown>) {
    if (!event.selection) return;
    
    const { totalItems, onIntervalChange } = this.config;
    const selection = event.selection as [number, number];
    
    let start: number, end: number;
    
    if (this.config.orientation === 'horizontal') {
      start = this.scale.invert(selection[0]);
      end = this.scale.invert(selection[1]);
    } else {
      start = this.scale.invert(selection[0]);
      end = this.scale.invert(selection[1]);
    }

    // Apply basic bounds without snapping during brushing for smooth interaction
    start = Math.max(0, start);
    end = Math.min(totalItems, end);

    // Update interval with continuous values
    const newInterval: Interval = {
      start: Math.floor(start),
      end: Math.ceil(end)
    };
    
    onIntervalChange(newInterval);
  }

  private handleBrushEnd(event: d3.D3BrushEvent<unknown>) {
    if (!event.selection) return;
    
    const { totalItems, snapToInteger, onIntervalChange } = this.config;
    const selection = event.selection as [number, number];
    
    let start: number, end: number;
    
    if (this.config.orientation === 'horizontal') {
      start = this.scale.invert(selection[0]);
      end = this.scale.invert(selection[1]);
    } else {
      start = this.scale.invert(selection[0]);
      end = this.scale.invert(selection[1]);
    }

    // Apply snapping only on end if enabled
    if (snapToInteger) {
      start = Math.round(Math.max(0, start));
      end = Math.round(Math.min(totalItems, end));
      
      // Ensure minimum selection size
      if (end - start < 1) {
        if (start === 0) {
          end = 1;
        } else if (end === totalItems) {
          start = totalItems - 1;
        } else {
          end = start + 1;
        }
      }
      
      // Update brush to snap position
      const snappedSelection: [number, number] = [
        this.scale(start),
        this.scale(end)
      ];
      
      this.brushG.call(this.brush.move, snappedSelection);
      
      // Update interval with snapped values
      const newInterval: Interval = {
        start: start,
        end: end
      };
      
      onIntervalChange(newInterval);
    } else {
      // Just apply bounds without snapping
      start = Math.max(0, start);
      end = Math.min(totalItems, end);
      
      const newInterval: Interval = {
        start: Math.floor(start),
        end: Math.ceil(end)
      };
      
      onIntervalChange(newInterval);
    }
  }

  private render() {
    const { margin } = this.config;
    
    // Create main group
    const g = this.svg.append('g')
      .attr('transform', `translate(${margin.left}, ${margin.top})`);

    // Render background track
    this.renderTrack(g);
    
    // Skip tick marks and labels for cleaner interface
    
    // Add brush
    this.brushG = g.append('g')
      .attr('class', 'brush')
      .call(this.brush);
      
    // Style the brush
    this.styleBrush();
    
    // Set initial brush position
    this.updateBrushPosition();
  }

  private renderTrack(g: d3.Selection<SVGGElement, unknown, null, undefined>) {
    const { orientation, width, height, margin } = this.config;
    
    if (orientation === 'horizontal') {
      const innerWidth = width - margin.left - margin.right;
      g.append('rect')
        .attr('class', 'track')
        .attr('x', 0)
        .attr('y', 15)
        .attr('width', innerWidth)
        .attr('height', 10)
        .attr('fill', '#e1e5e9')
        .attr('rx', 5);
    } else {
      const innerHeight = height - margin.top - margin.bottom;
      g.append('rect')
        .attr('class', 'track')
        .attr('x', 15)
        .attr('y', 0)
        .attr('width', 10)
        .attr('height', innerHeight)
        .attr('fill', '#e1e5e9')
        .attr('rx', 5);
    }
  }


  private styleBrush() {
    // Style the brush selection
    this.brushG.selectAll('.selection')
      .attr('fill', '#3b82f6')
      .attr('fill-opacity', 0.3)
      .attr('stroke', '#2563eb')
      .attr('stroke-width', 2)
      .attr('rx', 4);

    // Style the brush handles
    this.brushG.selectAll('.handle')
      .attr('fill', '#1f2937')
      .attr('stroke', '#ffffff')
      .attr('stroke-width', 2)
      .attr('rx', 3);

    // Add custom handle styling
    if (this.config.orientation === 'horizontal') {
      this.brushG.selectAll('.handle')
        .attr('width', 8)
        .attr('height', 40);
    } else {
      this.brushG.selectAll('.handle')
        .attr('width', 40)
        .attr('height', 8);
    }
  }

  private updateBrushPosition() {
    const { interval } = this.config;
    const selection: [number, number] = [
      this.scale(interval.start),
      this.scale(interval.end)
    ];
    
    this.brushG.call(this.brush.move, selection);
  }

  public updateConfig(newConfig: Partial<AxisBrushConfig>) {
    this.config = { ...this.config, ...newConfig };
    this.setupScale();
    
    // Update brush position if interval changed
    if (newConfig.interval) {
      this.updateBrushPosition();
    }
  }

  public destroy() {
    this.svg.remove();
  }
}