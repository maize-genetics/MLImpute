import React, { useState, useCallback, useEffect } from 'react';
import Icon from '@mdi/react';
import { mdiClose, mdiLink, mdiLinkOff } from '@mdi/js';
import './ExportModal.css';

export interface PathOverlayInfo {
  label: string;
  visible: boolean;
}

export interface ExportSettings {
  title: string;
  width: number;
  height: number;
  scale: number;
  includeLegend: boolean;
  pathVisibility: Record<string, boolean>;
  startPosition: number;
  endPosition: number;
}

interface ExportModalProps {
  defaultTitle: string;
  defaultWidth: number;
  defaultHeight: number;
  visibleStartPos: number;
  visibleEndPos: number;
  fullRangeStart: number;
  fullRangeEnd: number;
  pathOverlays: PathOverlayInfo[];
  onExport: (settings: ExportSettings) => void;
  onClose: () => void;
}

const SCALE_OPTIONS = [1, 2, 3, 4];

const ExportModal: React.FC<ExportModalProps> = ({
  defaultTitle,
  defaultWidth,
  defaultHeight,
  visibleStartPos,
  visibleEndPos,
  fullRangeStart,
  fullRangeEnd,
  pathOverlays,
  onExport,
  onClose,
}) => {
  const [title, setTitle] = useState(defaultTitle);
  const [width, setWidth] = useState(Math.round(defaultWidth));
  const [height, setHeight] = useState(Math.round(defaultHeight));
  const [scale, setScale] = useState(3);
  const [lockAspect, setLockAspect] = useState(true);
  const [includeLegend, setIncludeLegend] = useState(true);
  const [startPos, setStartPos] = useState(visibleStartPos);
  const [endPos, setEndPos] = useState(visibleEndPos);
  const [pathVisibility, setPathVisibility] = useState<Record<string, boolean>>(() => {
    const vis: Record<string, boolean> = {};
    for (const p of pathOverlays) {
      vis[p.label] = p.visible;
    }
    return vis;
  });

  const aspectRatio = defaultWidth / defaultHeight;

  const handleWidthChange = useCallback((newWidth: number) => {
    setWidth(newWidth);
    if (lockAspect && newWidth > 0) {
      setHeight(Math.round(newWidth / aspectRatio));
    }
  }, [lockAspect, aspectRatio]);

  const handleHeightChange = useCallback((newHeight: number) => {
    setHeight(newHeight);
    if (lockAspect && newHeight > 0) {
      setWidth(Math.round(newHeight * aspectRatio));
    }
  }, [lockAspect, aspectRatio]);

  const handlePathToggle = useCallback((label: string) => {
    setPathVisibility(prev => ({ ...prev, [label]: !prev[label] }));
  }, []);

  const handleExport = useCallback(() => {
    onExport({
      title,
      width,
      height,
      scale,
      includeLegend,
      pathVisibility,
      startPosition: startPos,
      endPosition: endPos,
    });
  }, [title, width, height, scale, includeLegend, pathVisibility, startPos, endPos, onExport]);

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [onClose]);

  const positionError =
    startPos >= endPos
      ? 'Start position must be less than end position'
      : startPos < fullRangeStart || endPos > fullRangeEnd
        ? `Positions must be within ${fullRangeStart.toLocaleString()} – ${fullRangeEnd.toLocaleString()}`
        : null;

  return (
    <div className="export-modal-backdrop" onClick={onClose}>
      <div className="export-modal" onClick={(e) => e.stopPropagation()}>
        <div className="export-modal-header">
          <h3>Export Image</h3>
          <button className="export-modal-close" onClick={onClose}>
            <Icon path={mdiClose} size={0.8} />
          </button>
        </div>

        <div className="export-modal-body">
          {/* Title */}
          <div className="export-modal-field">
            <label className="export-modal-label">Title</label>
            <input
              type="text"
              className="export-modal-input"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Image title"
            />
          </div>

          {/* Dimensions */}
          <div className="export-modal-field">
            <label className="export-modal-label">Dimensions (px)</label>
            <div className="export-modal-dimensions-row">
              <div className="export-modal-dim-input">
                <span className="export-modal-dim-label">W</span>
                <input
                  type="number"
                  className="export-modal-input export-modal-input-number"
                  value={width}
                  min={100}
                  onChange={(e) => handleWidthChange(parseInt(e.target.value) || 0)}
                />
              </div>
              <button
                className={`export-modal-aspect-lock ${lockAspect ? 'locked' : ''}`}
                onClick={() => setLockAspect(prev => !prev)}
                title={lockAspect ? 'Unlock aspect ratio' : 'Lock aspect ratio'}
              >
                <Icon path={lockAspect ? mdiLink : mdiLinkOff} size={0.7} />
              </button>
              <div className="export-modal-dim-input">
                <span className="export-modal-dim-label">H</span>
                <input
                  type="number"
                  className="export-modal-input export-modal-input-number"
                  value={height}
                  min={100}
                  onChange={(e) => handleHeightChange(parseInt(e.target.value) || 0)}
                />
              </div>
            </div>
          </div>

          {/* Scale */}
          <div className="export-modal-field">
            <label className="export-modal-label">Scale Factor</label>
            <div className="export-modal-scale-row">
              {SCALE_OPTIONS.map((s) => (
                <button
                  key={s}
                  className={`export-modal-scale-btn ${scale === s ? 'active' : ''}`}
                  onClick={() => setScale(s)}
                >
                  {s}x
                </button>
              ))}
            </div>
            <span className="export-modal-hint">
              Output: {(width * scale).toLocaleString()} × {(height * scale).toLocaleString()} px
            </span>
          </div>

          {/* Position Range */}
          <div className="export-modal-field">
            <label className="export-modal-label">Position Range</label>
            <div className="export-modal-range-row">
              <div className="export-modal-dim-input">
                <span className="export-modal-dim-label">Start</span>
                <input
                  type="number"
                  className="export-modal-input export-modal-input-number"
                  value={startPos}
                  min={fullRangeStart}
                  max={fullRangeEnd}
                  onChange={(e) => setStartPos(parseInt(e.target.value) || 0)}
                />
              </div>
              <span className="export-modal-range-separator">–</span>
              <div className="export-modal-dim-input">
                <span className="export-modal-dim-label">End</span>
                <input
                  type="number"
                  className="export-modal-input export-modal-input-number"
                  value={endPos}
                  min={fullRangeStart}
                  max={fullRangeEnd}
                  onChange={(e) => setEndPos(parseInt(e.target.value) || 0)}
                />
              </div>
            </div>
            {positionError && (
              <span className="export-modal-error">{positionError}</span>
            )}
          </div>

          {/* Legend toggle */}
          <div className="export-modal-field">
            <label className="export-modal-checkbox-row">
              <input
                type="checkbox"
                checked={includeLegend}
                onChange={(e) => setIncludeLegend(e.target.checked)}
              />
              <span>Include legend</span>
            </label>
          </div>

          {/* Path overlays */}
          {pathOverlays.length > 0 && (
            <div className="export-modal-field">
              <label className="export-modal-label">Path Overlays</label>
              <div className="export-modal-path-toggles">
                {pathOverlays.map((p) => (
                  <label key={p.label} className="export-modal-checkbox-row">
                    <input
                      type="checkbox"
                      checked={pathVisibility[p.label] ?? false}
                      onChange={() => handlePathToggle(p.label)}
                    />
                    <span>{p.label}</span>
                  </label>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="export-modal-footer">
          <button className="export-modal-btn export-modal-btn-cancel" onClick={onClose}>
            Cancel
          </button>
          <button
            className="export-modal-btn export-modal-btn-export"
            onClick={handleExport}
            disabled={!!positionError || width < 100 || height < 100}
          >
            Export
          </button>
        </div>
      </div>
    </div>
  );
};

export default ExportModal;
