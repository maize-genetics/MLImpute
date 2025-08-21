import React, { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { platform, arch } from '@tauri-apps/plugin-os';
import './SystemSettings.css';

export type SystemInfo = {
  platform?: string;
  arch?: string;
};

export type AdapterInfo = {
  name: string;
  backend: string;
  device_type: string;
  vendor: number;
  device: number;
};

interface SystemSettingsProps {
  className?: string;
  onGpuInfoChange?: (gpuAdapters: AdapterInfo[] | null) => void;
}

const SystemSettings: React.FC<SystemSettingsProps> = ({ className, onGpuInfoChange }) => {
  const [systemInfo, setSystemInfo] = useState<SystemInfo>({});
  const [gpuAdapters, setGpuAdapters] = useState<AdapterInfo[] | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isExpanded, setIsExpanded] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const loadSystemInfo = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      // Get system info
      const [platformName, archName, adapters] = await Promise.all([
        platform(),
        arch(),
        invoke<AdapterInfo[]>('gpu_adapters')
      ]);

      setSystemInfo({
        platform: platformName,
        arch: archName
      });

      setGpuAdapters(adapters);
      
      // Notify parent component of GPU information changes
      if (onGpuInfoChange) {
        onGpuInfoChange(adapters);
      }
    } catch (err) {
      console.error('Error loading system info:', err);
      setError(`Failed to load system information: ${err}`);
      
      // Notify parent component of error (no GPU info available)
      if (onGpuInfoChange) {
        onGpuInfoChange(null);
      }
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadSystemInfo();
  }, []);

  const toggleExpanded = () => {
    setIsExpanded(!isExpanded);
  };

  return (
    <div className={`system-settings ${className || ''}`}>
      <div className="system-settings-header" onClick={toggleExpanded}>
        <h4>
          System Settings
          <span className={`expand-icon ${isExpanded ? 'expanded' : ''}`}>▼</span>
        </h4>
      </div>

      {isExpanded && (
        <div className="system-settings-content">
          {error ? (
            <div className="error-message">
              <p>{error}</p>
            </div>
          ) : isLoading ? (
            <div className="loading-message">
              <p>Loading system information...</p>
            </div>
          ) : (
            <div className="system-info">
              <div className="info-item">
                <span className="info-label">OS Platform:</span>
                <span className="info-value">{systemInfo.platform || "Unknown"}</span>
              </div>
              
              <div className="info-item">
                <span className="info-label">CPU Architecture:</span>
                <span className="info-value">{systemInfo.arch || "Unknown"}</span>
              </div>
              
              <div className="info-item">
                <span className="info-label">GPU Adapters:</span>
                <span className="info-value">{gpuAdapters ? gpuAdapters.length : 0}</span>
              </div>

              {gpuAdapters && gpuAdapters.length > 0 && (
                <div className="gpu-adapters">
                  {gpuAdapters.map((adapter, index) => (
                    <div key={`${adapter.name}-${index}`} className="gpu-adapter">
                      <div className="adapter-header">GPU {index + 1}</div>
                      <div className="info-item">
                        <span className="info-label">Name:</span>
                        <span className="info-value">{adapter.name}</span>
                      </div>
                      <div className="info-item">
                        <span className="info-label">Type:</span>
                        <span className="info-value">{adapter.device_type}</span>
                      </div>
                      <div className="info-item">
                        <span className="info-label">Backend:</span>
                        <span className="info-value">{adapter.backend}</span>
                      </div>
                      <div className="info-item">
                        <span className="info-label">Vendor ID:</span>
                        <span className="info-value">0x{adapter.vendor.toString(16)}</span>
                      </div>
                      <div className="info-item">
                        <span className="info-label">Device ID:</span>
                        <span className="info-value">0x{adapter.device.toString(16)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default SystemSettings;