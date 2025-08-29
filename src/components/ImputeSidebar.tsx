import React, { useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { open } from '@tauri-apps/plugin-dialog';
import SystemSettings, { type AdapterInfo } from './SystemSettings';
import './ImputeSidebar.css';

interface ImputeArgs {
  input_path: string;
  output_path: string;
  model: string;
  weight?: string;
  collapse?: boolean;
  verbose?: boolean;
  global_weights?: string;
  hmm?: boolean;
  diploid?: boolean;
  collapse_bed?: boolean;
}

interface ImputeResult {
  success: boolean;
  message: string;
  output_file?: string;
  execution_time?: number;
  visualization_data?: string;
}

interface BedProcessResult {
  success: boolean;
  message: string;
  visualization_data?: string;
  error?: string;
}

interface VisualizationData {
  status: string;
  matrix?: {
    data: string;
    shape: number[];
    dtype: string;
  };
  row_labels?: string[];
  col_labels?: string[];
  metadata?: any;
  error?: string;
}

interface ImputeSidebarProps {
  onResults?: (results: ImputeResult) => void;
  onVisualizationData?: (data: VisualizationData) => void;
}

const ImputeSidebar: React.FC<ImputeSidebarProps> = ({ onResults }) => {
  const [inputPath, setInputPath] = useState<string>('');
  const [outputPath, setOutputPath] = useState<string>('output_imputed.bed');
  const [model, setModel] = useState<string>('knn');
  const [weight, setWeight] = useState<string>('global');
  const [globalWeights, setGlobalWeights] = useState<string>('');
  const [bedFilePath, setBedFilePath] = useState<string>('');
  const [collapse, setCollapse] = useState<boolean>(false);
  const [verbose, setVerbose] = useState<boolean>(false);
  const [hmm, setHmm] = useState<boolean>(false);
  const [diploid, setDiploid] = useState<boolean>(false);
  const [collapseBed, setCollapseBed] = useState<boolean>(false);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [result, setResult] = useState<ImputeResult | null>(null);
  const [gpuAdapters, setGpuAdapters] = useState<AdapterInfo[] | null>(null);

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      console.log('Copied to clipboard:', text);
    } catch (err) {
      console.error('Failed to copy to clipboard:', err);
      // Fallback for older browsers
      const textArea = document.createElement('textarea');
      textArea.value = text;
      document.body.appendChild(textArea);
      textArea.focus();
      textArea.select();
      try {
        document.execCommand('copy');
        console.log('Copied to clipboard (fallback):', text);
      } catch (fallbackErr) {
        console.error('Fallback copy failed:', fallbackErr);
      }
      document.body.removeChild(textArea);
    }
  };

  // Check if Nvidia GPU is present
  const hasNvidiaGpu = () => {
    if (!gpuAdapters || gpuAdapters.length === 0) return false;
    
    return gpuAdapters.some(adapter => 
      adapter.name.toLowerCase().includes('nvidia') || 
      adapter.backend.toLowerCase().includes('cuda') ||
      adapter.vendor === 0x10de  // Nvidia's vendor ID
    );
  };

  // Handle GPU info changes from SystemSettings
  const handleGpuInfoChange = (adapters: AdapterInfo[] | null) => {
    setGpuAdapters(adapters);
    
    // Reset model to KNN if no Nvidia GPU is detected and current model is not KNN
    if (!adapters || !hasNvidiaGpuInList(adapters)) {
      if (model !== 'knn') {
        setModel('knn');
      }
    }
  };

  // Helper function to check Nvidia GPU in a given list
  const hasNvidiaGpuInList = (adapters: AdapterInfo[]) => {
    return adapters.some(adapter => 
      adapter.name.toLowerCase().includes('nvidia') || 
      adapter.backend.toLowerCase().includes('cuda') ||
      adapter.vendor === 0x10de  // Nvidia's vendor ID
    );
  };

  const selectFile = async (
    setter: (path: string) => void, 
    title: string, 
    filters?: { name: string; extensions: string[] }[]
  ) => {
    try {
      const selected = await open({
        title,
        multiple: false,
        filters: filters || [{ name: 'All Files', extensions: ['*'] }]
      });
      
      if (selected && typeof selected === 'string') {
        setter(selected);
      }
    } catch (error) {
      console.error('Error selecting file:', error);
      alert(`Error opening file dialog: ${error}`);
    }
  };

  const runImputation = async () => {
    if (!inputPath) {
      alert('Please select an input file');
      return;
    }

    setIsRunning(true);
    setResult(null);

    try {
      const args: ImputeArgs = {
        input_path: inputPath,
        output_path: outputPath,
        model,
        weight: weight || undefined,
        collapse: collapse || undefined,
        verbose: verbose || undefined,
        global_weights: globalWeights || undefined,
        hmm: hmm || undefined,
        diploid: diploid || undefined,
        collapse_bed: collapseBed || undefined,
      };

      const response = await invoke<ImputeResult>('run_python_imputation', { args });
      setResult(response);
      
      if (onResults) {
        onResults(response);
      }
    } catch (error) {
      const errorResult: ImputeResult = {
        success: false,
        message: `Error: ${error}`,
      };
      setResult(errorResult);
      
      if (onResults) {
        onResults(errorResult);
      }
    } finally {
      setIsRunning(false);
    }
  };

  const processBedFile = async () => {
    if (!bedFilePath) {
      alert('Please select a BED file');
      return;
    }

    setIsRunning(true);
    setResult(null);

    try {
      const response = await invoke<BedProcessResult>('process_bed_file', { filePath: bedFilePath });
      
      if (response.success && response.visualization_data) {
        // Parse the nested JSON structure
        const vizResult = JSON.parse(response.visualization_data);
        
        // Merge highlight data into visualization data metadata for frontend access
        const vizDataWithHighlights = {
          ...vizResult.visualization_data,
          metadata: {
            ...vizResult.visualization_data.metadata,
            type: 'bed_file_visualization',
            highlights: vizResult.highlight_data
          }
        };
        
        // Convert back to string for the same flow as regular imputation
        const bedResult: ImputeResult = {
          success: true,
          message: response.message,
          visualization_data: JSON.stringify(vizDataWithHighlights),
        };
        setResult(bedResult);
        
        // Use the same flow as regular imputation - this will handle validation
        if (onResults) {
          onResults(bedResult);
        }
      } else {
        const errorResult: ImputeResult = {
          success: false,
          message: response.message || 'Failed to process BED file',
        };
        setResult(errorResult);
        
        if (onResults) {
          onResults(errorResult);
        }
      }
    } catch (error) {
      const errorResult: ImputeResult = {
        success: false,
        message: `Error processing BED file: ${error}`,
      };
      setResult(errorResult);
      
      if (onResults) {
        onResults(errorResult);
      }
    } finally {
      setIsRunning(false);
    }
  };

  const resetForm = () => {
    setInputPath('');
    setOutputPath('output_imputed.bed');
    setModel('knn');
    setWeight('global');
    setGlobalWeights('');
    setBedFilePath('');
    setCollapse(false);
    setVerbose(false);
    setHmm(false);
    setDiploid(false);
    setCollapseBed(false);
    setResult(null);
  };

  return (
    <div className="impute-sidebar">
      <div className="sidebar-header">
        <h2>ML Imputation Tool</h2>
      </div>

      <SystemSettings onGpuInfoChange={handleGpuInfoChange} />
      <div className="form-section">
        <h3>Input</h3>
        
        <div className="input-group">
          <label>Input File (PS4G):</label>
          <div className="file-input">
            <input
              type="text"
              value={inputPath}
              onChange={(e) => setInputPath(e.target.value)}
              placeholder="Select PS4G file..."
              readOnly
            />
            <button
              onClick={() => selectFile(setInputPath, 'Select PS4G Input File', [
                { name: 'PS4G Files', extensions: ['ps4g'] },
                { name: 'All Files', extensions: ['*'] }
              ])}
              disabled={isRunning}
            >
              Browse
            </button>
          </div>
        </div>

        <div className="input-group">
          <label>Global Weights (optional):</label>
          <div className="file-input">
            <input
              type="text"
              value={globalWeights}
              onChange={(e) => setGlobalWeights(e.target.value)}
              placeholder="Select weights file..."
              readOnly
            />
            <button
              onClick={() => selectFile(setGlobalWeights, 'Select Global Weights File', [
                { name: 'NumPy Arrays', extensions: ['npy'] },
                { name: 'All Files', extensions: ['*'] }
              ])}
              disabled={isRunning}
            >
              Browse
            </button>
          </div>
        </div>
      </div>

      <div className="form-section">
        <h3>BED File Visualization</h3>
        <p className="section-description">Upload an existing BED file to visualize directly</p>
        
        <div className="input-group">
          <label>BED File:</label>
          <div className="file-input">
            <input
              type="text"
              value={bedFilePath}
              onChange={(e) => setBedFilePath(e.target.value)}
              placeholder="Select BED file..."
              readOnly
            />
            <button
              onClick={() => selectFile(setBedFilePath, 'Select BED File', [
                { name: 'BED Files', extensions: ['bed'] },
                { name: 'All Files', extensions: ['*'] }
              ])}
              disabled={isRunning}
            >
              Browse
            </button>
          </div>
        </div>

        <div className="button-group">
          <button
            onClick={processBedFile}
            disabled={isRunning || !bedFilePath}
            className="run-button bed-button"
          >
            {isRunning ? 'Processing...' : 'Load BED Visualization'}
          </button>
        </div>
      </div>

      <div className="divider">
        <span>OR</span>
      </div>

      <div className="form-section">
        <h3>Model Settings</h3>
        
        <div className="input-group">
          <label>Model:</label>
          <select value={model} onChange={(e) => setModel(e.target.value)} disabled={isRunning}>
            <option value="knn">KNN</option>
            {hasNvidiaGpu() && (
              <>
                <option value="mamba">BiMamba</option>
                <option value="modernbert">ModernBERT</option>
              </>
            )}
          </select>
        </div>

        <div className="input-group">
          <label>Weight Strategy:</label>
          <select value={weight} onChange={(e) => setWeight(e.target.value)} disabled={isRunning}>
            <option value="global">Global</option>
            <option value="unweighted">Unweighted</option>
          </select>
        </div>
      </div>

      <div className="form-section">
        <h3>Options</h3>
        
        <div className="checkbox-group">
          <label>
            <input
              type="checkbox"
              checked={collapse}
              onChange={(e) => setCollapse(e.target.checked)}
              disabled={isRunning}
            />
            Collapse gamete sets
          </label>
        </div>

        <div className="checkbox-group">
          <label>
            <input
              type="checkbox"
              checked={hmm}
              onChange={(e) => setHmm(e.target.checked)}
              disabled={isRunning}
            />
            Use HMM post-processing
          </label>
        </div>

        <div className="checkbox-group">
          <label>
            <input
              type="checkbox"
              checked={diploid}
              onChange={(e) => setDiploid(e.target.checked)}
              disabled={isRunning}
            />
            Diploid mode
          </label>
        </div>

        <div className="checkbox-group">
          <label>
            <input
              type="checkbox"
              checked={collapseBed}
              onChange={(e) => setCollapseBed(e.target.checked)}
              disabled={isRunning}
            />
            Collapse BED regions
          </label>
        </div>

        <div className="checkbox-group">
          <label>
            <input
              type="checkbox"
              checked={verbose}
              onChange={(e) => setVerbose(e.target.checked)}
              disabled={isRunning}
            />
            Verbose output
          </label>
        </div>
      </div>

      <div className="form-section">
        <h3>Output</h3>
        
        <div className="input-group">
          <label>Output BED File:</label>
          <input
            type="text"
            value={outputPath}
            onChange={(e) => setOutputPath(e.target.value)}
            placeholder="output_imputed.bed"
            disabled={isRunning}
          />
        </div>
      </div>

      <div className="button-group">
        <button
          onClick={runImputation}
          disabled={isRunning || !inputPath}
          className="run-button"
        >
          {isRunning ? 'Running...' : 'Run Imputation'}
        </button>
        <button
          onClick={resetForm}
          disabled={isRunning}
          className="reset-button"
        >
          Reset
        </button>
      </div>


      {result && (
        <div className={`result-section ${result.success ? 'success' : 'error'}`}>
          <h4 style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            {result.success ? 'Success' : 'Error'}
            <button
              onClick={() => {
                const details = [
                  result.success ? 'Success' : 'Error',
                  result.execution_time ? `Completed in ${result.execution_time.toFixed(2)}s` : '',
                  result.message,
                  result.output_file ? `Output: ${result.output_file}` : ''
                ].filter(Boolean).join('\n');
                copyToClipboard(details);
              }}
              style={{
                background: 'none',
                border: 'none',
                color: 'inherit',
                cursor: 'pointer',
                padding: '0.25rem',
                fontSize: '0.75rem',
                opacity: 0.7,
                borderRadius: '0.25rem'
              }}
              title="Copy to clipboard"
              onMouseOver={(e) => e.currentTarget.style.opacity = '1'}
              onMouseOut={(e) => e.currentTarget.style.opacity = '0.7'}
            >
              📋
            </button>
          </h4>
          {result.execution_time && (
            <p className="execution-time">
              Completed in {result.execution_time.toFixed(2)}s
            </p>
          )}
          <div className="result-message">
            <pre>{result.message}</pre>
          </div>
          {result.output_file && (
            <div className="output-file">
              <strong>Output:</strong> {result.output_file}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ImputeSidebar;