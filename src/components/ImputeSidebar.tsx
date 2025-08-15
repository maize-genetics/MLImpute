import React, { useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { open } from '@tauri-apps/plugin-dialog';
import { generateRandomMatrix, generateRandomHighlights } from './utils';
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
  demoMatrix?: {
    matrix: number[][];
    rowLabels: string[];
    colLabels: string[];
    highlights: Array<{col: string; row: string}>;
  };
}

interface ImputeSidebarProps {
  onResults?: (results: ImputeResult) => void;
  onVisualizationData?: (data: VisualizationData) => void;
}

const ImputeSidebar: React.FC<ImputeSidebarProps> = ({ onResults, onVisualizationData }) => {
  const [inputPath, setInputPath] = useState<string>('');
  const [outputPath, setOutputPath] = useState<string>('output_imputed.bed');
  const [model, setModel] = useState<string>('knn');
  const [weight, setWeight] = useState<string>('global');
  const [globalWeights, setGlobalWeights] = useState<string>('');
  const [collapse, setCollapse] = useState<boolean>(false);
  const [verbose, setVerbose] = useState<boolean>(false);
  const [hmm, setHmm] = useState<boolean>(false);
  const [diploid, setDiploid] = useState<boolean>(false);
  const [collapseBed, setCollapseBed] = useState<boolean>(false);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [isDemoRunning, setIsDemoRunning] = useState<boolean>(false);
  const [result, setResult] = useState<ImputeResult | null>(null);
  const [demoSamples, setDemoSamples] = useState<number>(50);
  const [demoPositions, setDemoPositions] = useState<number>(25000);

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

  const runDemo = () => {
    setIsDemoRunning(true);

    try {
      // Generate demo matrix data using local utilities
      const { matrix, rowLabels, colLabels } = generateRandomMatrix(demoSamples, demoPositions);
      const highlights = generateRandomHighlights(rowLabels, colLabels);

      const demoData: VisualizationData = {
        status: 'success',
        demoMatrix: {
          matrix,
          rowLabels,
          colLabels,
          highlights
        }
      };

      if (onVisualizationData) {
        onVisualizationData(demoData);
      }
    } catch (error) {
      console.error('Demo error:', error);
      alert(`Demo error: ${error}`);
    } finally {
      setIsDemoRunning(false);
    }
  };

  const resetForm = () => {
    setInputPath('');
    setOutputPath('output_imputed.bed');
    setModel('knn');
    setWeight('global');
    setGlobalWeights('');
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
        <h3>Model Settings</h3>
        
        <div className="input-group">
          <label>Model:</label>
          <select value={model} onChange={(e) => setModel(e.target.value)} disabled={isRunning}>
            <option value="knn">KNN</option>
            <option value="mamba">BiMamba</option>
            <option value="modernbert">ModernBERT</option>
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
          disabled={isRunning || !inputPath || isDemoRunning}
          className="run-button"
        >
          {isRunning ? 'Running...' : 'Run Imputation'}
        </button>
        <button
          onClick={resetForm}
          disabled={isRunning || isDemoRunning}
          className="reset-button"
        >
          Reset
        </button>
      </div>

      <div className="demo-section">
        <div className="demo-info">
          <h4>Quick Demo</h4>
          <p>Generate a demo visualization matrix with configurable dimensions.</p>
        </div>
        
        <div className="demo-config">
          <div className="input-group">
            <label>Samples:</label>
            <input
              type="number"
              value={demoSamples}
              onChange={(e) => setDemoSamples(Math.max(1, parseInt(e.target.value) || 1))}
              min="1"
              max="1000"
              disabled={isRunning || isDemoRunning}
            />
          </div>
          
          <div className="input-group">
            <label>Positions:</label>
            <input
              type="number"
              value={demoPositions}
              onChange={(e) => setDemoPositions(Math.max(1, parseInt(e.target.value) || 1))}
              min="1"
              max="50000"
              disabled={isRunning || isDemoRunning}
            />
          </div>
        </div>
        
        <button
          onClick={runDemo}
          disabled={isRunning || isDemoRunning}
          className="demo-button"
        >
          {isDemoRunning ? 'Generating Demo...' : `Run Demo (${demoSamples} × ${demoPositions})`}
        </button>
      </div>

      {result && (
        <div className={`result-section ${result.success ? 'success' : 'error'}`}>
          <h4>{result.success ? 'Success' : 'Error'}</h4>
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