import React, { useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { open, save } from '@tauri-apps/plugin-dialog';
import Icon from '@mdi/react';
import { mdiContentCopy, mdiCog, mdiClose } from '@mdi/js';
import SystemSettings, { type AdapterInfo } from './SystemSettings';
import './ImputePage.css';

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

const ImputePage: React.FC = () => {
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
  const [result, setResult] = useState<ImputeResult | null>(null);
  const [gpuAdapters, setGpuAdapters] = useState<AdapterInfo[] | null>(null);
  const [showSystemSettings, setShowSystemSettings] = useState<boolean>(false);

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
    } catch (err) {
      console.error('Failed to copy to clipboard:', err);
      const textArea = document.createElement('textarea');
      textArea.value = text;
      document.body.appendChild(textArea);
      textArea.focus();
      textArea.select();
      try {
        document.execCommand('copy');
      } catch (fallbackErr) {
        console.error('Fallback copy failed:', fallbackErr);
      }
      document.body.removeChild(textArea);
    }
  };

  const hasNvidiaGpu = () => {
    if (!gpuAdapters || gpuAdapters.length === 0) return false;
    return gpuAdapters.some(adapter =>
      adapter.name.toLowerCase().includes('nvidia') ||
      adapter.backend.toLowerCase().includes('cuda') ||
      adapter.vendor === 0x10de
    );
  };

  const handleGpuInfoChange = (adapters: AdapterInfo[] | null) => {
    setGpuAdapters(adapters);
    if (!adapters || !hasNvidiaGpuInList(adapters)) {
      if (model !== 'knn') {
        setModel('knn');
      }
    }
  };

  const hasNvidiaGpuInList = (adapters: AdapterInfo[]) => {
    return adapters.some(adapter =>
      adapter.name.toLowerCase().includes('nvidia') ||
      adapter.backend.toLowerCase().includes('cuda') ||
      adapter.vendor === 0x10de
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

  const selectOutputFile = async () => {
    try {
      const selected = await save({
        title: 'Select Output Location',
        defaultPath: outputPath || 'output_imputed.bed',
        filters: [
          { name: 'BED Files', extensions: ['bed'] },
          { name: 'All Files', extensions: ['*'] },
        ],
      });
      if (selected) {
        setOutputPath(selected);
      }
    } catch (error) {
      console.error('Error selecting output file:', error);
      alert(`Error opening save dialog: ${error}`);
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
    } catch (error) {
      setResult({
        success: false,
        message: `Error: ${error}`,
      });
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
    setCollapse(false);
    setVerbose(false);
    setHmm(false);
    setDiploid(false);
    setCollapseBed(false);
    setResult(null);
  };

  return (
    <div className="impute-page">
      <div className="impute-page-inner">
        <div className="impute-page-header">
          <div>
            <h2>ML Imputation Tool</h2>
            <p className="impute-page-subtitle">Configure and run haplotype imputation models</p>
          </div>
          <button
            className="settings-icon-button"
            onClick={() => setShowSystemSettings(true)}
            title="System Specs"
          >
            <Icon path={mdiCog} size={1} />
          </button>
        </div>

        {showSystemSettings && (
          <div className="settings-modal-overlay" onClick={() => setShowSystemSettings(false)}>
            <div className="settings-modal" onClick={(e) => e.stopPropagation()}>
              <div className="settings-modal-header">
                <h3>System Specs</h3>
                <button
                  className="settings-modal-close"
                  onClick={() => setShowSystemSettings(false)}
                >
                  <Icon path={mdiClose} size={0.9} />
                </button>
              </div>
              <div className="settings-modal-body">
                <SystemSettings onGpuInfoChange={handleGpuInfoChange} />
              </div>
            </div>
          </div>
        )}

        <div className="impute-grid">
          {/* Left column: Input & Output */}
          <div className="impute-column">
            <div className="impute-card">
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
                      { name: 'PS4G Files (*.ps4g, *.ps4g.txt, *_ps4g.txt)', extensions: ['ps4g', 'txt'] },
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

            <div className="impute-card">
              <h3>Output</h3>

              <div className="input-group">
                <label>Output BED File:</label>
                <div className="file-input">
                  <input
                    type="text"
                    value={outputPath}
                    onChange={(e) => setOutputPath(e.target.value)}
                    placeholder="output_imputed.bed"
                    disabled={isRunning}
                  />
                  <button
                    onClick={selectOutputFile}
                    disabled={isRunning}
                  >
                    Browse
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Right column: Model Settings & Options */}
          <div className="impute-column">
            <div className="impute-card">
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

            <div className="impute-card">
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
          </div>
        </div>

        {/* Action buttons - full width below the grid */}
        <div className="impute-actions">
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

        {/* Results */}
        {result && (
          <div className={`result-card ${result.success ? 'success' : 'error'}`}>
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
                className="copy-button"
                title="Copy to clipboard"
              >
                <Icon path={mdiContentCopy} size={0.7} />
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
    </div>
  );
};

export default ImputePage;
