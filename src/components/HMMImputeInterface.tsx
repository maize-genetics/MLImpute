import React, { useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { open } from '@tauri-apps/plugin-dialog';
import './HMMImputeInterface.css';

interface HmmImputeArgs {
  input_path: string;
  output_bed: string;
  global_weights: string;
  ps4g_file: string;
  diploid: boolean;
}

interface HmmImputeResult {
  success: boolean;
  message: string;
  output_file?: string;
}

const HMMImputeInterface: React.FC = () => {
  const [inputPath, setInputPath] = useState<string>('');
  const [outputBed, setOutputBed] = useState<string>('imputed_path.bed');
  const [globalWeights, setGlobalWeights] = useState<string>('');
  const [ps4gFile, setPs4gFile] = useState<string>('');
  const [diploid, setDiploid] = useState<boolean>(false);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [result, setResult] = useState<HmmImputeResult | null>(null);

  const selectFile = async (setter: (path: string) => void, title: string, filters?: any[]) => {
    console.log('selectFile called with:', { title, filters });
    try {
      console.log('Opening file dialog...');
      const selected = await open({
        title,
        multiple: false,
        filters: filters || [
          { name: 'All Files', extensions: ['*'] }
        ]
      });
      
      console.log('Dialog result:', selected);
      
      if (selected && typeof selected === 'string') {
        console.log('Setting file path:', selected);
        setter(selected);
      } else {
        console.log('No file selected or invalid selection');
      }
    } catch (error) {
      console.error('Error selecting file:', error);
      alert(`Error opening file dialog: ${error}`);
    }
  };

  const runImputation = async () => {
    if (!inputPath || !globalWeights || !ps4gFile) {
      alert('Please select all required input files');
      return;
    }

    setIsRunning(true);
    setResult(null);

    try {
      const args: HmmImputeArgs = {
        input_path: inputPath,
        output_bed: outputBed,
        global_weights: globalWeights,
        ps4g_file: ps4gFile,
        diploid
      };

      const response = await invoke<HmmImputeResult>('run_hmm_imputation', { args });
      setResult(response);
    } catch (error) {
      setResult({
        success: false,
        message: `Error: ${error}`,
        output_file: undefined
      });
    } finally {
      setIsRunning(false);
    }
  };

  const resetForm = () => {
    setInputPath('');
    setOutputBed('imputed_path.bed');
    setGlobalWeights('');
    setPs4gFile('');
    setDiploid(false);
    setResult(null);
  };

  return (
    <div className="hmm-impute-container">
      <div className="form-section">
        <h2>Input Files</h2>
        
        <div className="file-input-group">
          <label htmlFor="input-path">Input Path (Required):</label>
          <div className="file-input-row">
            <input
              id="input-path"
              type="text"
              value={inputPath}
              onChange={(e) => setInputPath(e.target.value)}
              placeholder="Select input file (.npy)"
              readOnly
            />
            <button
              type="button"
              onClick={() => selectFile(setInputPath, 'Select Input File', [
                { name: 'NumPy Arrays', extensions: ['npy'] },
                { name: 'All Files', extensions: ['*'] }
              ])}
              disabled={isRunning}
            >
              Browse
            </button>
          </div>
        </div>

        <div className="file-input-group">
          <label htmlFor="global-weights">Global Weights (Required):</label>
          <div className="file-input-row">
            <input
              id="global-weights"
              type="text"
              value={globalWeights}
              onChange={(e) => setGlobalWeights(e.target.value)}
              placeholder="Select global weights file (.npy)"
              readOnly
            />
            <button
              type="button"
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

        <div className="file-input-group">
          <label htmlFor="ps4g-file">PS4G File (Required):</label>
          <div className="file-input-row">
            <input
              id="ps4g-file"
              type="text"
              value={ps4gFile}
              onChange={(e) => setPs4gFile(e.target.value)}
              placeholder="Select PS4G file"
              readOnly
            />
            <button
              type="button"
              onClick={() => selectFile(setPs4gFile, 'Select PS4G File')}
              disabled={isRunning}
            >
              Browse
            </button>
          </div>
        </div>

        <h2>Output Settings</h2>
        
        <div className="file-input-group">
          <label htmlFor="output-bed">Output BED File:</label>
          <input
            id="output-bed"
            type="text"
            value={outputBed}
            onChange={(e) => setOutputBed(e.target.value)}
            placeholder="Output file path"
            disabled={isRunning}
          />
        </div>

        <div className="checkbox-group">
          <label>
            <input
              type="checkbox"
              checked={diploid}
              onChange={(e) => setDiploid(e.target.checked)}
              disabled={isRunning}
            />
            Use diploid mode
          </label>
        </div>

        <div className="button-group">
          <button
            onClick={runImputation}
            disabled={isRunning || !inputPath || !globalWeights || !ps4gFile}
            className="run-button"
          >
            {isRunning ? 'Running...' : 'Run HMM Imputation'}
          </button>
          <button
            onClick={resetForm}
            disabled={isRunning}
            className="reset-button"
          >
            Reset
          </button>
        </div>
      </div>

      {result && (
        <div className={`result-section ${result.success ? 'success' : 'error'}`}>
          <h3>{result.success ? 'Success' : 'Error'}</h3>
          <div className="result-message">
            <pre>{result.message}</pre>
          </div>
          {result.output_file && (
            <div className="output-file">
              <strong>Output file created:</strong> {result.output_file}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default HMMImputeInterface;