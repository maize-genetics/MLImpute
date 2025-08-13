/**
 * Utilities for handling NumPy arrays transferred from Python backend
 */

export interface NumpyArray {
  data: string;
  shape: number[];
  dtype: string;
}

export interface VisualizationData {
  status: string;
  matrix?: NumpyArray;
  row_labels?: string[];
  col_labels?: string[];
  metadata?: Record<string, any>;
  error?: string;
}

/**
 * Decode a base64-encoded NumPy array to a JavaScript 2D array
 */
export function decodeNumpyArray(encoded: NumpyArray): number[][] {
  // Decode base64 to binary data
  const binaryString = atob(encoded.data);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  
  // Convert to appropriate typed array based on dtype
  let typedArray: ArrayLike<number>;
  switch (encoded.dtype) {
    case 'float64':
      typedArray = new Float64Array(bytes.buffer);
      break;
    case 'float32':
      typedArray = new Float32Array(bytes.buffer);
      break;
    case 'int32':
      typedArray = new Int32Array(bytes.buffer);
      break;
    case 'int16':
      typedArray = new Int16Array(bytes.buffer);
      break;
    case 'uint8':
      typedArray = new Uint8Array(bytes.buffer);
      break;
    default:
      console.warn(`Unsupported dtype: ${encoded.dtype}, falling back to Float64Array`);
      typedArray = new Float64Array(bytes.buffer);
  }
  
  // Reshape to 2D array
  if (encoded.shape.length !== 2) {
    throw new Error(`Expected 2D array, got ${encoded.shape.length}D array`);
  }
  
  const [rows, cols] = encoded.shape;
  const result: number[][] = [];
  
  for (let i = 0; i < rows; i++) {
    const row: number[] = [];
    for (let j = 0; j < cols; j++) {
      row.push(typedArray[i * cols + j]);
    }
    result.push(row);
  }
  
  return result;
}

/**
 * Decode a 1D NumPy array to a JavaScript array
 */
export function decode1DNumpyArray(encoded: NumpyArray): number[] {
  const binaryString = atob(encoded.data);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  
  let typedArray: ArrayLike<number>;
  switch (encoded.dtype) {
    case 'float64':
      typedArray = new Float64Array(bytes.buffer);
      break;
    case 'float32':
      typedArray = new Float32Array(bytes.buffer);
      break;
    case 'int32':
      typedArray = new Int32Array(bytes.buffer);
      break;
    case 'int16':
      typedArray = new Int16Array(bytes.buffer);
      break;
    case 'uint8':
      typedArray = new Uint8Array(bytes.buffer);
      break;
    default:
      typedArray = new Float64Array(bytes.buffer);
  }
  
  return Array.from(typedArray);
}

/**
 * Get array statistics for debugging/validation
 */
export function getArrayStats(arr: number[][]): {
  shape: [number, number];
  min: number;
  max: number;
  mean: number;
  hasNaN: boolean;
  hasInfinite: boolean;
} {
  const rows = arr.length;
  const cols = rows > 0 ? arr[0].length : 0;
  
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let count = 0;
  let hasNaN = false;
  let hasInfinite = false;
  
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      const val = arr[i][j];
      
      if (isNaN(val)) {
        hasNaN = true;
        continue;
      }
      
      if (!isFinite(val)) {
        hasInfinite = true;
        continue;
      }
      
      min = Math.min(min, val);
      max = Math.max(max, val);
      sum += val;
      count++;
    }
  }
  
  return {
    shape: [rows, cols] as [number, number],
    min: count > 0 ? min : NaN,
    max: count > 0 ? max : NaN,
    mean: count > 0 ? sum / count : NaN,
    hasNaN,
    hasInfinite,
  };
}

/**
 * Validate visualization data response
 */
export function validateVisualizationData(data: VisualizationData): {
  isValid: boolean;
  errors: string[];
} {
  const errors: string[] = [];
  
  if (!data) {
    return { isValid: false, errors: ['Data is null or undefined'] };
  }
  
  if (data.status === 'error') {
    errors.push(`Python backend error: ${data.error || 'Unknown error'}`);
  }
  
  if (data.matrix) {
    if (!data.matrix.data || !data.matrix.shape || !data.matrix.dtype) {
      errors.push('Matrix data is incomplete');
    }
    
    if (data.matrix.shape.length !== 2) {
      errors.push(`Expected 2D matrix, got ${data.matrix.shape.length}D`);
    }
    
    // Validate base64 data
    try {
      atob(data.matrix.data);
    } catch (e) {
      errors.push('Matrix data is not valid base64');
    }
  }
  
  return {
    isValid: errors.length === 0,
    errors,
  };
}