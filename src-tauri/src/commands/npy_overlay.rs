use ndarray::Array2;
use ndarray_npy::ReadNpyExt;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

#[derive(Debug, Serialize, Deserialize)]
pub struct NpyOverlayResult {
    pub success: bool,
    pub true_paths: Vec<Vec<usize>>,
    pub predicted_paths: Vec<Vec<usize>>,
    pub num_positions: usize,
    pub is_diploid_true: bool,
    pub is_diploid_predicted: bool,
    pub error: Option<String>,
}

impl NpyOverlayResult {
    fn error(msg: String) -> Self {
        Self {
            success: false,
            true_paths: vec![],
            predicted_paths: vec![],
            num_positions: 0,
            is_diploid_true: false,
            is_diploid_predicted: false,
            error: Some(msg),
        }
    }
}

/// Try reading an .npy file as various numeric types, returning f64 values.
/// ndarray-npy requires the exact type match, so we try common types in order.
fn read_npy_as_f64(path: &Path) -> Result<Array2<f64>, String> {
    let try_read = |path: &Path| -> Result<Array2<f64>, String> {
        // Try f64 first (most common for matrix files)
        if let Ok(arr) = read_typed::<f64>(path) {
            return Ok(arr);
        }
        // Try f32
        if let Ok(arr) = read_typed::<f32>(path) {
            return Ok(arr.mapv(|v| v as f64));
        }
        // Try i64
        if let Ok(arr) = read_typed::<i64>(path) {
            return Ok(arr.mapv(|v| v as f64));
        }
        // Try i32
        if let Ok(arr) = read_typed::<i32>(path) {
            return Ok(arr.mapv(|v| v as f64));
        }
        // Try i16
        if let Ok(arr) = read_typed::<i16>(path) {
            return Ok(arr.mapv(|v| v as f64));
        }
        // Try u8
        if let Ok(arr) = read_typed::<u8>(path) {
            return Ok(arr.mapv(|v| v as f64));
        }
        // Try i8
        if let Ok(arr) = read_typed::<i8>(path) {
            return Ok(arr.mapv(|v| v as f64));
        }
        // Try u64
        if let Ok(arr) = read_typed::<u64>(path) {
            return Ok(arr.mapv(|v| v as f64));
        }
        // Try u32
        if let Ok(arr) = read_typed::<u32>(path) {
            return Ok(arr.mapv(|v| v as f64));
        }
        // Try u16
        if let Ok(arr) = read_typed::<u16>(path) {
            return Ok(arr.mapv(|v| v as f64));
        }

        Err(format!(
            "Unsupported .npy dtype in file: {}",
            path.display()
        ))
    };

    try_read(path)
}

fn read_typed<T>(path: &Path) -> Result<Array2<T>, String>
where
    T: ndarray_npy::ReadableElement,
{
    let file =
        File::open(path).map_err(|e| format!("Failed to open {}: {}", path.display(), e))?;
    let reader = BufReader::new(file);
    Array2::<T>::read_npy(reader).map_err(|e| format!("Failed to read npy: {}", e))
}

/// Extract path columns from a 2D array, converting values to usize gamete indices.
fn extract_path_columns(arr: &Array2<f64>, start_col: usize) -> Vec<Vec<usize>> {
    let ncols = arr.ncols();
    let mut paths = Vec::new();
    for col_idx in start_col..ncols {
        let col: Vec<usize> = arr
            .column(col_idx)
            .iter()
            .map(|&v| v.round() as usize)
            .collect();
        paths.push(col);
    }
    paths
}

#[tauri::command]
pub async fn load_npy_overlay(
    matrix_path: Option<String>,
    predictions_path: Option<String>,
    expected_num_positions: usize,
    expected_num_gametes: usize,
) -> Result<NpyOverlayResult, String> {
    let mut true_paths: Vec<Vec<usize>> = vec![];
    let mut predicted_paths: Vec<Vec<usize>> = vec![];
    let mut is_diploid_true = false;
    let mut is_diploid_predicted = false;
    let mut num_positions = expected_num_positions;

    if matrix_path.is_none() && predictions_path.is_none() {
        return Ok(NpyOverlayResult::error(
            "At least one file path (matrix or predictions) must be provided".to_string(),
        ));
    }

    // Load matrix file and extract true paths from label columns
    if let Some(ref mpath) = matrix_path {
        let path = Path::new(mpath);
        if !path.exists() {
            return Ok(NpyOverlayResult::error(format!(
                "Matrix file not found: {}",
                mpath
            )));
        }

        let matrix = read_npy_as_f64(path)?;
        let (nrows, ncols) = (matrix.nrows(), matrix.ncols());

        if nrows != expected_num_positions {
            return Ok(NpyOverlayResult::error(format!(
                "Matrix has {} rows but expected {} positions (from PS4G chromosome data)",
                nrows, expected_num_positions
            )));
        }

        let label_cols = ncols.saturating_sub(expected_num_gametes);
        if label_cols == 0 {
            return Ok(NpyOverlayResult::error(format!(
                "Matrix has {} columns but expected at least {} gamete columns + label columns",
                ncols, expected_num_gametes
            )));
        }

        if label_cols > 2 {
            return Ok(NpyOverlayResult::error(format!(
                "Matrix has {} label columns (expected 1 for haploid or 2 for diploid). \
                 Total cols: {}, gametes: {}",
                label_cols, ncols, expected_num_gametes
            )));
        }

        true_paths = extract_path_columns(&matrix, expected_num_gametes);
        is_diploid_true = true_paths.len() == 2;
        num_positions = nrows;
    }

    // Load predictions file
    if let Some(ref ppath) = predictions_path {
        let path = Path::new(ppath);
        if !path.exists() {
            return Ok(NpyOverlayResult::error(format!(
                "Predictions file not found: {}",
                ppath
            )));
        }

        let preds = read_npy_as_f64(path)?;
        let (nrows, ncols) = (preds.nrows(), preds.ncols());

        if nrows != expected_num_positions {
            return Ok(NpyOverlayResult::error(format!(
                "Predictions file has {} rows but expected {} positions (from PS4G chromosome data)",
                nrows, expected_num_positions
            )));
        }

        if ncols == 0 || ncols > 2 {
            return Ok(NpyOverlayResult::error(format!(
                "Predictions file has {} columns (expected 1 for haploid or 2 for diploid)",
                ncols
            )));
        }

        predicted_paths = extract_path_columns(&preds, 0);
        is_diploid_predicted = predicted_paths.len() == 2;
        num_positions = nrows;
    }

    Ok(NpyOverlayResult {
        success: true,
        true_paths,
        predicted_paths,
        num_positions,
        is_diploid_true,
        is_diploid_predicted,
        error: None,
    })
}
