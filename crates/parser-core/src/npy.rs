use crate::types::NpyOverlayResult;
use ndarray::Array2;
use ndarray_npy::ReadNpyExt;
use std::io::Read;

/// Try reading an .npy file as various numeric types from a reader, returning f64 values.
pub fn read_npy_as_f64_from_reader(mut reader: impl Read) -> Result<Array2<f64>, String> {
    let mut buf = Vec::new();
    reader
        .read_to_end(&mut buf)
        .map_err(|e| format!("Failed to read npy data: {}", e))?;

    if let Ok(arr) = try_read_typed::<f64>(&buf) {
        return Ok(arr);
    }
    if let Ok(arr) = try_read_typed::<f32>(&buf) {
        return Ok(arr.mapv(|v| v as f64));
    }
    if let Ok(arr) = try_read_typed::<i64>(&buf) {
        return Ok(arr.mapv(|v| v as f64));
    }
    if let Ok(arr) = try_read_typed::<i32>(&buf) {
        return Ok(arr.mapv(|v| v as f64));
    }
    if let Ok(arr) = try_read_typed::<i16>(&buf) {
        return Ok(arr.mapv(|v| v as f64));
    }
    if let Ok(arr) = try_read_typed::<u8>(&buf) {
        return Ok(arr.mapv(|v| v as f64));
    }
    if let Ok(arr) = try_read_typed::<i8>(&buf) {
        return Ok(arr.mapv(|v| v as f64));
    }
    if let Ok(arr) = try_read_typed::<u64>(&buf) {
        return Ok(arr.mapv(|v| v as f64));
    }
    if let Ok(arr) = try_read_typed::<u32>(&buf) {
        return Ok(arr.mapv(|v| v as f64));
    }
    if let Ok(arr) = try_read_typed::<u16>(&buf) {
        return Ok(arr.mapv(|v| v as f64));
    }

    Err("Unsupported .npy dtype".to_string())
}

fn try_read_typed<T: ndarray_npy::ReadableElement>(buf: &[u8]) -> Result<Array2<T>, String> {
    let cursor = std::io::Cursor::new(buf);
    Array2::<T>::read_npy(cursor).map_err(|e| format!("{}", e))
}

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

/// Load NPY overlay data from readers. Pass `None` for either reader if that
/// file is not available.
pub fn load_npy_overlay_from_readers(
    observed_reader: Option<impl Read>,
    predictions_reader: Option<impl Read>,
    expected_num_positions: usize,
    expected_num_gametes: usize,
) -> Result<NpyOverlayResult, String> {
    let mut true_paths: Vec<Vec<usize>> = vec![];
    let mut predicted_paths: Vec<Vec<usize>> = vec![];
    let mut is_diploid_true = false;
    let mut is_diploid_predicted = false;
    let mut num_positions = expected_num_positions;

    if observed_reader.is_none() && predictions_reader.is_none() {
        return Ok(NpyOverlayResult::error(
            "At least one file (observed or predictions) must be provided".to_string(),
        ));
    }

    if let Some(reader) = observed_reader {
        let observed = read_npy_as_f64_from_reader(reader)?;
        let (nrows, ncols) = (observed.nrows(), observed.ncols());

        if nrows != expected_num_positions {
            return Ok(NpyOverlayResult::error(format!(
                "Observed file has {} rows but expected {} positions (from PS4G chromosome data)",
                nrows, expected_num_positions
            )));
        }

        let label_cols = ncols.saturating_sub(expected_num_gametes);
        if label_cols == 0 {
            return Ok(NpyOverlayResult::error(format!(
                "Observed file has {} columns but expected at least {} gamete columns + label columns",
                ncols, expected_num_gametes
            )));
        }

        if label_cols > 2 {
            return Ok(NpyOverlayResult::error(format!(
                "Observed file has {} label columns (expected 1 for haploid or 2 for diploid). \
                 Total cols: {}, gametes: {}",
                label_cols, ncols, expected_num_gametes
            )));
        }

        true_paths = extract_path_columns(&observed, expected_num_gametes);
        is_diploid_true = true_paths.len() == 2;
        num_positions = nrows;
    }

    if let Some(reader) = predictions_reader {
        let preds = read_npy_as_f64_from_reader(reader)?;
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
