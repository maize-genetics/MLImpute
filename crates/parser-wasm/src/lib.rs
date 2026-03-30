use js_sys::Function;
use parser_core::types::*;
use std::io::Cursor;
use wasm_bindgen::prelude::*;

// ============================================================================
// PS4G
// ============================================================================

/// Opaque handle holding parsed PS4G data in WASM memory.
/// Allows chromosome matrix retrieval without re-parsing.
#[wasm_bindgen]
pub struct PS4GFileHandle {
    cached: CachedPS4GData,
}

#[wasm_bindgen]
impl PS4GFileHandle {
    /// Build a chromosome matrix from the cached data.
    pub fn get_chromosome_matrix(&self, chromosome: &str) -> Result<JsValue, JsValue> {
        let result = parser_core::build_chromosome_matrix(&self.cached, chromosome)
            .map_err(|e| JsValue::from_str(&e))?;
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Encode a chromosome matrix as a compact base64 binary payload.
    pub fn get_chromosome_matrix_binary(&self, chromosome: &str) -> Result<JsValue, JsValue> {
        let matrix = parser_core::build_chromosome_matrix(&self.cached, chromosome)
            .map_err(|e| JsValue::from_str(&e))?;
        let binary = parser_core::encode_matrix_binary(&matrix);
        serde_wasm_bindgen::to_value(&binary).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

/// Parse a PS4G file from raw bytes.
/// Returns a `PS4GParseResult` as a JS object. The `PS4GFileHandle` for
/// subsequent chromosome matrix calls is returned via `get_ps4g_handle`
/// on the same data.
#[wasm_bindgen]
pub fn parse_ps4g_file(
    data: &[u8],
    file_size: u64,
    on_progress: Option<Function>,
) -> Result<JsValue, JsValue> {
    let cursor = Cursor::new(data);
    let (result, _cached) = parser_core::parse_ps4g(
        cursor,
        Some(file_size),
        |p: PS4GProgress| {
            if let Some(ref f) = on_progress {
                if let Ok(val) = serde_wasm_bindgen::to_value(&p) {
                    let _ = f.call1(&JsValue::NULL, &val);
                }
            }
        },
    )
    .map_err(|e| JsValue::from_str(&e))?;

    serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Parse a PS4G file and return an opaque handle for chromosome matrix queries.
#[wasm_bindgen]
pub fn parse_ps4g_to_handle(
    data: &[u8],
    file_size: u64,
    on_progress: Option<Function>,
) -> Result<PS4GFileHandle, JsValue> {
    let cursor = Cursor::new(data);
    let (_result, cached) = parser_core::parse_ps4g(
        cursor,
        Some(file_size),
        |p: PS4GProgress| {
            if let Some(ref f) = on_progress {
                if let Ok(val) = serde_wasm_bindgen::to_value(&p) {
                    let _ = f.call1(&JsValue::NULL, &val);
                }
            }
        },
    )
    .map_err(|e| JsValue::from_str(&e))?;

    Ok(PS4GFileHandle { cached })
}

// ============================================================================
// BED
// ============================================================================

/// Opaque handle holding raw BED file bytes in WASM memory.
/// BED parsing does not cache intermediate data the way PS4G does, so the
/// handle stores the raw bytes for re-reading during matrix construction.
#[wasm_bindgen]
pub struct BEDFileHandle {
    data: Vec<u8>,
    file_size: u64,
}

#[wasm_bindgen]
impl BEDFileHandle {
    /// Build a chromosome matrix for the given chromosome.
    pub fn get_chromosome_matrix(
        &self,
        chromosome: &str,
        on_progress: Option<Function>,
    ) -> Result<JsValue, JsValue> {
        let cursor = Cursor::new(&self.data);
        let result = parser_core::build_bed_chromosome_matrix(
            cursor,
            Some(self.file_size),
            chromosome,
            |p: BEDMatrixProgress| {
                if let Some(ref f) = on_progress {
                    if let Ok(val) = serde_wasm_bindgen::to_value(&p) {
                        let _ = f.call1(&JsValue::NULL, &val);
                    }
                }
            },
        )
        .map_err(|e| JsValue::from_str(&e))?;

        serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

/// Parse a BED file from raw bytes.
#[wasm_bindgen]
pub fn parse_bed_file(
    data: &[u8],
    file_size: u64,
    on_progress: Option<Function>,
) -> Result<JsValue, JsValue> {
    let cursor = Cursor::new(data);
    let result = parser_core::parse_bed(
        cursor,
        Some(file_size),
        |p: BEDProgress| {
            if let Some(ref f) = on_progress {
                if let Ok(val) = serde_wasm_bindgen::to_value(&p) {
                    let _ = f.call1(&JsValue::NULL, &val);
                }
            }
        },
    )
    .map_err(|e| JsValue::from_str(&e))?;

    serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Parse a BED file and return an opaque handle for chromosome matrix queries.
#[wasm_bindgen]
pub fn parse_bed_to_handle(
    data: &[u8],
    file_size: u64,
    on_progress: Option<Function>,
) -> Result<BEDFileHandle, JsValue> {
    let cursor = Cursor::new(data);
    // Run the initial parse to validate the data and report progress
    let _result = parser_core::parse_bed(
        cursor,
        Some(file_size),
        |p: BEDProgress| {
            if let Some(ref f) = on_progress {
                if let Ok(val) = serde_wasm_bindgen::to_value(&p) {
                    let _ = f.call1(&JsValue::NULL, &val);
                }
            }
        },
    )
    .map_err(|e| JsValue::from_str(&e))?;

    Ok(BEDFileHandle {
        data: data.to_vec(),
        file_size,
    })
}

// ============================================================================
// NPY Overlay
// ============================================================================

/// Load NPY overlay data from raw bytes.
/// Pass empty `Uint8Array` (length 0) for files that should be skipped.
#[wasm_bindgen]
pub fn load_npy_overlay(
    observed_data: &[u8],
    predictions_data: &[u8],
    expected_num_positions: usize,
    expected_num_gametes: usize,
) -> Result<JsValue, JsValue> {
    let observed_reader: Option<Cursor<&[u8]>> = if observed_data.is_empty() {
        None
    } else {
        Some(Cursor::new(observed_data))
    };

    let predictions_reader: Option<Cursor<&[u8]>> = if predictions_data.is_empty() {
        None
    } else {
        Some(Cursor::new(predictions_data))
    };

    let result = parser_core::load_npy_overlay_from_readers(
        observed_reader,
        predictions_reader,
        expected_num_positions,
        expected_num_gametes,
    )
    .map_err(|e| JsValue::from_str(&e))?;

    serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
}
