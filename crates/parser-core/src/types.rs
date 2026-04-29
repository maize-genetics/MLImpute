use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// PS4G Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameteInfo {
    pub gamete: String,
    pub gamete_index: u32,
    pub read_count: u64,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PS4GDataRow {
    pub gamete_set: Vec<u32>,
    pub ref_contig: String,
    pub ref_pos_binned: u64,
    pub count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PS4GMetadata {
    pub version: Option<String>,
    pub command: Option<String>,
    pub total_unique_counts: Option<u64>,
    pub gametes: Vec<GameteInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PS4GSummary {
    pub total_rows: usize,
    pub unique_positions: usize,
    pub chromosomes: Vec<String>,
    pub chromosome_counts: HashMap<String, usize>,
    pub gamete_count: usize,
    pub position_range: HashMap<String, (u64, u64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PS4GProgress {
    pub rows_processed: usize,
    pub bytes_processed: u64,
    pub total_bytes: u64,
    pub percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PS4GParseResult {
    pub success: bool,
    pub metadata: PS4GMetadata,
    pub summary: PS4GSummary,
    pub data_preview: Vec<PS4GDataRow>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChromosomeMatrixResult {
    pub success: bool,
    pub chromosome: String,
    pub matrix: Vec<Vec<u32>>,
    pub positions: Vec<u64>,
    pub gamete_names: Vec<String>,
    pub num_gametes: usize,
    pub num_positions: usize,
    pub position_range: (u64, u64),
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChromosomeMatrixProgress {
    pub rows_processed: usize,
    pub chromosome: String,
    pub percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChromosomeMatrixBinaryResult {
    pub success: bool,
    pub chromosome: String,
    pub matrix_data: String,
    pub shape: [usize; 2],
    pub dtype: String,
    pub positions: Vec<u64>,
    pub gamete_names: Vec<String>,
    pub position_range: (u64, u64),
    pub error: Option<String>,
}

/// Platform-agnostic cached PS4G data (no filesystem types)
#[derive(Debug, Clone)]
pub struct CachedPS4GData {
    pub metadata: PS4GMetadata,
    pub chromosome_data: FxHashMap<String, FxHashMap<u64, FxHashMap<u32, u32>>>,
    pub chromosomes: Vec<String>,
    pub chromosome_counts: FxHashMap<String, usize>,
    pub position_ranges: FxHashMap<String, (u64, u64)>,
    pub total_rows: usize,
    pub unique_positions: usize,
}

// ============================================================================
// BED Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BEDDataRow {
    pub chrom: String,
    pub start: u64,
    pub end: u64,
    pub parent1: String,
    pub parent2: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParentStats {
    pub parent_id: String,
    pub regions_as_parent1: usize,
    pub regions_as_parent2: usize,
    pub total_regions: usize,
    pub coverage_bp_as_parent1: u64,
    pub coverage_bp_as_parent2: u64,
    pub total_coverage_bp: u64,
    pub chromosome_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BEDSummary {
    pub total_rows: usize,
    pub chromosomes: Vec<String>,
    pub chromosome_counts: HashMap<String, usize>,
    pub position_range: HashMap<String, (u64, u64)>,
    pub total_coverage_bp: u64,
    pub avg_region_size_bp: f64,
    pub unique_parents: Vec<String>,
    pub unique_parent_pairs: usize,
    pub parent_stats: Vec<ParentStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BEDProgress {
    pub rows_processed: usize,
    pub bytes_processed: u64,
    pub total_bytes: u64,
    pub percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BEDParseResult {
    pub success: bool,
    pub summary: BEDSummary,
    pub data_preview: Vec<BEDDataRow>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BEDRegion {
    pub start: u64,
    pub end: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BEDMatrixProgress {
    pub rows_processed: usize,
    pub chromosome: String,
    pub percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BEDChromosomeMatrixResult {
    pub success: bool,
    pub chromosome: String,
    pub matrix: Vec<Vec<u8>>,
    pub parent_names: Vec<String>,
    pub regions: Vec<BEDRegion>,
    pub num_parents: usize,
    pub num_regions: usize,
    pub parent1_path: Vec<usize>,
    pub parent2_path: Vec<usize>,
    pub error: Option<String>,
}

// ============================================================================
// NPY Overlay Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
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
    pub fn error(msg: String) -> Self {
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
