use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Mutex;
use tauri::{Emitter, State};

/// Cached data for a single PS4G file
/// Stores all parsed data to avoid re-parsing when switching chromosomes
#[derive(Debug, Clone)]
pub struct CachedPS4GFile {
    /// Path to the cached file (for cache invalidation)
    pub file_path: String,
    /// File modification time (for cache invalidation)
    pub modified_time: std::time::SystemTime,
    /// Parsed metadata
    pub metadata: PS4GMetadata,
    /// Per-chromosome position data: chromosome -> (position -> (gamete_idx -> count))
    pub chromosome_data: FxHashMap<String, FxHashMap<u64, FxHashMap<u32, u32>>>,
    /// Sorted chromosome list
    pub chromosomes: Vec<String>,
    /// Chromosome counts
    pub chromosome_counts: FxHashMap<String, usize>,
    /// Position ranges per chromosome
    pub position_ranges: FxHashMap<String, (u64, u64)>,
    /// Total rows
    pub total_rows: usize,
    /// Unique positions count
    pub unique_positions: usize,
}

/// Thread-safe cache wrapper for PS4G files
/// Allows caching multiple files simultaneously
pub struct PS4GCache {
    /// Map from file path to cached data
    pub cache: Mutex<FxHashMap<String, CachedPS4GFile>>,
}

impl PS4GCache {
    pub fn new() -> Self {
        PS4GCache {
            cache: Mutex::new(FxHashMap::default()),
        }
    }

    /// Check if file is cached and still valid (not modified since caching)
    pub fn get_cached(&self, file_path: &str) -> Option<CachedPS4GFile> {
        let cache = self.cache.lock().ok()?;
        let cached = cache.get(file_path)?;

        // Check if file has been modified since caching
        if let Ok(metadata) = std::fs::metadata(file_path) {
            if let Ok(modified) = metadata.modified() {
                if modified == cached.modified_time {
                    return Some(cached.clone());
                }
            }
        }
        None
    }

    /// Store parsed file data in cache
    pub fn store(&self, data: CachedPS4GFile) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.insert(data.file_path.clone(), data);
        }
    }

    /// Clear cache for a specific file
    #[allow(dead_code)]
    pub fn invalidate(&self, file_path: &str) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.remove(file_path);
        }
    }
}

impl Default for PS4GCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Gamete metadata from PS4G header
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameteInfo {
    pub gamete: String,
    pub gamete_index: u32,
    pub read_count: u64,
    pub weight: f64,
}

/// A single data row from the PS4G file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PS4GDataRow {
    pub gamete_set: Vec<u32>,
    pub ref_contig: String,
    pub ref_pos_binned: u64,
    pub count: u32,
}

/// Metadata extracted from PS4G file header
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PS4GMetadata {
    pub version: Option<String>,
    pub command: Option<String>,
    pub total_unique_counts: Option<u64>,
    pub gametes: Vec<GameteInfo>,
}

/// Summary statistics for the PS4G file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PS4GSummary {
    pub total_rows: usize,
    pub unique_positions: usize,
    pub chromosomes: Vec<String>,
    pub chromosome_counts: HashMap<String, usize>,
    pub gamete_count: usize,
    pub position_range: HashMap<String, (u64, u64)>,
}

/// Progress update sent during parsing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PS4GProgress {
    pub rows_processed: usize,
    pub bytes_processed: u64,
    pub total_bytes: u64,
    pub percent: f64,
}

/// Complete parsed PS4G file result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PS4GParseResult {
    pub success: bool,
    pub metadata: PS4GMetadata,
    pub summary: PS4GSummary,
    pub data_preview: Vec<PS4GDataRow>,
    pub error: Option<String>,
}

const PREVIEW_ROW_LIMIT: usize = 100;
const PROGRESS_UPDATE_INTERVAL: usize = 100_000;

/// Parse a PS4G file and return structured data
/// Optimized for large files (8M+ rows) with streaming computation
/// Caches parsed data for fast chromosome switching
#[tauri::command]
pub async fn parse_ps4g_file(
    file_path: String,
    cache: State<'_, PS4GCache>,
    window: tauri::Window,
) -> Result<PS4GParseResult, String> {
    let path = Path::new(&file_path);

    if !path.exists() {
        return Err(format!("File not found: {}", file_path));
    }

    // Get file metadata for cache validation
    let file_metadata = std::fs::metadata(path)
        .map_err(|e| format!("Failed to get file metadata: {}", e))?;
    let file_size = file_metadata.len();
    let modified_time = file_metadata
        .modified()
        .unwrap_or(std::time::SystemTime::UNIX_EPOCH);

    let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file); // 1MB buffer for better I/O

    let mut metadata = PS4GMetadata {
        version: None,
        command: None,
        total_unique_counts: None,
        gametes: Vec::new(),
    };

    // Streaming data structures - only store first N rows for preview
    let mut data_preview: Vec<PS4GDataRow> = Vec::with_capacity(PREVIEW_ROW_LIMIT);
    let mut total_rows: usize = 0;

    // Use FxHashMap for faster chromosome index lookups (string interning)
    let mut chromosome_to_index: FxHashMap<String, u16> = FxHashMap::default();
    let mut index_to_chromosome: Vec<String> = Vec::new();
    let mut unique_positions: FxHashSet<(u16, u64)> = FxHashSet::default();

    // Statistics tracked per chromosome (by index for efficiency)
    let mut chromosome_counts: Vec<usize> = Vec::new();
    let mut position_ranges: Vec<(u64, u64)> = Vec::new();

    // Per-chromosome data for caching: chromosome_idx -> (position -> (gamete_idx -> count))
    let mut chromosome_position_data: Vec<FxHashMap<u64, FxHashMap<u32, u32>>> = Vec::new();

    let mut in_header = true;
    let mut bytes_processed: u64 = 0;
    let mut line_buf = String::with_capacity(256);

    loop {
        line_buf.clear();
        let bytes_read = reader
            .read_line(&mut line_buf)
            .map_err(|e| format!("Failed to read line: {}", e))?;

        if bytes_read == 0 {
            break; // EOF
        }

        bytes_processed += bytes_read as u64;
        let line = line_buf.trim();

        if line.is_empty() {
            continue;
        }

        // Parse header lines
        if line.starts_with('#') {
            parse_header_line(line, &mut metadata);
            continue;
        }

        // Skip the column header line
        if in_header && line.starts_with("gameteSet") {
            in_header = false;
            continue;
        }

        // Parse data row with streaming computation
        if let Some(row) = parse_data_row(line) {
            // Get or create chromosome index (string interning with FxHashMap)
            let chr_idx = get_or_create_chromosome_index_fx(
                row.ref_contig,
                &mut chromosome_to_index,
                &mut index_to_chromosome,
                &mut chromosome_counts,
                &mut position_ranges,
                &mut chromosome_position_data,
            );

            // Track unique positions using (chr_index, position) tuple
            unique_positions.insert((chr_idx, row.ref_pos_binned));

            // Update chromosome counts
            chromosome_counts[chr_idx as usize] += 1;

            // Update position range for this chromosome
            let range = &mut position_ranges[chr_idx as usize];
            if row.ref_pos_binned < range.0 {
                range.0 = row.ref_pos_binned;
            }
            if row.ref_pos_binned > range.1 {
                range.1 = row.ref_pos_binned;
            }

            // Store position data for caching
            let chr_data = &mut chromosome_position_data[chr_idx as usize];
            let pos_entry = chr_data.entry(row.ref_pos_binned).or_default();
            for gamete_idx in &row.gamete_set {
                *pos_entry.entry(*gamete_idx).or_insert(0) += row.count;
            }

            // Only store rows for preview (first N rows)
            if data_preview.len() < PREVIEW_ROW_LIMIT {
                data_preview.push(PS4GDataRow {
                    gamete_set: row.gamete_set,
                    ref_contig: row.ref_contig.to_string(), // Only allocate for preview rows
                    ref_pos_binned: row.ref_pos_binned,
                    count: row.count,
                });
            }

            total_rows += 1;

            // Emit progress update every N rows
            if total_rows % PROGRESS_UPDATE_INTERVAL == 0 {
                let percent = if file_size > 0 {
                    (bytes_processed as f64 / file_size as f64) * 100.0
                } else {
                    0.0
                };

                let _ = window.emit(
                    "ps4g-progress",
                    PS4GProgress {
                        rows_processed: total_rows,
                        bytes_processed,
                        total_bytes: file_size,
                        percent,
                    },
                );
            }
        }
    }

    // Convert indexed chromosome data to FxHashMap format for caching
    let mut chromosome_data_map: FxHashMap<String, FxHashMap<u64, FxHashMap<u32, u32>>> =
        FxHashMap::default();
    let mut chromosome_counts_fx: FxHashMap<String, usize> = FxHashMap::default();
    let mut position_range_fx: FxHashMap<String, (u64, u64)> = FxHashMap::default();

    for (idx, chr_name) in index_to_chromosome.iter().enumerate() {
        chromosome_data_map.insert(
            chr_name.clone(),
            std::mem::take(&mut chromosome_position_data[idx]),
        );
        chromosome_counts_fx.insert(chr_name.clone(), chromosome_counts[idx]);
        position_range_fx.insert(chr_name.clone(), position_ranges[idx]);
    }

    // Sort chromosomes naturally
    let mut chromosomes: Vec<String> = index_to_chromosome;
    chromosomes.sort_by(|a, b| natural_sort(a, b));

    // Store in cache for fast chromosome matrix retrieval
    let cached = CachedPS4GFile {
        file_path: file_path.clone(),
        modified_time,
        metadata: metadata.clone(),
        chromosome_data: chromosome_data_map,
        chromosomes: chromosomes.clone(),
        chromosome_counts: chromosome_counts_fx.clone(),
        position_ranges: position_range_fx.clone(),
        total_rows,
        unique_positions: unique_positions.len(),
    };
    cache.store(cached);

    // Convert to standard HashMap for JSON serialization (API compatibility)
    let chromosome_counts_map: HashMap<String, usize> = chromosome_counts_fx.into_iter().collect();
    let position_range_map: HashMap<String, (u64, u64)> = position_range_fx.into_iter().collect();

    // Create summary
    let summary = PS4GSummary {
        total_rows,
        unique_positions: unique_positions.len(),
        chromosomes,
        chromosome_counts: chromosome_counts_map,
        gamete_count: metadata.gametes.len(),
        position_range: position_range_map,
    };

    // Emit final progress
    let _ = window.emit(
        "ps4g-progress",
        PS4GProgress {
            rows_processed: total_rows,
            bytes_processed: file_size,
            total_bytes: file_size,
            percent: 100.0,
        },
    );

    Ok(PS4GParseResult {
        success: true,
        metadata,
        summary,
        data_preview,
        error: None,
    })
}

/// Parse a header line and update metadata
#[inline]
fn parse_header_line(line: &str, metadata: &mut PS4GMetadata) {
    if line.starts_with("##PS4G") {
        return; // File format marker
    } else if let Some(version) = line.strip_prefix("#version=") {
        metadata.version = Some(version.to_string());
    } else if let Some(command) = line.strip_prefix("#Command:") {
        metadata.command = Some(command.trim().to_string());
    } else if let Some(count_str) = line.strip_prefix("#TotalUniqueCounts:") {
        if let Ok(count) = count_str.trim().parse::<u64>() {
            metadata.total_unique_counts = Some(count);
        }
    } else if line.starts_with("#gamete\t") {
        // Header row for gamete data, skip
        return;
    } else if line.starts_with('#') && line.contains(':') && line.contains('\t') {
        // Gamete data line: #gamete:phase\tindex\tcount
        let content = line.trim_start_matches('#');
        let parts: Vec<&str> = content.split('\t').collect();
        if parts.len() >= 3 {
            let gamete_full = parts[0];
            let gamete_name = gamete_full.split(':').next().unwrap_or(gamete_full);

            if let (Ok(idx), Ok(count)) = (parts[1].parse::<u32>(), parts[2].parse::<u64>()) {
                let total = metadata.total_unique_counts.unwrap_or(1);
                let weight = count as f64 / total as f64;

                metadata.gametes.push(GameteInfo {
                    gamete: gamete_name.to_string(),
                    gamete_index: idx,
                    read_count: count,
                    weight,
                });
            }
        }
    }
}

/// Parsed data row components (uses borrowed string to avoid allocation)
struct ParsedDataRow<'a> {
    gamete_set: Vec<u32>,
    ref_contig: &'a str,
    ref_pos_binned: u64,
    count: u32,
}

/// Parse a data row and return its components
/// Returns borrowed string reference for ref_contig to avoid allocation per row
#[inline]
fn parse_data_row(line: &str) -> Option<ParsedDataRow<'_>> {
    let mut parts = line.split('\t');

    let gamete_set_str = parts.next()?;
    let ref_contig = parts.next()?;
    let ref_pos_str = parts.next()?;
    let count_str = parts.next()?;

    let gamete_set: Vec<u32> = gamete_set_str
        .split(',')
        .filter_map(|s| s.trim().parse::<u32>().ok())
        .collect();

    let ref_pos_binned = ref_pos_str.parse::<u64>().ok()?;
    let count = count_str.parse::<u32>().ok()?;

    Some(ParsedDataRow {
        gamete_set,
        ref_contig,
        ref_pos_binned,
        count,
    })
}

/// Get or create a chromosome index for string interning (FxHashMap version)
#[inline]
fn get_or_create_chromosome_index_fx(
    chr_name: &str,
    chromosome_to_index: &mut FxHashMap<String, u16>,
    index_to_chromosome: &mut Vec<String>,
    chromosome_counts: &mut Vec<usize>,
    position_ranges: &mut Vec<(u64, u64)>,
    chromosome_position_data: &mut Vec<FxHashMap<u64, FxHashMap<u32, u32>>>,
) -> u16 {
    if let Some(&idx) = chromosome_to_index.get(chr_name) {
        idx
    } else {
        let idx = index_to_chromosome.len() as u16;
        chromosome_to_index.insert(chr_name.to_string(), idx);
        index_to_chromosome.push(chr_name.to_string());
        chromosome_counts.push(0);
        position_ranges.push((u64::MAX, 0)); // Will be updated on first position
        chromosome_position_data.push(FxHashMap::default());
        idx
    }
}

/// Natural sort comparison for chromosome names (chr1, chr2, chr10, etc.)
fn natural_sort(a: &str, b: &str) -> std::cmp::Ordering {
    // Extract numeric parts for natural sorting
    let extract_num = |s: &str| -> Option<u32> {
        s.chars()
            .filter(|c| c.is_ascii_digit())
            .collect::<String>()
            .parse::<u32>()
            .ok()
    };

    match (extract_num(a), extract_num(b)) {
        (Some(na), Some(nb)) => na.cmp(&nb),
        _ => a.cmp(b),
    }
}

/// Result structure for chromosome-specific matrix data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChromosomeMatrixResult {
    pub success: bool,
    pub chromosome: String,
    /// Matrix data: rows = gametes (sorted by index), columns = positions (sorted)
    /// Values are read counts (0 = no reads, >0 = has reads)
    pub matrix: Vec<Vec<u32>>,
    /// Binned position values for x-axis labels (sorted)
    pub positions: Vec<u64>,
    /// Gamete names for y-axis labels (sorted by gamete_index)
    pub gamete_names: Vec<String>,
    /// Number of gametes (rows)
    pub num_gametes: usize,
    /// Number of positions (columns)
    pub num_positions: usize,
    /// Position range (min, max)
    pub position_range: (u64, u64),
    /// Error message if unsuccessful
    pub error: Option<String>,
}

/// Progress update for chromosome matrix loading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChromosomeMatrixProgress {
    pub rows_processed: usize,
    pub chromosome: String,
    pub percent: f64,
}

const MATRIX_PROGRESS_UPDATE_INTERVAL: usize = 50_000;

/// Load chromosome-specific matrix data for heatmap visualization
/// Uses cached data when available for instant chromosome switching
#[tauri::command]
pub async fn get_chromosome_matrix(
    file_path: String,
    chromosome: String,
    cache: State<'_, PS4GCache>,
    window: tauri::Window,
) -> Result<ChromosomeMatrixResult, String> {
    let path = Path::new(&file_path);

    if !path.exists() {
        return Ok(ChromosomeMatrixResult {
            success: false,
            chromosome: chromosome.clone(),
            matrix: vec![],
            positions: vec![],
            gamete_names: vec![],
            num_gametes: 0,
            num_positions: 0,
            position_range: (0, 0),
            error: Some(format!("File not found: {}", file_path)),
        });
    }

    // Try to use cached data first (instant chromosome switching)
    if let Some(cached) = cache.get_cached(&file_path) {
        return build_matrix_from_cache(&cached, &chromosome, &window);
    }

    // No cache available - parse the file (slower path)
    // This can happen if get_chromosome_matrix is called before parse_ps4g_file
    get_chromosome_matrix_uncached(&file_path, &chromosome, &cache, &window).await
}

/// Build matrix from cached data (fast path)
fn build_matrix_from_cache(
    cached: &CachedPS4GFile,
    chromosome: &str,
    window: &tauri::Window,
) -> Result<ChromosomeMatrixResult, String> {
    // Emit instant progress since we're using cache
    let _ = window.emit(
        "chromosome-matrix-progress",
        ChromosomeMatrixProgress {
            rows_processed: cached.total_rows,
            chromosome: chromosome.to_string(),
            percent: 100.0,
        },
    );

    // Get chromosome data from cache
    let position_data = match cached.chromosome_data.get(chromosome) {
        Some(data) => data,
        None => {
            return Ok(ChromosomeMatrixResult {
                success: false,
                chromosome: chromosome.to_string(),
                matrix: vec![],
                positions: vec![],
                gamete_names: vec![],
                num_gametes: 0,
                num_positions: 0,
                position_range: (0, 0),
                error: Some(format!("No data found for chromosome: {}", chromosome)),
            });
        }
    };

    if position_data.is_empty() {
        return Ok(ChromosomeMatrixResult {
            success: false,
            chromosome: chromosome.to_string(),
            matrix: vec![],
            positions: vec![],
            gamete_names: vec![],
            num_gametes: 0,
            num_positions: 0,
            position_range: (0, 0),
            error: Some(format!("No data found for chromosome: {}", chromosome)),
        });
    }

    // Use gametes from cached metadata
    let mut gametes = cached.metadata.gametes.clone();
    gametes.sort_by_key(|g| g.gamete_index);

    // Build gamete index to row mapping using FxHashMap
    let gamete_idx_to_row: FxHashMap<u32, usize> = gametes
        .iter()
        .enumerate()
        .map(|(row, g)| (g.gamete_index, row))
        .collect();

    // Sort positions
    let mut positions: Vec<u64> = position_data.keys().copied().collect();
    positions.sort();

    // Build position to column mapping using FxHashMap
    let pos_to_col: FxHashMap<u64, usize> = positions
        .iter()
        .enumerate()
        .map(|(col, &pos)| (pos, col))
        .collect();

    let num_gametes = gametes.len();
    let num_positions = positions.len();

    // Build matrix (gametes x positions)
    let mut matrix: Vec<Vec<u32>> = vec![vec![0; num_positions]; num_gametes];

    for (&pos, gamete_counts) in position_data {
        if let Some(&col) = pos_to_col.get(&pos) {
            for (&gamete_idx, &count) in gamete_counts {
                if let Some(&row) = gamete_idx_to_row.get(&gamete_idx) {
                    matrix[row][col] = count;
                }
            }
        }
    }

    // Get position range
    let position_range = if !positions.is_empty() {
        (*positions.first().unwrap(), *positions.last().unwrap())
    } else {
        (0, 0)
    };

    // Extract gamete names
    let gamete_names: Vec<String> = gametes.iter().map(|g| g.gamete.clone()).collect();

    Ok(ChromosomeMatrixResult {
        success: true,
        chromosome: chromosome.to_string(),
        matrix,
        positions,
        gamete_names,
        num_gametes,
        num_positions,
        position_range,
        error: None,
    })
}

/// Parse file and build matrix when cache is not available (slow path)
/// Also populates the cache for future requests
async fn get_chromosome_matrix_uncached(
    file_path: &str,
    chromosome: &str,
    cache: &State<'_, PS4GCache>,
    window: &tauri::Window,
) -> Result<ChromosomeMatrixResult, String> {
    let path = Path::new(file_path);

    // Get file metadata
    let file_metadata = std::fs::metadata(path)
        .map_err(|e| format!("Failed to get file metadata: {}", e))?;
    let file_size = file_metadata.len();
    let modified_time = file_metadata
        .modified()
        .unwrap_or(std::time::SystemTime::UNIX_EPOCH);

    let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file); // 1MB buffer

    let mut metadata = PS4GMetadata {
        version: None,
        command: None,
        total_unique_counts: None,
        gametes: Vec::new(),
    };

    // Use FxHashMap for faster lookups
    let mut chromosome_to_index: FxHashMap<String, u16> = FxHashMap::default();
    let mut index_to_chromosome: Vec<String> = Vec::new();
    let mut chromosome_counts: Vec<usize> = Vec::new();
    let mut position_ranges: Vec<(u64, u64)> = Vec::new();
    let mut chromosome_position_data: Vec<FxHashMap<u64, FxHashMap<u32, u32>>> = Vec::new();

    let mut in_header = true;
    let mut bytes_processed: u64 = 0;
    let mut rows_processed: usize = 0;
    let mut line_buf = String::with_capacity(256);
    let mut total_rows: usize = 0;

    loop {
        line_buf.clear();
        let bytes_read = reader
            .read_line(&mut line_buf)
            .map_err(|e| format!("Failed to read line: {}", e))?;

        if bytes_read == 0 {
            break; // EOF
        }

        bytes_processed += bytes_read as u64;
        let line = line_buf.trim();

        if line.is_empty() {
            continue;
        }

        // Parse header lines to get gamete info
        if line.starts_with('#') {
            parse_header_line(line, &mut metadata);
            continue;
        }

        // Skip the column header line
        if in_header && line.starts_with("gameteSet") {
            in_header = false;
            continue;
        }

        // Parse data row - collect ALL data for caching
        if let Some(row) = parse_data_row(line) {
            let chr_idx = get_or_create_chromosome_index_fx(
                row.ref_contig,
                &mut chromosome_to_index,
                &mut index_to_chromosome,
                &mut chromosome_counts,
                &mut position_ranges,
                &mut chromosome_position_data,
            );

            // Update chromosome counts
            chromosome_counts[chr_idx as usize] += 1;

            // Update position range
            let range = &mut position_ranges[chr_idx as usize];
            if row.ref_pos_binned < range.0 {
                range.0 = row.ref_pos_binned;
            }
            if row.ref_pos_binned > range.1 {
                range.1 = row.ref_pos_binned;
            }

            // Store position data for caching
            let chr_data = &mut chromosome_position_data[chr_idx as usize];
            let pos_entry = chr_data.entry(row.ref_pos_binned).or_default();
            for gamete_idx in row.gamete_set {
                *pos_entry.entry(gamete_idx).or_insert(0) += row.count;
            }

            total_rows += 1;
            rows_processed += 1;

            // Emit progress update
            if rows_processed % MATRIX_PROGRESS_UPDATE_INTERVAL == 0 {
                let percent = if file_size > 0 {
                    (bytes_processed as f64 / file_size as f64) * 100.0
                } else {
                    0.0
                };

                let _ = window.emit(
                    "chromosome-matrix-progress",
                    ChromosomeMatrixProgress {
                        rows_processed,
                        chromosome: chromosome.to_string(),
                        percent,
                    },
                );
            }
        }
    }

    // Build cache data
    let mut chromosome_data_map: FxHashMap<String, FxHashMap<u64, FxHashMap<u32, u32>>> =
        FxHashMap::default();
    let mut chromosome_counts_fx: FxHashMap<String, usize> = FxHashMap::default();
    let mut position_range_fx: FxHashMap<String, (u64, u64)> = FxHashMap::default();
    let mut unique_positions: usize = 0;

    for (idx, chr_name) in index_to_chromosome.iter().enumerate() {
        unique_positions += chromosome_position_data[idx].len();
        chromosome_data_map.insert(
            chr_name.clone(),
            std::mem::take(&mut chromosome_position_data[idx]),
        );
        chromosome_counts_fx.insert(chr_name.clone(), chromosome_counts[idx]);
        position_range_fx.insert(chr_name.clone(), position_ranges[idx]);
    }

    // Sort chromosomes naturally
    let mut chromosomes: Vec<String> = index_to_chromosome;
    chromosomes.sort_by(|a, b| natural_sort(a, b));

    // Store in cache
    let cached = CachedPS4GFile {
        file_path: file_path.to_string(),
        modified_time,
        metadata: metadata.clone(),
        chromosome_data: chromosome_data_map,
        chromosomes,
        chromosome_counts: chromosome_counts_fx,
        position_ranges: position_range_fx,
        total_rows,
        unique_positions,
    };
    cache.store(cached.clone());

    // Now build the matrix for the requested chromosome from cache
    build_matrix_from_cache(&cached, chromosome, window)
}

// ============================================================================
// Binary Matrix Encoding for Faster Data Transfer
// ============================================================================

use base64::{engine::general_purpose, Engine as _};

/// Compact result structure with base64-encoded matrix data
/// Uses ~75% less bandwidth than nested Vec<Vec<u32>> JSON
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChromosomeMatrixBinaryResult {
    pub success: bool,
    pub chromosome: String,
    /// Base64-encoded flat matrix data (row-major order, u32 little-endian)
    pub matrix_data: String,
    /// Matrix shape: [num_gametes, num_positions]
    pub shape: [usize; 2],
    /// Data type identifier for frontend decoding
    pub dtype: String,
    /// Binned position values for x-axis labels (sorted)
    pub positions: Vec<u64>,
    /// Gamete names for y-axis labels (sorted by gamete_index)
    pub gamete_names: Vec<String>,
    /// Position range (min, max)
    pub position_range: (u64, u64),
    /// Error message if unsuccessful
    pub error: Option<String>,
}

/// Get chromosome matrix in compact binary format (base64-encoded)
/// Faster serialization and ~75% smaller payload than JSON arrays
#[tauri::command]
pub async fn get_chromosome_matrix_binary(
    file_path: String,
    chromosome: String,
    cache: State<'_, PS4GCache>,
    window: tauri::Window,
) -> Result<ChromosomeMatrixBinaryResult, String> {
    // First get the regular matrix result
    let result = get_chromosome_matrix(file_path, chromosome.clone(), cache, window).await?;

    if !result.success {
        return Ok(ChromosomeMatrixBinaryResult {
            success: false,
            chromosome: result.chromosome,
            matrix_data: String::new(),
            shape: [0, 0],
            dtype: "uint32".to_string(),
            positions: vec![],
            gamete_names: vec![],
            position_range: (0, 0),
            error: result.error,
        });
    }

    // Convert matrix to flat u32 array in row-major order
    let num_rows = result.matrix.len();
    let num_cols = if num_rows > 0 { result.matrix[0].len() } else { 0 };

    let flat_data: Vec<u8> = result
        .matrix
        .iter()
        .flat_map(|row| row.iter().flat_map(|&val| val.to_le_bytes()))
        .collect();

    // Encode as base64
    let matrix_data = general_purpose::STANDARD.encode(&flat_data);

    Ok(ChromosomeMatrixBinaryResult {
        success: true,
        chromosome: result.chromosome,
        matrix_data,
        shape: [num_rows, num_cols],
        dtype: "uint32".to_string(),
        positions: result.positions,
        gamete_names: result.gamete_names,
        position_range: result.position_range,
        error: None,
    })
}
