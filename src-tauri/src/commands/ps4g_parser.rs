use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use tauri::Emitter;

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
#[tauri::command]
pub async fn parse_ps4g_file(
    file_path: String,
    window: tauri::Window,
) -> Result<PS4GParseResult, String> {
    let path = Path::new(&file_path);

    if !path.exists() {
        return Err(format!("File not found: {}", file_path));
    }

    // Get file size for progress reporting
    let file_size = std::fs::metadata(path)
        .map(|m| m.len())
        .unwrap_or(0);

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

    // Use chromosome index interning for efficient unique position tracking
    // Instead of HashSet<String> with "chr_pos" format, use HashSet<(u16, u64)>
    let mut chromosome_to_index: HashMap<String, u16> = HashMap::new();
    let mut index_to_chromosome: Vec<String> = Vec::new();
    let mut unique_positions: HashSet<(u16, u64)> = HashSet::new();

    // Statistics tracked per chromosome (by index for efficiency)
    let mut chromosome_counts: Vec<usize> = Vec::new();
    let mut position_ranges: Vec<(u64, u64)> = Vec::new();

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
        if let Some((gamete_set, ref_contig, ref_pos_binned, count)) = parse_data_row(line) {
            // Get or create chromosome index (string interning)
            let chr_idx = get_or_create_chromosome_index(
                &ref_contig,
                &mut chromosome_to_index,
                &mut index_to_chromosome,
                &mut chromosome_counts,
                &mut position_ranges,
            );

            // Track unique positions using (chr_index, position) tuple
            // This avoids creating a format string for every row
            unique_positions.insert((chr_idx, ref_pos_binned));

            // Update chromosome counts
            chromosome_counts[chr_idx as usize] += 1;

            // Update position range for this chromosome
            let range = &mut position_ranges[chr_idx as usize];
            if ref_pos_binned < range.0 {
                range.0 = ref_pos_binned;
            }
            if ref_pos_binned > range.1 {
                range.1 = ref_pos_binned;
            }

            // Only store rows for preview (first N rows)
            if data_preview.len() < PREVIEW_ROW_LIMIT {
                data_preview.push(PS4GDataRow {
                    gamete_set,
                    ref_contig,
                    ref_pos_binned,
                    count,
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

    // Convert indexed chromosome data back to HashMap format for API compatibility
    let mut chromosome_counts_map: HashMap<String, usize> = HashMap::new();
    let mut position_range_map: HashMap<String, (u64, u64)> = HashMap::new();

    for (idx, chr_name) in index_to_chromosome.iter().enumerate() {
        chromosome_counts_map.insert(chr_name.clone(), chromosome_counts[idx]);
        position_range_map.insert(chr_name.clone(), position_ranges[idx]);
    }

    // Sort chromosomes naturally
    let mut chromosomes: Vec<String> = index_to_chromosome;
    chromosomes.sort_by(|a, b| natural_sort(a, b));

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

/// Parse a data row and return its components
/// Returns None if the row is invalid
#[inline]
fn parse_data_row(line: &str) -> Option<(Vec<u32>, String, u64, u32)> {
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

    Some((gamete_set, ref_contig.to_string(), ref_pos_binned, count))
}

/// Get or create a chromosome index for string interning
#[inline]
fn get_or_create_chromosome_index(
    chr_name: &str,
    chromosome_to_index: &mut HashMap<String, u16>,
    index_to_chromosome: &mut Vec<String>,
    chromosome_counts: &mut Vec<usize>,
    position_ranges: &mut Vec<(u64, u64)>,
) -> u16 {
    if let Some(&idx) = chromosome_to_index.get(chr_name) {
        idx
    } else {
        let idx = index_to_chromosome.len() as u16;
        chromosome_to_index.insert(chr_name.to_string(), idx);
        index_to_chromosome.push(chr_name.to_string());
        chromosome_counts.push(0);
        position_ranges.push((u64::MAX, 0)); // Will be updated on first position
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
