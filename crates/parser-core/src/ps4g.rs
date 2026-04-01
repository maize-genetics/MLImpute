use crate::types::*;
use crate::util::natural_sort;
use base64::{engine::general_purpose, Engine as _};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::HashMap;
use std::io::BufRead;

const PREVIEW_ROW_LIMIT: usize = 100;
const PROGRESS_UPDATE_INTERVAL: usize = 100_000;

/// Parse a PS4G file from any `BufRead` source and return structured data plus
/// cached chromosome data for fast matrix retrieval.
pub fn parse_ps4g(
    mut reader: impl BufRead,
    file_size: Option<u64>,
    mut on_progress: impl FnMut(PS4GProgress),
) -> Result<(PS4GParseResult, CachedPS4GData), String> {
    let total_bytes = file_size.unwrap_or(0);

    let mut metadata = PS4GMetadata {
        version: None,
        command: None,
        total_unique_counts: None,
        gametes: Vec::new(),
    };

    let mut data_preview: Vec<PS4GDataRow> = Vec::with_capacity(PREVIEW_ROW_LIMIT);
    let mut total_rows: usize = 0;

    let mut chromosome_to_index: FxHashMap<String, u16> = FxHashMap::default();
    let mut index_to_chromosome: Vec<String> = Vec::new();
    let mut unique_positions: FxHashSet<(u16, u64)> = FxHashSet::default();

    let mut chromosome_counts: Vec<usize> = Vec::new();
    let mut position_ranges: Vec<(u64, u64)> = Vec::new();
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
            break;
        }

        bytes_processed += bytes_read as u64;
        let line = line_buf.trim();

        if line.is_empty() {
            continue;
        }

        if line.starts_with('#') {
            parse_header_line(line, &mut metadata);
            continue;
        }

        if in_header && line.starts_with("gameteSet") {
            in_header = false;
            continue;
        }

        if let Some(row) = parse_data_row(line) {
            let chr_idx = get_or_create_chromosome_index(
                row.ref_contig,
                &mut chromosome_to_index,
                &mut index_to_chromosome,
                &mut chromosome_counts,
                &mut position_ranges,
                &mut chromosome_position_data,
            );

            unique_positions.insert((chr_idx, row.ref_pos_binned));
            chromosome_counts[chr_idx as usize] += 1;

            let range = &mut position_ranges[chr_idx as usize];
            if row.ref_pos_binned < range.0 {
                range.0 = row.ref_pos_binned;
            }
            if row.ref_pos_binned > range.1 {
                range.1 = row.ref_pos_binned;
            }

            let chr_data = &mut chromosome_position_data[chr_idx as usize];
            let pos_entry = chr_data.entry(row.ref_pos_binned).or_default();
            for gamete_idx in &row.gamete_set {
                *pos_entry.entry(*gamete_idx).or_insert(0) += row.count;
            }

            if data_preview.len() < PREVIEW_ROW_LIMIT {
                data_preview.push(PS4GDataRow {
                    gamete_set: row.gamete_set,
                    ref_contig: row.ref_contig.to_string(),
                    ref_pos_binned: row.ref_pos_binned,
                    count: row.count,
                });
            }

            total_rows += 1;

            if total_rows % PROGRESS_UPDATE_INTERVAL == 0 {
                let percent = if total_bytes > 0 {
                    (bytes_processed as f64 / total_bytes as f64) * 100.0
                } else {
                    0.0
                };
                on_progress(PS4GProgress {
                    rows_processed: total_rows,
                    bytes_processed,
                    total_bytes,
                    percent,
                });
            }
        }
    }

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

    let mut chromosomes: Vec<String> = index_to_chromosome;
    chromosomes.sort_by(|a, b| natural_sort(a, b));

    let chromosome_counts_map: HashMap<String, usize> =
        chromosome_counts_fx.iter().map(|(k, v)| (k.clone(), *v)).collect();
    let position_range_map: HashMap<String, (u64, u64)> =
        position_range_fx.iter().map(|(k, v)| (k.clone(), *v)).collect();

    let summary = PS4GSummary {
        total_rows,
        unique_positions: unique_positions.len(),
        chromosomes: chromosomes.clone(),
        chromosome_counts: chromosome_counts_map,
        gamete_count: metadata.gametes.len(),
        position_range: position_range_map,
    };

    on_progress(PS4GProgress {
        rows_processed: total_rows,
        bytes_processed: total_bytes,
        total_bytes,
        percent: 100.0,
    });

    let cached = CachedPS4GData {
        metadata: metadata.clone(),
        chromosome_data: chromosome_data_map,
        chromosomes,
        chromosome_counts: chromosome_counts_fx,
        position_ranges: position_range_fx,
        total_rows,
        unique_positions: unique_positions.len(),
    };

    let result = PS4GParseResult {
        success: true,
        metadata,
        summary,
        data_preview,
        error: None,
    };

    Ok((result, cached))
}

/// Build a chromosome matrix from previously cached PS4G data.
pub fn build_chromosome_matrix(
    cached: &CachedPS4GData,
    chromosome: &str,
) -> Result<ChromosomeMatrixResult, String> {
    let position_data = match cached.chromosome_data.get(chromosome) {
        Some(data) if !data.is_empty() => data,
        _ => {
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

    let mut gametes = cached.metadata.gametes.clone();
    gametes.sort_by_key(|g| g.gamete_index);

    let gamete_idx_to_row: FxHashMap<u32, usize> = gametes
        .iter()
        .enumerate()
        .map(|(row, g)| (g.gamete_index, row))
        .collect();

    let mut positions: Vec<u64> = position_data.keys().copied().collect();
    positions.sort();

    let pos_to_col: FxHashMap<u64, usize> = positions
        .iter()
        .enumerate()
        .map(|(col, &pos)| (pos, col))
        .collect();

    let num_gametes = gametes.len();
    let num_positions = positions.len();

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

    let position_range = if !positions.is_empty() {
        (*positions.first().unwrap(), *positions.last().unwrap())
    } else {
        (0, 0)
    };

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

/// Encode a chromosome matrix result as a compact base64 binary payload.
pub fn encode_matrix_binary(
    result: &ChromosomeMatrixResult,
) -> ChromosomeMatrixBinaryResult {
    if !result.success {
        return ChromosomeMatrixBinaryResult {
            success: false,
            chromosome: result.chromosome.clone(),
            matrix_data: String::new(),
            shape: [0, 0],
            dtype: "uint32".to_string(),
            positions: vec![],
            gamete_names: vec![],
            position_range: (0, 0),
            error: result.error.clone(),
        };
    }

    let num_rows = result.matrix.len();
    let num_cols = if num_rows > 0 { result.matrix[0].len() } else { 0 };

    let flat_data: Vec<u8> = result
        .matrix
        .iter()
        .flat_map(|row| row.iter().flat_map(|&val| val.to_le_bytes()))
        .collect();

    let matrix_data = general_purpose::STANDARD.encode(&flat_data);

    ChromosomeMatrixBinaryResult {
        success: true,
        chromosome: result.chromosome.clone(),
        matrix_data,
        shape: [num_rows, num_cols],
        dtype: "uint32".to_string(),
        positions: result.positions.clone(),
        gamete_names: result.gamete_names.clone(),
        position_range: result.position_range,
        error: None,
    }
}

// ============================================================================
// Internal helpers
// ============================================================================

#[inline]
pub(crate) fn parse_header_line(line: &str, metadata: &mut PS4GMetadata) {
    if line.starts_with("##PS4G") {
        return;
    } else if let Some(version) = line.strip_prefix("#version=") {
        metadata.version = Some(version.to_string());
    } else if let Some(command) = line.strip_prefix("#Command:") {
        metadata.command = Some(command.trim().to_string());
    } else if let Some(count_str) = line.strip_prefix("#TotalUniqueCounts:") {
        if let Ok(count) = count_str.trim().parse::<u64>() {
            metadata.total_unique_counts = Some(count);
        }
    } else if line.starts_with("#gamete\t") {
        return;
    } else if line.starts_with('#') && line.contains(':') && line.contains('\t') {
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

struct ParsedDataRow<'a> {
    gamete_set: Vec<u32>,
    ref_contig: &'a str,
    ref_pos_binned: u64,
    count: u32,
}

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

#[inline]
fn get_or_create_chromosome_index(
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
        position_ranges.push((u64::MAX, 0));
        chromosome_position_data.push(FxHashMap::default());
        idx
    }
}
