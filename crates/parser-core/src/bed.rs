use crate::types::*;
use crate::util::natural_sort;
use std::collections::{HashMap, HashSet};
use std::io::BufRead;

const PREVIEW_ROW_LIMIT: usize = 100;
const PROGRESS_UPDATE_INTERVAL: usize = 10_000;

/// Parse a BED file from any `BufRead` source and return structured summary data.
pub fn parse_bed(
    mut reader: impl BufRead,
    file_size: Option<u64>,
    mut on_progress: impl FnMut(BEDProgress),
) -> Result<BEDParseResult, String> {
    let total_bytes = file_size.unwrap_or(0);

    let mut data_preview: Vec<BEDDataRow> = Vec::with_capacity(PREVIEW_ROW_LIMIT);
    let mut total_rows: usize = 0;
    let mut total_coverage_bp: u64 = 0;

    let mut chromosome_counts: HashMap<String, usize> = HashMap::new();
    let mut position_ranges: HashMap<String, (u64, u64)> = HashMap::new();
    let mut unique_parents: HashSet<String> = HashSet::new();
    let mut unique_parent_pairs: HashSet<(String, String)> = HashSet::new();

    let mut parent_regions_as_p1: HashMap<String, usize> = HashMap::new();
    let mut parent_regions_as_p2: HashMap<String, usize> = HashMap::new();
    let mut parent_coverage_as_p1: HashMap<String, u64> = HashMap::new();
    let mut parent_coverage_as_p2: HashMap<String, u64> = HashMap::new();
    let mut parent_chromosomes: HashMap<String, HashSet<String>> = HashMap::new();

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

        if line.starts_with("chrom") || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 5 {
            continue;
        }

        let chrom = parts[0];
        let start = match parts[1].parse::<u64>() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let end = match parts[2].parse::<u64>() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let parent1 = parts[3];
        let parent2 = parts[4];

        *chromosome_counts.entry(chrom.to_string()).or_insert(0) += 1;

        let range = position_ranges
            .entry(chrom.to_string())
            .or_insert((u64::MAX, 0));
        if start < range.0 {
            range.0 = start;
        }
        if end > range.1 {
            range.1 = end;
        }

        if end > start {
            total_coverage_bp += end - start;
        }

        unique_parents.insert(parent1.to_string());
        unique_parents.insert(parent2.to_string());
        unique_parent_pairs.insert((parent1.to_string(), parent2.to_string()));

        let region_size = if end > start { end - start } else { 0 };

        *parent_regions_as_p1.entry(parent1.to_string()).or_insert(0) += 1;
        *parent_regions_as_p2.entry(parent2.to_string()).or_insert(0) += 1;
        *parent_coverage_as_p1.entry(parent1.to_string()).or_insert(0) += region_size;
        *parent_coverage_as_p2.entry(parent2.to_string()).or_insert(0) += region_size;

        parent_chromosomes
            .entry(parent1.to_string())
            .or_default()
            .insert(chrom.to_string());
        parent_chromosomes
            .entry(parent2.to_string())
            .or_default()
            .insert(chrom.to_string());

        if data_preview.len() < PREVIEW_ROW_LIMIT {
            data_preview.push(BEDDataRow {
                chrom: chrom.to_string(),
                start,
                end,
                parent1: parent1.to_string(),
                parent2: parent2.to_string(),
            });
        }

        total_rows += 1;

        if total_rows % PROGRESS_UPDATE_INTERVAL == 0 {
            let percent = if total_bytes > 0 {
                (bytes_processed as f64 / total_bytes as f64) * 100.0
            } else {
                0.0
            };
            on_progress(BEDProgress {
                rows_processed: total_rows,
                bytes_processed,
                total_bytes,
                percent,
            });
        }
    }

    let mut chromosomes: Vec<String> = chromosome_counts.keys().cloned().collect();
    chromosomes.sort_by(|a, b| natural_sort(a, b));

    let mut sorted_parents: Vec<String> = unique_parents.into_iter().collect();
    sorted_parents.sort_by(|a, b| natural_sort(a, b));

    let avg_region_size_bp = if total_rows > 0 {
        total_coverage_bp as f64 / total_rows as f64
    } else {
        0.0
    };

    let parent_stats: Vec<ParentStats> = sorted_parents
        .iter()
        .map(|pid| {
            let r_p1 = *parent_regions_as_p1.get(pid).unwrap_or(&0);
            let r_p2 = *parent_regions_as_p2.get(pid).unwrap_or(&0);
            let c_p1 = *parent_coverage_as_p1.get(pid).unwrap_or(&0);
            let c_p2 = *parent_coverage_as_p2.get(pid).unwrap_or(&0);
            let chr_count = parent_chromosomes
                .get(pid)
                .map(|s| s.len())
                .unwrap_or(0);
            ParentStats {
                parent_id: pid.clone(),
                regions_as_parent1: r_p1,
                regions_as_parent2: r_p2,
                total_regions: r_p1 + r_p2,
                coverage_bp_as_parent1: c_p1,
                coverage_bp_as_parent2: c_p2,
                total_coverage_bp: c_p1 + c_p2,
                chromosome_count: chr_count,
            }
        })
        .collect();

    let summary = BEDSummary {
        total_rows,
        chromosomes,
        chromosome_counts,
        position_range: position_ranges,
        total_coverage_bp,
        avg_region_size_bp,
        unique_parents: sorted_parents,
        unique_parent_pairs: unique_parent_pairs.len(),
        parent_stats,
    };

    on_progress(BEDProgress {
        rows_processed: total_rows,
        bytes_processed: total_bytes,
        total_bytes,
        percent: 100.0,
    });

    Ok(BEDParseResult {
        success: true,
        summary,
        data_preview,
        error: None,
    })
}

/// Build a heatmap matrix for a single chromosome from a BED file reader.
/// Re-reads the file from the provided `BufRead` source.
pub fn build_bed_chromosome_matrix(
    mut reader: impl BufRead,
    file_size: Option<u64>,
    chromosome: &str,
    mut on_progress: impl FnMut(BEDMatrixProgress),
) -> Result<BEDChromosomeMatrixResult, String> {
    let total_bytes = file_size.unwrap_or(0);

    let mut chrom_rows: Vec<(u64, u64, String, String)> = Vec::new();
    let mut unique_parents: HashSet<String> = HashSet::new();
    let mut bytes_processed: u64 = 0;
    let mut lines_read: usize = 0;
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
        if line.is_empty() || line.starts_with("chrom") || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 5 {
            continue;
        }

        lines_read += 1;

        if parts[0] != chromosome {
            if lines_read % PROGRESS_UPDATE_INTERVAL == 0 {
                let percent = if total_bytes > 0 {
                    (bytes_processed as f64 / total_bytes as f64) * 100.0
                } else {
                    0.0
                };
                on_progress(BEDMatrixProgress {
                    rows_processed: chrom_rows.len(),
                    chromosome: chromosome.to_string(),
                    percent,
                });
            }
            continue;
        }

        let start = match parts[1].parse::<u64>() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let end = match parts[2].parse::<u64>() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let parent1 = parts[3].to_string();
        let parent2 = parts[4].to_string();

        unique_parents.insert(parent1.clone());
        unique_parents.insert(parent2.clone());
        chrom_rows.push((start, end, parent1, parent2));

        if lines_read % PROGRESS_UPDATE_INTERVAL == 0 {
            let percent = if total_bytes > 0 {
                (bytes_processed as f64 / total_bytes as f64) * 100.0
            } else {
                0.0
            };
            on_progress(BEDMatrixProgress {
                rows_processed: chrom_rows.len(),
                chromosome: chromosome.to_string(),
                percent,
            });
        }
    }

    if chrom_rows.is_empty() {
        return Ok(BEDChromosomeMatrixResult {
            success: true,
            chromosome: chromosome.to_string(),
            matrix: vec![],
            parent_names: vec![],
            regions: vec![],
            num_parents: 0,
            num_regions: 0,
            parent1_path: vec![],
            parent2_path: vec![],
            error: None,
        });
    }

    chrom_rows.sort_by_key(|r| r.0);

    let mut parent_names: Vec<String> = unique_parents.into_iter().collect();
    parent_names.sort_by(|a, b| natural_sort(a, b));

    let parent_index: HashMap<String, usize> = parent_names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.clone(), i))
        .collect();

    let num_parents = parent_names.len();
    let num_regions = chrom_rows.len();

    let mut matrix: Vec<Vec<u8>> = vec![vec![0u8; num_regions]; num_parents];
    let mut parent1_path: Vec<usize> = Vec::with_capacity(num_regions);
    let mut parent2_path: Vec<usize> = Vec::with_capacity(num_regions);
    let mut regions: Vec<BEDRegion> = Vec::with_capacity(num_regions);

    for (col, (start, end, p1, p2)) in chrom_rows.iter().enumerate() {
        regions.push(BEDRegion {
            start: *start,
            end: *end,
        });

        let p1_idx = parent_index[p1];
        let p2_idx = parent_index[p2];

        if p1_idx == p2_idx {
            matrix[p1_idx][col] = 3;
        } else {
            matrix[p1_idx][col] = 1;
            matrix[p2_idx][col] = 2;
        }

        parent1_path.push(p1_idx);
        parent2_path.push(p2_idx);
    }

    on_progress(BEDMatrixProgress {
        rows_processed: num_regions,
        chromosome: chromosome.to_string(),
        percent: 100.0,
    });

    Ok(BEDChromosomeMatrixResult {
        success: true,
        chromosome: chromosome.to_string(),
        matrix,
        parent_names,
        regions,
        num_parents,
        num_regions,
        parent1_path,
        parent2_path,
        error: None,
    })
}
