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

    // `in_header` tracks the leading run of `#`-prefixed (and blank) lines,
    // which is the only place metadata and gamete records are recognized.
    // `in_gamete_section` additionally tracks whether we're between the
    // `#gamete` tag line and the end of that header block — gamete records
    // are only ever parsed there, never by line shape alone (see
    // `parse_gamete_record`).
    let mut in_header = true;
    let mut in_gamete_section = false;
    // If the header block ends with no gamete records recognized (no
    // `#gamete` tag was ever seen), fall back to synthesizing one gamete per
    // distinct index that actually appears in the data section's
    // `gameteSet` column, named by that index. `need_fallback_gametes` is
    // decided once, right when the header block ends.
    let mut need_fallback_gametes = false;
    let mut fallback_tally: FxHashMap<u32, u64> = FxHashMap::default();
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
            if in_header {
                if is_gamete_section_tag(line) {
                    in_gamete_section = true;
                } else if !parse_metadata_line(line, &mut metadata) && in_gamete_section {
                    // Inside the gamete section, every non-metadata `#` line
                    // is expected to be a record. A malformed one is skipped
                    // (not fatal, section stays open) rather than treated as
                    // "not a gamete record" the way the old shape sniff did.
                    if let Some(g) = parse_gamete_record(line, metadata.total_unique_counts) {
                        metadata.gametes.push(g);
                    }
                }
            }
            // `#` lines outside the header block are inert trailing
            // comments — never scanned for metadata or gamete records.
            continue;
        }

        if in_header {
            // First non-'#' line ends the header block (and with it, the
            // gamete section, if one was ever opened).
            in_header = false;
            in_gamete_section = false;
            need_fallback_gametes = metadata.gametes.is_empty();

            if line.starts_with("gameteSet") {
                continue;
            }
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

            if need_fallback_gametes {
                for gamete_idx in &row.gamete_set {
                    *fallback_tally.entry(*gamete_idx).or_insert(0) += row.count as u64;
                }
            }

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

    // No `#gamete` section was found in the header — synthesize one gamete
    // per distinct index seen in the data section's `gameteSet` column,
    // named by that index, rather than falling back to shape-sniffing `#`
    // lines (the thing this section-gated design replaces).
    if need_fallback_gametes && !fallback_tally.is_empty() {
        let total = metadata.total_unique_counts.unwrap_or(1);
        let mut indices: Vec<u32> = fallback_tally.keys().copied().collect();
        indices.sort_unstable();
        for idx in indices {
            let name = idx.to_string();
            let read_count = fallback_tally[&idx];
            metadata.gametes.push(GameteInfo {
                gamete: name.clone(),
                sample_name: name,
                gamete_idx: 0,
                gamete_index: idx,
                read_count,
                weight: read_count as f64 / total as f64,
            });
        }
    }

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

    // Name gametes by bare sample name, except when two gametes in this
    // panel share a sample name (e.g. "B73:0" and "B73:1") — then fall back
    // to "sample:idx" for just those so rows stay distinguishable.
    let mut sample_name_counts: FxHashMap<&str, usize> = FxHashMap::default();
    for g in &gametes {
        *sample_name_counts.entry(g.sample_name.as_str()).or_insert(0) += 1;
    }
    let gamete_names: Vec<String> = gametes
        .iter()
        .map(|g| {
            if sample_name_counts.get(g.sample_name.as_str()).copied().unwrap_or(0) > 1 {
                format!("{}:{}", g.sample_name, g.gamete_idx)
            } else {
                g.sample_name.clone()
            }
        })
        .collect();

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

/// Parse a gamete header field into (sample_name, gamete_idx).
///
/// Per the PS4G spec, the field is either `<sampleName>` (index implicitly
/// 0) or `<sampleName>:<gameteIdx>`. Only a trailing `:<digits>` suffix is
/// treated as an index; a sample name that itself contains `:` but has no
/// pure-digit suffix is kept whole.
#[inline]
fn parse_sample_gamete(field: &str) -> (String, u32) {
    if let Some((name, suffix)) = field.rsplit_once(':') {
        if !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_digit()) {
            if let Ok(idx) = suffix.parse::<u32>() {
                return (name.to_string(), idx);
            }
        }
    }
    (field.to_string(), 0)
}

/// True if `line` is the `#gamete\tgameteIndex\tcount` tag line that opens
/// the gamete section (see `parse_ps4g`). Only the first tab field is
/// checked — the column names after it are informational — so this matches
/// regardless of exact header spelling, and matching is case-insensitive
/// and tolerant of extra leading `#`s (mirroring the `#PS4G`/`##PS4G` magic
/// line).
#[inline]
pub(crate) fn is_gamete_section_tag(line: &str) -> bool {
    line.trim_start_matches('#')
        .split('\t')
        .next()
        .map(|field| field.trim().eq_ignore_ascii_case("gamete"))
        .unwrap_or(false)
}

/// Consume a keyed metadata line (`#PS4G`, `#version=`, `#Command:`,
/// `#TotalUniqueCounts:`). Returns `true` if `line` was one of these, so the
/// caller knows not to also try `parse_gamete_record` on it — these keys
/// take precedence even inside the gamete section (a producer-declared
/// `#TotalUniqueCounts:` line interleaved among gamete records shouldn't be
/// mistaken for a malformed record).
#[inline]
pub(crate) fn parse_metadata_line(line: &str, metadata: &mut PS4GMetadata) -> bool {
    if line.trim_start_matches('#') == "PS4G" {
        // Magic line, e.g. "#PS4G" (v2.0 form) or "##PS4G" (legacy form).
        true
    } else if let Some(version) = line.strip_prefix("#version=") {
        metadata.version = Some(version.to_string());
        true
    } else if let Some(command) = line.strip_prefix("#Command:") {
        metadata.command = Some(command.trim().to_string());
        true
    } else if let Some(count_str) = line.strip_prefix("#TotalUniqueCounts:") {
        if let Ok(count) = count_str.trim().parse::<u64>() {
            metadata.total_unique_counts = Some(count);
        }
        true
    } else {
        false
    }
}

/// Parse a line already known — by its position under the `#gamete` tag —
/// to be a gamete record, e.g. `"#B73\t0\t784970"` or `"#B73:0\t0\t784970"`.
/// Returns `None` if the line is malformed (not a shape sniff: the caller
/// has already established this line *should* be a record).
///
/// `total_unique_counts` is the file's declared `#TotalUniqueCounts:`
/// value, if seen before this line — weight is normalized against it
/// (falling back to 1 if unknown), matching the file's own row-count
/// column, e.g. `read_count / total_unique_counts`.
#[inline]
pub(crate) fn parse_gamete_record(
    line: &str,
    total_unique_counts: Option<u64>,
) -> Option<GameteInfo> {
    let content = line.trim_start_matches('#');
    let parts: Vec<&str> = content.split('\t').collect();
    if parts.len() < 3 {
        return None;
    }
    let gamete_full = parts[0];
    let idx = parts[1].parse::<u32>().ok()?;
    let count = parts[2].parse::<u64>().ok()?;
    let (sample_name, gamete_idx) = parse_sample_gamete(gamete_full);
    let total = total_unique_counts.unwrap_or(1);

    Some(GameteInfo {
        gamete: sample_name.clone(),
        sample_name,
        gamete_idx,
        gamete_index: idx,
        read_count: count,
        weight: count as f64 / total as f64,
    })
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn fresh_metadata() -> PS4GMetadata {
        PS4GMetadata {
            version: None,
            command: None,
            total_unique_counts: None,
            gametes: Vec::new(),
        }
    }

    #[test]
    fn parses_bare_gamete_name() {
        let g = parse_gamete_record("#B73\t0\t60", Some(100)).unwrap();
        assert_eq!(g.gamete, "B73");
        assert_eq!(g.sample_name, "B73");
        assert_eq!(g.gamete_idx, 0);
        assert_eq!(g.gamete_index, 0);
        assert_eq!(g.read_count, 60);
        assert!((g.weight - 0.6).abs() < 1e-9);
    }

    #[test]
    fn parses_colon_suffixed_gamete_name() {
        let g = parse_gamete_record("#B73:1\t2\t40", Some(100)).unwrap();
        assert_eq!(g.gamete, "B73");
        assert_eq!(g.sample_name, "B73");
        assert_eq!(g.gamete_idx, 1);
        assert_eq!(g.gamete_index, 2);
    }

    #[test]
    fn parse_gamete_record_rejects_malformed_lines() {
        assert!(parse_gamete_record("#B73\t0", None).is_none()); // too few fields
        assert!(parse_gamete_record("#B73\tx\t4", None).is_none()); // non-integer index
        assert!(parse_gamete_record("#B73\t0\tx", None).is_none()); // non-integer count
    }

    #[test]
    fn gamete_tag_recognized_in_all_forms() {
        assert!(is_gamete_section_tag("#gamete\tgameteIndex\tcount"));
        assert!(is_gamete_section_tag("#gamete"));
        assert!(is_gamete_section_tag("##gamete\tgameteIndex\tcount"));
        assert!(is_gamete_section_tag("#GAMETE\tgameteIndex\tcount"));
        assert!(!is_gamete_section_tag("#B73\t0\t4"));
        assert!(!is_gamete_section_tag("#Command: ropebwt3 refmap"));
        assert!(!is_gamete_section_tag("#version=2.0"));
    }

    #[test]
    fn command_line_with_colon_is_recognized_as_metadata() {
        let mut metadata = fresh_metadata();
        assert!(parse_metadata_line(
            "#Command: ropebwt3 refmap --ref-prefix=B73 --max-occ=-1",
            &mut metadata,
        ));
        assert_eq!(
            metadata.command.as_deref(),
            Some("ropebwt3 refmap --ref-prefix=B73 --max-occ=-1")
        );
    }

    #[test]
    fn magic_and_version_lines_are_metadata_not_gamete_records() {
        let mut metadata = fresh_metadata();
        assert!(parse_metadata_line("#PS4G", &mut metadata));
        assert!(parse_metadata_line("##PS4G", &mut metadata));
        assert!(parse_metadata_line("#version=2.0", &mut metadata));
        assert!(metadata.gametes.is_empty());
        assert_eq!(metadata.version.as_deref(), Some("2.0"));
    }

    #[test]
    fn total_unique_counts_drives_weight() {
        let g1 = parse_gamete_record("#B73\t0\t250", Some(1000)).unwrap();
        let g2 = parse_gamete_record("#B97\t1\t750", Some(1000)).unwrap();
        assert!((g1.weight - 0.25).abs() < 1e-9);
        assert!((g2.weight - 0.75).abs() < 1e-9);
    }

    fn sample_ps4g_bytes() -> &'static [u8] {
        b"#TotalUniqueCounts: 4\n\
          #gamete\tgameteIndex\tcount\n\
          #B73:0\t0\t4\n\
          #CML247:0\t1\t2\n\
          #W22:0\t2\t1\n\
          gameteSet\trefContig\trefPosBinned\tcount\n\
          0\tchr1\t1000\t1\n\
          0,1\tchr1\t1000\t1\n\
          0\tchr1\t2000\t1\n\
          0,1,2\tchr1\t2000\t1\n"
    }

    #[test]
    fn records_before_gamete_tag_are_ignored() {
        // A gamete-shaped line appearing before the #gamete tag opens the
        // section is not a record -- position, not shape, is what matters.
        let bytes = b"#B73:0\t0\t4\n\
                       #gamete\tgameteIndex\tcount\n\
                       #CML247:0\t1\t2\n\
                       #W22:0\t2\t1\n\
                       gameteSet\trefContig\trefPosBinned\tcount\n\
                       0\tchr1\t1000\t1\n\
                       0,1\tchr1\t1000\t1\n\
                       0\tchr1\t2000\t1\n\
                       0,1,2\tchr1\t2000\t1\n";
        let (result, _cached) = parse_ps4g(Cursor::new(&bytes[..]), None, |_| {}).unwrap();

        assert_eq!(result.summary.gamete_count, 2);
        assert!(result
            .metadata
            .gametes
            .iter()
            .all(|g| g.sample_name != "B73"));
    }

    #[test]
    fn three_column_comment_outside_section_is_not_a_gamete() {
        // The reviewer's regression case: a #-line with the old gamete
        // shape (3 tab fields, integer cols 2-3) that is NOT a gamete
        // record, placed both before the tag and after the data section.
        let bytes = b"#binSize\t256\t1\n\
                       #gamete\tgameteIndex\tcount\n\
                       #B73:0\t0\t4\n\
                       #CML247:0\t1\t2\n\
                       #W22:0\t2\t1\n\
                       gameteSet\trefContig\trefPosBinned\tcount\n\
                       0\tchr1\t1000\t1\n\
                       0,1\tchr1\t1000\t1\n\
                       0\tchr1\t2000\t1\n\
                       0,1,2\tchr1\t2000\t1\n\
                       #binSize\t256\t1\n";
        let (result, _cached) = parse_ps4g(Cursor::new(&bytes[..]), None, |_| {}).unwrap();

        assert_eq!(result.summary.gamete_count, 3);
    }

    #[test]
    fn comment_after_data_section_is_ignored() {
        // '#' lines are only recognized inside the leading header block --
        // once data rows start, trailing '#' lines (metadata or gamete
        // shaped) are inert comments.
        let bytes = b"#gamete\tgameteIndex\tcount\n\
                       #B73:0\t0\t4\n\
                       gameteSet\trefContig\trefPosBinned\tcount\n\
                       0\tchr1\t1000\t4\n\
                       #B99\t9\t9\n\
                       #version=9.9\n";
        let (result, _cached) = parse_ps4g(Cursor::new(&bytes[..]), None, |_| {}).unwrap();

        assert_eq!(result.summary.gamete_count, 1);
        assert_eq!(result.metadata.version, None);
    }

    #[test]
    fn file_without_gamete_tag_synthesizes_ids_from_data() {
        // No #gamete tag anywhere -- rather than falling back to shape
        // sniffing, gametes are synthesized from the indices actually used
        // in the data section's gameteSet column, named by that index.
        let bytes = b"#TotalUniqueCounts: 4\n\
                       gameteSet\trefContig\trefPosBinned\tcount\n\
                       0\tchr1\t1000\t1\n\
                       0,1\tchr1\t1000\t1\n\
                       0\tchr1\t2000\t1\n\
                       0,1,2\tchr1\t2000\t1\n";
        let (result, _cached) = parse_ps4g(Cursor::new(&bytes[..]), None, |_| {}).unwrap();

        assert_eq!(result.summary.gamete_count, 3);
        let mut names: Vec<&str> = result
            .metadata
            .gametes
            .iter()
            .map(|g| g.sample_name.as_str())
            .collect();
        names.sort();
        assert_eq!(names, vec!["0", "1", "2"]);

        let by_index: FxHashMap<u32, u64> = result
            .metadata
            .gametes
            .iter()
            .map(|g| (g.gamete_index, g.read_count))
            .collect();
        assert_eq!(by_index[&0], 4); // hits all 4 rows
        assert_eq!(by_index[&1], 2); // hits rows 2, 4
        assert_eq!(by_index[&2], 1); // hits row 4
    }

    #[test]
    fn malformed_record_inside_section_is_skipped() {
        let bytes = b"#gamete\tgameteIndex\tcount\n\
                       #B73:0\t0\t4\n\
                       #bad\tnot-a-number\t2\n\
                       #W22:0\t2\t1\n\
                       gameteSet\trefContig\trefPosBinned\tcount\n\
                       0\tchr1\t1000\t1\n";
        let (result, _cached) = parse_ps4g(Cursor::new(&bytes[..]), None, |_| {}).unwrap();

        assert_eq!(result.summary.gamete_count, 2);
    }

    #[test]
    fn full_file_parses_three_gametes() {
        let (result, _cached) =
            parse_ps4g(Cursor::new(sample_ps4g_bytes()), None, |_| {}).unwrap();
        assert_eq!(result.summary.gamete_count, 3);
        assert_eq!(result.summary.total_rows, 4);
    }
}
