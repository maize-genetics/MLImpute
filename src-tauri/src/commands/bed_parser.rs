use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use serde::{Deserialize, Serialize};
use tauri::command;
use base64::{Engine as _, engine::general_purpose};

#[derive(Debug, Serialize, Deserialize)]
pub struct BedRow {
    pub chrom_idx: i32,
    pub pos: i64,
    pub parent1: String,
    pub parent2: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HighlightData {
    pub row: String,
    pub col: String,
    pub parent: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MatrixData {
    pub data: String,
    pub shape: Vec<usize>,
    pub dtype: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VisualizationData {
    pub status: String,
    pub matrix: MatrixData,
    pub row_labels: Vec<String>,
    pub col_labels: Vec<String>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BedVisualizationResult {
    pub success: bool,
    pub message: String,
    pub visualization_data: Option<String>,
    pub error: Option<String>,
}

fn parse_bed_file(file_path: &str) -> Result<Vec<BedRow>, String> {
    let content = fs::read_to_string(file_path)
        .map_err(|e| format!("Failed to read file: {}", e))?;
    
    let mut rows = Vec::new();
    let lines: Vec<&str> = content.lines().collect();
    
    if lines.is_empty() {
        return Err("Empty BED file".to_string());
    }
    
    // Skip header line if it exists
    let start_idx = if lines[0].starts_with("chrom") || lines[0].starts_with("#") { 1 } else { 0 };
    
    for (line_num, line) in lines.iter().enumerate().skip(start_idx) {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 4 {
            return Err(format!("Invalid BED format at line {}: expected 4 columns, found {}", line_num + 1, parts.len()));
        }
        
        let chrom_idx = parts[0].parse::<i32>()
            .map_err(|_| format!("Invalid chrom_idx at line {}: '{}'", line_num + 1, parts[0]))?;
        
        let pos = parts[1].parse::<i64>()
            .map_err(|_| format!("Invalid pos at line {}: '{}'", line_num + 1, parts[1]))?;
        
        rows.push(BedRow {
            chrom_idx,
            pos,
            parent1: parts[2].to_string(),
            parent2: parts[3].to_string(),
        });
    }
    
    if rows.is_empty() {
        return Err("No valid data rows found in BED file".to_string());
    }
    
    Ok(rows)
}

fn bed_to_matrix(bed_data: Vec<BedRow>) -> Result<(VisualizationData, Vec<HighlightData>), String> {
    // Collect unique positions (chrom_idx, pos pairs) and unique parents from parent1 and parent2 columns
    let mut position_pairs = HashSet::new();
    let mut parents = HashSet::new();
    
    for row in &bed_data {
        // Store as (chrom_idx, pos) pair for proper numeric sorting
        position_pairs.insert((row.chrom_idx, row.pos));
        // Collect all unique entries from parent1 and parent2 columns
        parents.insert(row.parent1.clone());
        parents.insert(row.parent2.clone());
    }
    
    // Sort positions numerically by chrom_idx first, then by pos
    let mut sorted_positions: Vec<(i32, i64)> = position_pairs.into_iter().collect();
    sorted_positions.sort_by(|a, b| {
        a.0.cmp(&b.0).then(a.1.cmp(&b.1))
    });
    
    // Create column labels with proper ChrIdx format
    let col_labels: Vec<String> = sorted_positions.iter()
        .map(|(chrom_idx, pos)| format!("ChrIdx{}_{}", chrom_idx, pos))
        .collect();
    
    let mut row_labels: Vec<String> = parents.into_iter().collect();
    row_labels.sort();
    
    // Create position and parent lookup maps
    let pos_to_idx: HashMap<String, usize> = col_labels.iter()
        .enumerate()
        .map(|(i, pos)| (pos.clone(), i))
        .collect();
    
    let parent_to_idx: HashMap<String, usize> = row_labels.iter()
        .enumerate()
        .map(|(i, parent)| (parent.clone(), i))
        .collect();
    
    // Create matrix: rows = unique parents, cols = unique position pairs
    let rows = row_labels.len(); // unique parents from parent1/parent2 columns
    let cols = col_labels.len(); // unique chrom_idx/pos pairs
    let mut matrix = vec![0i32; rows * cols];
    
    let mut highlights = Vec::new();
    
    // Populate matrix and create highlights
    for row in bed_data {
        let pos_label = format!("ChrIdx{}_{}", row.chrom_idx, row.pos);
        let pos_idx = pos_to_idx[&pos_label];
        
        // Mark presence for parent1
        let parent1_idx = parent_to_idx[&row.parent1];
        matrix[parent1_idx * cols + pos_idx] = 1;
        
        highlights.push(HighlightData {
            row: row.parent1.clone(),     // row = parent
            col: pos_label.clone(),       // col = position  
            parent: "parent1".to_string(),
        });
        
        // Mark presence for parent2 (always create highlight, even if same as parent1)
        let parent2_idx = parent_to_idx[&row.parent2];
        matrix[parent2_idx * cols + pos_idx] = 1;
        
        highlights.push(HighlightData {
            row: row.parent2,             // row = parent
            col: pos_label,               // col = position
            parent: "parent2".to_string(),
        });
    }
    
    // Convert matrix to base64 encoded binary format (matching Python pipeline)
    let matrix_bytes: Vec<u8> = matrix.into_iter()
        .flat_map(|i| i.to_le_bytes().to_vec())
        .collect();
    
    let matrix_base64 = general_purpose::STANDARD.encode(&matrix_bytes);
    
    let mut metadata = HashMap::new();
    metadata.insert("type".to_string(), "BED file visualization".to_string());
    metadata.insert("description".to_string(), format!("Parsed BED file with {} positions and {} parents", rows, cols));
    
    let viz_data = VisualizationData {
        status: "success".to_string(),
        matrix: MatrixData {
            data: matrix_base64,
            shape: vec![rows, cols],
            dtype: "int32".to_string(),
        },
        row_labels: row_labels,  // unique parents from parent1/parent2 columns are rows
        col_labels: col_labels, // unique chrom_idx/pos pairs are columns
        metadata,
    };
    
    Ok((viz_data, highlights))
}

#[command]
pub async fn process_bed_file(file_path: String) -> Result<BedVisualizationResult, String> {
    // Validate file exists
    if !Path::new(&file_path).exists() {
        return Ok(BedVisualizationResult {
            success: false,
            message: "BED file not found".to_string(),
            visualization_data: None,
            error: Some("File does not exist".to_string()),
        });
    }
    
    match parse_bed_file(&file_path) {
        Ok(bed_data) => {
            match bed_to_matrix(bed_data) {
                Ok((viz_data, highlights)) => {
                    // Combine visualization data with highlights
                    let result = serde_json::json!({
                        "visualization_data": viz_data,
                        "highlight_data": highlights
                    });
                    
                    Ok(BedVisualizationResult {
                        success: true,
                        message: format!("Successfully processed BED file with {} positions", viz_data.row_labels.len()),
                        visualization_data: Some(result.to_string()),
                        error: None,
                    })
                }
                Err(e) => Ok(BedVisualizationResult {
                    success: false,
                    message: "Failed to convert BED to matrix".to_string(),
                    visualization_data: None,
                    error: Some(e),
                })
            }
        }
        Err(e) => Ok(BedVisualizationResult {
            success: false,
            message: "Failed to parse BED file".to_string(),
            visualization_data: None,
            error: Some(e),
        })
    }
}