use parser_core::types::*;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use tauri::Emitter;

#[tauri::command]
pub async fn parse_bed_file(
    file_path: String,
    window: tauri::Window,
) -> Result<BEDParseResult, String> {
    let path = Path::new(&file_path);

    if !path.exists() {
        return Err(format!("File not found: {}", file_path));
    }

    let file_metadata =
        std::fs::metadata(path).map_err(|e| format!("Failed to get file metadata: {}", e))?;
    let file_size = file_metadata.len();

    let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
    let reader = BufReader::with_capacity(1024 * 1024, file);

    let window_clone = window.clone();
    parser_core::parse_bed(reader, Some(file_size), move |progress| {
        let _ = window_clone.emit("bed-progress", &progress);
    })
}

#[tauri::command]
pub async fn get_bed_chromosome_matrix(
    file_path: String,
    chromosome: String,
    window: tauri::Window,
) -> Result<BEDChromosomeMatrixResult, String> {
    let path = Path::new(&file_path);
    if !path.exists() {
        return Err(format!("File not found: {}", file_path));
    }

    let file_metadata =
        std::fs::metadata(path).map_err(|e| format!("Failed to get file metadata: {}", e))?;
    let file_size = file_metadata.len();

    let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
    let reader = BufReader::with_capacity(1024 * 1024, file);

    let window_clone = window.clone();
    parser_core::build_bed_chromosome_matrix(reader, Some(file_size), &chromosome, move |progress| {
        let _ = window_clone.emit("bed-matrix-progress", &progress);
    })
}
