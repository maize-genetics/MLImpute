use serde::{Deserialize, Serialize};
use tauri_plugin_shell::ShellExt;

#[derive(Serialize, Deserialize, Debug)]
pub struct NumpyArray {
    pub data: String,
    pub shape: Vec<usize>,
    pub dtype: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct VisualizationData {
    pub status: String,
    pub matrix: Option<NumpyArray>,
    pub row_labels: Option<Vec<String>>,
    pub col_labels: Option<Vec<String>>,
    pub metadata: Option<serde_json::Value>,
    pub error: Option<String>,
}

#[tauri::command]
pub async fn get_sample_visualization_data(
    app: tauri::AppHandle,
    rows: Option<u32>,
    cols: Option<u32>,
    seed: Option<u32>,
) -> Result<VisualizationData, String> {
    let shell = app.shell();
    // Get the current working directory and navigate to project root
    let current_dir = std::env::current_dir().map_err(|e| e.to_string())?;
    
    // If we're in src-tauri directory, go up one level to project root
    let project_root = if current_dir.file_name().map(|name| name == "src-tauri").unwrap_or(false) {
        current_dir.parent().ok_or("Cannot find project root")?.to_path_buf()
    } else {
        current_dir
    };
    
    let script_path = project_root.join("src/python/array_utils.py");
    
    println!("Looking for Python script at: {}", script_path.display());
    
    if !script_path.exists() {
        let error_msg = format!("Python script not found at: {}", script_path.display());
        println!("{}", error_msg);
        return Err(error_msg);
    }

    let mut args = vec![script_path.to_string_lossy().to_string()];
    
    if let Some(r) = rows {
        args.extend(vec!["--rows".to_string(), r.to_string()]);
    }
    if let Some(c) = cols {
        args.extend(vec!["--cols".to_string(), c.to_string()]);
    }
    if let Some(s) = seed {
        args.extend(vec!["--seed".to_string(), s.to_string()]);
    }

    // Try to use pixi python if available, otherwise fall back to system python
    let has_pixi = std::process::Command::new("pixi").arg("--version").output().is_ok();
    println!("Has pixi: {}", has_pixi);
    
    let cmd = if has_pixi {
        println!("Using pixi to run python with args: {:?}", args);
        shell.command("pixi").args(&["run", "python"]).args(&args)
    } else {
        println!("Using system python with args: {:?}", args);
        shell.command("python").args(&args)
    };

    println!("Executing command...");
    let out = cmd.output().await.map_err(|e| {
        println!("Command execution failed: {}", e);
        e.to_string()
    })?;
    
    println!("Command exit status: {:?}", out.status);
    println!("Command stdout: {}", String::from_utf8_lossy(&out.stdout));
    println!("Command stderr: {}", String::from_utf8_lossy(&out.stderr));
    
    if out.status.success() {
        println!("Command succeeded, parsing JSON...");
        let stdout_str = String::from_utf8_lossy(&out.stdout);
        let result: VisualizationData = serde_json::from_str(&stdout_str)
            .map_err(|e| {
                println!("JSON parse failed. Raw output: '{}'", stdout_str);
                println!("Parse error: {}", e);
                format!("JSON parse error: {} (raw output: {})", e, stdout_str)
            })?;
        Ok(result)
    } else {
        let error_msg = String::from_utf8_lossy(&out.stderr);
        Ok(VisualizationData {
            status: "error".to_string(),
            matrix: None,
            row_labels: None,
            col_labels: None,
            metadata: None,
            error: Some(error_msg.to_string()),
        })
    }
}

#[tauri::command]
pub async fn run_imputation_visualization(
    app: tauri::AppHandle,
    input_file: String,
    model: String,
    weight: Option<String>,
) -> Result<VisualizationData, String> {
    let shell = app.shell();
    // Get the current working directory and navigate to project root
    let current_dir = std::env::current_dir().map_err(|e| e.to_string())?;
    
    // If we're in src-tauri directory, go up one level to project root
    let project_root = if current_dir.file_name().map(|name| name == "src-tauri").unwrap_or(false) {
        current_dir.parent().ok_or("Cannot find project root")?.to_path_buf()
    } else {
        current_dir
    };
    
    let script_path = project_root.join("src/impute.py");
    
    if !script_path.exists() {
        return Err(format!("Python script not found at: {}", script_path.display()));
    }

    // Create temporary output file
    let temp_output = std::env::temp_dir().join("imputation_result.bed");
    
    let mut args = vec![
        script_path.to_string_lossy().to_string(),
        "--input".to_string(),
        input_file,
        "--output".to_string(),
        temp_output.to_string_lossy().to_string(),
        "--model".to_string(),
        model,
    ];
    
    if let Some(w) = weight {
        args.extend(vec!["--weight".to_string(), w]);
    }

    // Try to use pixi python if available, otherwise fall back to system python
    let cmd = if std::process::Command::new("pixi").arg("--version").output().is_ok() {
        shell.command("pixi").args(&["run", "python"]).args(&args)
    } else {
        shell.command("python").args(&args)
    };

    let out = cmd.output().await.map_err(|e| e.to_string())?;
    if out.status.success() {
        // For now, return success status - you can extend this to process the actual results
        Ok(VisualizationData {
            status: "success".to_string(),
            matrix: None,
            row_labels: None,
            col_labels: None,
            metadata: Some(serde_json::json!({
                "message": "Imputation completed successfully",
                "output_file": temp_output.to_string_lossy()
            })),
            error: None,
        })
    } else {
        let error_msg = String::from_utf8_lossy(&out.stderr);
        Ok(VisualizationData {
            status: "error".to_string(),
            matrix: None,
            row_labels: None,
            col_labels: None,
            metadata: None,
            error: Some(error_msg.to_string()),
        })
    }
}