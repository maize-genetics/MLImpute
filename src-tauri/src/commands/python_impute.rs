use serde::{Deserialize, Serialize};
use std::process::Command;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct ImputeArgs {
    pub input_path: String,
    pub output_path: String,
    pub model: String,
    pub weight: Option<String>,
    pub collapse: Option<bool>,
    pub verbose: Option<bool>,
    pub global_weights: Option<String>,
    pub hmm: Option<bool>,
    pub diploid: Option<bool>,
    pub collapse_bed: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct ImputeResult {
    pub success: bool,
    pub message: String,
    pub output_file: Option<String>,
    pub execution_time: Option<f64>,
    pub visualization_data: Option<String>,
}

#[tauri::command]
pub async fn run_python_imputation(args: ImputeArgs) -> Result<ImputeResult, String> {
    // Auto-detect working directory (handle Tauri running from src-tauri/)
    let current_dir = std::env::current_dir().map_err(|e| format!("Failed to get current directory: {}", e))?;
    let project_root = if current_dir.file_name().unwrap_or_default() == "src-tauri" {
        current_dir.parent().unwrap().to_path_buf()
    } else {
        current_dir
    };

    // Auto-detect pixi vs system python
    let (python_cmd, python_args) = if Command::new("pixi").arg("--version").output().is_ok() {
        ("pixi", vec!["run", "python"])
    } else {
        ("python", vec![])
    };

    // Build the command - use the visualization version
    let script_path = project_root.join("src/python/impute_with_viz.py");
    
    if !script_path.exists() {
        return Ok(ImputeResult {
            success: false,
            message: format!("Python script not found at: {}", script_path.display()),
            output_file: None,
            execution_time: None,
            visualization_data: None,
        });
    }

    let mut cmd = Command::new(python_cmd);
    cmd.current_dir(&project_root);
    
    // Add pixi run arguments if using pixi
    for arg in python_args {
        cmd.arg(arg);
    }
    
    // Add the script path
    cmd.arg(script_path.to_str().unwrap());
    
    // Add required arguments
    cmd.arg("--input").arg(&args.input_path);
    cmd.arg("--output").arg(&args.output_path);
    cmd.arg("--model").arg(&args.model);
    
    // Add optional arguments
    if let Some(weight) = &args.weight {
        cmd.arg("--weight").arg(weight);
    }
    
    if args.collapse.unwrap_or(false) {
        cmd.arg("--collapse");
    }
    
    if args.verbose.unwrap_or(false) {
        cmd.arg("--verbose");
    }
    
    if let Some(global_weights) = &args.global_weights {
        cmd.arg("--global-weights").arg(global_weights);
    }
    
    if args.hmm.unwrap_or(false) {
        cmd.arg("--HMM").arg("true");
    }
    
    if args.diploid.unwrap_or(false) {
        cmd.arg("--diploid").arg("true");
    }
    
    if args.collapse_bed.unwrap_or(false) {
        cmd.arg("--collapse-bed");
    }

    let start_time = std::time::Instant::now();
    
    // Execute the command
    match cmd.output() {
        Ok(output) => {
            let execution_time = start_time.elapsed().as_secs_f64();
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            
            if output.status.success() {
                let output_path_exists = Path::new(&args.output_path).exists();
                
                // Parse the JSON output to extract visualization data
                let visualization_data = if !stdout.trim().is_empty() {
                    // Try to parse the stdout as JSON to extract visualization data
                    match serde_json::from_str::<serde_json::Value>(&stdout) {
                        Ok(json) => {
                            if let Some(viz_data) = json.get("visualization_data") {
                                // The visualization_data is a string containing JSON, so we extract it as a string
                                if let Some(viz_str) = viz_data.as_str() {
                                    Some(viz_str.to_string())
                                } else {
                                    // If it's already a JSON object, convert it back to string
                                    Some(viz_data.to_string())
                                }
                            } else {
                                // If no visualization_data field, return the whole stdout
                                Some(stdout.to_string())
                            }
                        }
                        Err(_) => Some(stdout.to_string())
                    }
                } else {
                    None
                };
                
                Ok(ImputeResult {
                    success: true,
                    message: format!("Imputation completed successfully.\n\nSTDOUT:\n{}\n\nSTDERR:\n{}", stdout, stderr),
                    output_file: if output_path_exists { Some(args.output_path) } else { None },
                    execution_time: Some(execution_time),
                    visualization_data,
                })
            } else {
                Ok(ImputeResult {
                    success: false,
                    message: format!("Imputation failed with exit code {}.\n\nSTDOUT:\n{}\n\nSTDERR:\n{}", 
                                   output.status.code().unwrap_or(-1), stdout, stderr),
                    output_file: None,
                    execution_time: Some(execution_time),
                    visualization_data: None,
                })
            }
        }
        Err(e) => {
            Ok(ImputeResult {
                success: false,
                message: format!("Failed to execute Python script: {}", e),
                output_file: None,
                execution_time: None,
                visualization_data: None,
            })
        }
    }
}