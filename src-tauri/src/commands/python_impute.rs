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
            
            // Filter stderr to only include log messages with [INFO], [WARNING], or [ERROR] prefixes
            let filtered_stderr = stderr
                .lines()
                .filter(|line| {
                    line.contains("[INFO]") || line.contains("[WARNING]") || line.contains("[ERROR]")
                })
                .collect::<Vec<&str>>()
                .join("\n");
            
            // First, try to parse the JSON response from Python to get the actual success status
            let parsed_result = if !stdout.trim().is_empty() {
                match serde_json::from_str::<serde_json::Value>(&stdout) {
                    Ok(json) => Some(json),
                    Err(_) => None
                }
            } else {
                None
            };
            
            // Check success status from both process exit code and JSON response
            let is_successful = output.status.success() && 
                parsed_result.as_ref()
                    .and_then(|json| json.get("success"))
                    .and_then(|success| success.as_bool())
                    .unwrap_or(false);
            
            if is_successful {
                let output_path_exists = Path::new(&args.output_path).exists();
                
                // Extract visualization data from the parsed JSON
                let visualization_data = parsed_result.as_ref()
                    .and_then(|json| json.get("visualization_data"))
                    .map(|viz_data| {
                        if let Some(viz_str) = viz_data.as_str() {
                            viz_str.to_string()
                        } else {
                            // If it's already a JSON object, convert it back to string
                            viz_data.to_string()
                        }
                    });
                
                // Get the message from the JSON response or use a default
                let message = parsed_result.as_ref()
                    .and_then(|json| json.get("message"))
                    .and_then(|msg| msg.as_str())
                    .unwrap_or("Imputation completed successfully");
                
                Ok(ImputeResult {
                    success: true,
                    message: if filtered_stderr.trim().is_empty() {
                        message.to_string()
                    } else {
                        format!("{}\n\nPYTHON_LOGS:\n{}", message, filtered_stderr)
                    },
                    output_file: if output_path_exists { Some(args.output_path) } else { None },
                    execution_time: Some(execution_time),
                    visualization_data,
                })
            } else {
                // Get the error message from JSON response if available, otherwise use process error
                let error_message = if !output.status.success() {
                    if filtered_stderr.trim().is_empty() {
                        format!("Imputation failed with exit code {}", output.status.code().unwrap_or(-1))
                    } else {
                        format!("Imputation failed with exit code {}.\n\nPYTHON_LOGS:\n{}", 
                               output.status.code().unwrap_or(-1), filtered_stderr)
                    }
                } else {
                    // Process succeeded but JSON indicates failure
                    let json_message = parsed_result.as_ref()
                        .and_then(|json| json.get("message"))
                        .and_then(|msg| msg.as_str())
                        .unwrap_or("Imputation failed");
                    if filtered_stderr.trim().is_empty() {
                        json_message.to_string()
                    } else {
                        format!("{}\n\nPYTHON_LOGS:\n{}", json_message, filtered_stderr)
                    }
                };
                
                Ok(ImputeResult {
                    success: false,
                    message: error_message,
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