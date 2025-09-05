use serde::{Deserialize, Serialize};
use std::process::Command;
use std::path::Path;
use tauri::Manager;
use crate::commands::python_bootstrap::{get_python_status, bootstrap_python};

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
pub async fn run_python_imputation(app: tauri::AppHandle, args: ImputeArgs) -> Result<ImputeResult, String> {
    // Auto-detect working directory (handle Tauri running from src-tauri/)
    let current_dir = std::env::current_dir().map_err(|e| format!("Failed to get current directory: {}", e))?;
    let project_root = if current_dir.file_name().unwrap_or_default() == "src-tauri" {
        current_dir.parent().unwrap().to_path_buf()
    } else {
        current_dir
    };

    // Try to use the bootstrap Python first, fallback to development environment
    let python_executable = match get_python_status(app.clone()).await {
        Ok(status) if status.initialized => {
            // Use bootstrapped Python
            status.python_path.ok_or("Python path not available")?
        }
        _ => {
            // In development, try to bootstrap or fallback to pixi/system python
            match bootstrap_python(app.clone()).await {
                Ok(bootstrap_result) if bootstrap_result.initialized => {
                    bootstrap_result.python_path.ok_or("Python path not available after bootstrap")?
                }
                _ => {
                    // Fallback to development environment (pixi or system python)
                    if Command::new("pixi").arg("--version").output().is_ok() {
                        return run_with_pixi_python(&project_root, args).await;
                    } else {
                        "python".to_string()
                    }
                }
            }
        }
    };

    // Find the Python script - check bundled resources first, then development path
    let script_path = match app.path().resource_dir() {
        Ok(resource_dir) => {
            // In a packaged app, Python scripts are bundled in _up_/src/python/
            // because we specified "../src/python" in tauri.conf.json resources
            let bundled_script = resource_dir.join("_up_").join("src").join("python").join("impute_with_viz.py");
            if bundled_script.exists() {
                bundled_script
            } else {
                // Fallback to development path relative to project root
                project_root.join("src").join("python").join("impute_with_viz.py")
            }
        }
        Err(_) => {
            // In development or when resource_dir fails, use development path
            project_root.join("src").join("python").join("impute_with_viz.py")
        }
    };
    
    if !script_path.exists() {
        // Add debug information to help diagnose the issue
        let debug_info = match app.path().resource_dir() {
            Ok(resource_dir) => {
                format!(
                    "Python script not found at: {}\nResource directory: {}\nProject root: {}\nBundled script would be at: {}",
                    script_path.display(),
                    resource_dir.display(),
                    project_root.display(),
                    resource_dir.join("_up_").join("src").join("python").join("impute_with_viz.py").display()
                )
            }
            Err(e) => {
                format!(
                    "Python script not found at: {}\nProject root: {}\nResource directory error: {}",
                    script_path.display(),
                    project_root.display(),
                    e
                )
            }
        };
        
        return Ok(ImputeResult {
            success: false,
            message: debug_info,
            output_file: None,
            execution_time: None,
            visualization_data: None,
        });
    }

    // Determine the correct working directory and Python path
    let (working_dir, python_path_env) = if let Ok(resource_dir) = app.path().resource_dir() {
        // In packaged app, use the bundled Python source directory as working dir
        let bundled_src_dir = resource_dir.join("_up_").join("src");
        if bundled_src_dir.exists() {
            // Set PYTHONPATH to include the bundled src directory
            (bundled_src_dir.clone(), Some(bundled_src_dir.to_string_lossy().to_string()))
        } else {
            (project_root.clone(), None)
        }
    } else {
        (project_root.clone(), None)
    };

    let mut cmd = Command::new(&python_executable);
    cmd.current_dir(&working_dir);
    
    // Set PYTHONPATH if we have bundled Python code
    if let Some(ref python_path) = python_path_env {
        cmd.env("PYTHONPATH", python_path);
    }
    
    // Debug: Log the Python executable being used
    eprintln!("DEBUG: Using Python executable: {}", python_executable);
    eprintln!("DEBUG: Script path: {}", script_path.display());
    eprintln!("DEBUG: Working directory: {}", working_dir.display());
    if let Some(ref path) = python_path_env {
        eprintln!("DEBUG: PYTHONPATH: {}", path);
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
            
            // Debug: Log the raw Python output to understand what's happening
            eprintln!("DEBUG: Python exit code: {:?}", output.status.code());
            eprintln!("DEBUG: Python stdout: {}", stdout);
            eprintln!("DEBUG: Python stderr: {}", stderr);
            
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

// Helper function to run with pixi python (for development)
async fn run_with_pixi_python(project_root: &std::path::Path, args: ImputeArgs) -> Result<ImputeResult, String> {
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

    let mut cmd = Command::new("pixi");
    cmd.current_dir(project_root);
    cmd.args(&["run", "python"]);
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
            
            // Debug: Log the raw Python output to understand what's happening
            eprintln!("DEBUG: Python exit code: {:?}", output.status.code());
            eprintln!("DEBUG: Python stdout: {}", stdout);
            eprintln!("DEBUG: Python stderr: {}", stderr);
            
            // Filter stderr to only include log messages with [INFO], [WARNING], or [ERROR] prefixes
            let filtered_stderr = stderr
                .lines()
                .filter(|line| {
                    line.contains("[INFO]") || line.contains("[WARNING]") || line.contains("[ERROR]")
                })
                .collect::<Vec<&str>>()
                .join("\n");
            
            if output.status.success() {
                let output_path_exists = Path::new(&args.output_path).exists();
                
                Ok(ImputeResult {
                    success: true,
                    message: if filtered_stderr.trim().is_empty() {
                        "Imputation completed successfully".to_string()
                    } else {
                        format!("Imputation completed successfully\n\nPYTHON_LOGS:\n{}", filtered_stderr)
                    },
                    output_file: if output_path_exists { Some(args.output_path) } else { None },
                    execution_time: Some(execution_time),
                    visualization_data: None,
                })
            } else {
                let error_message = if filtered_stderr.trim().is_empty() {
                    format!("Imputation failed with exit code {}", output.status.code().unwrap_or(-1))
                } else {
                    format!("Imputation failed with exit code {}.\n\nPYTHON_LOGS:\n{}", 
                           output.status.code().unwrap_or(-1), filtered_stderr)
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