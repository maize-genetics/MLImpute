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
}

/// Append the common imputation CLI arguments to a Command.
fn append_impute_args(cmd: &mut Command, args: &ImputeArgs) {
    cmd.arg("--input").arg(&args.input_path);
    cmd.arg("--output").arg(&args.output_path);
    cmd.arg("--model").arg(&args.model);

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
        cmd.arg("--hmm").arg("true");
    }

    if args.diploid.unwrap_or(false) {
        cmd.arg("--diploid").arg("true");
    }

    if args.collapse_bed.unwrap_or(false) {
        cmd.arg("--collapse-bed");
    }
}

/// Build an ImputeResult from the output of a finished Python process.
fn build_result(output: std::process::Output, execution_time: f64, output_path: &str) -> ImputeResult {
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Filter stderr to only include structured log messages
    let filtered_stderr: String = stderr
        .lines()
        .filter(|line| {
            line.contains("[INFO]") || line.contains("[WARNING]") || line.contains("[ERROR]")
        })
        .collect::<Vec<&str>>()
        .join("\n");

    if output.status.success() {
        let output_path_exists = Path::new(output_path).exists();

        ImputeResult {
            success: true,
            message: if filtered_stderr.trim().is_empty() {
                "Imputation completed successfully".to_string()
            } else {
                format!("Imputation completed successfully\n\nPYTHON_LOGS:\n{}", filtered_stderr)
            },
            output_file: if output_path_exists { Some(output_path.to_string()) } else { None },
            execution_time: Some(execution_time),
        }
    } else {
        let error_message = if filtered_stderr.trim().is_empty() {
            format!("Imputation failed with exit code {}", output.status.code().unwrap_or(-1))
        } else {
            format!(
                "Imputation failed with exit code {}.\n\nPYTHON_LOGS:\n{}",
                output.status.code().unwrap_or(-1),
                filtered_stderr
            )
        };

        ImputeResult {
            success: false,
            message: error_message,
            output_file: None,
            execution_time: Some(execution_time),
        }
    }
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
            status.python_path.ok_or("Python path not available")?
        }
        _ => {
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
            let bundled_script = resource_dir.join("_up_").join("src").join("python").join("impute.py");
            if bundled_script.exists() {
                bundled_script
            } else {
                project_root.join("src").join("python").join("impute.py")
            }
        }
        Err(_) => {
            project_root.join("src").join("python").join("impute.py")
        }
    };

    if !script_path.exists() {
        let debug_info = match app.path().resource_dir() {
            Ok(resource_dir) => {
                format!(
                    "Python script not found at: {}\nResource directory: {}\nProject root: {}",
                    script_path.display(),
                    resource_dir.display(),
                    project_root.display()
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
        });
    }

    // Determine the correct working directory and Python path
    let (working_dir, python_path_env) = if let Ok(resource_dir) = app.path().resource_dir() {
        let bundled_src_dir = resource_dir.join("_up_").join("src");
        if bundled_src_dir.exists() {
            (bundled_src_dir.clone(), Some(bundled_src_dir.to_string_lossy().to_string()))
        } else {
            (project_root.clone(), None)
        }
    } else {
        (project_root.clone(), None)
    };

    let mut cmd = Command::new(&python_executable);
    cmd.current_dir(&working_dir);

    if let Some(ref python_path) = python_path_env {
        cmd.env("PYTHONPATH", python_path);
    }

    cmd.arg(script_path.to_str().unwrap());
    append_impute_args(&mut cmd, &args);

    let start_time = std::time::Instant::now();

    match cmd.output() {
        Ok(output) => {
            let execution_time = start_time.elapsed().as_secs_f64();
            Ok(build_result(output, execution_time, &args.output_path))
        }
        Err(e) => {
            Ok(ImputeResult {
                success: false,
                message: format!("Failed to execute Python script: {}", e),
                output_file: None,
                execution_time: None,
            })
        }
    }
}

/// Helper function to run with pixi python (for development).
async fn run_with_pixi_python(project_root: &std::path::Path, args: ImputeArgs) -> Result<ImputeResult, String> {
    let script_path = project_root.join("src").join("python").join("impute.py");

    if !script_path.exists() {
        return Ok(ImputeResult {
            success: false,
            message: format!("Python script not found at: {}", script_path.display()),
            output_file: None,
            execution_time: None,
        });
    }

    let mut cmd = Command::new("pixi");
    cmd.current_dir(project_root);
    cmd.env("PYTHONPATH", project_root.join("src").to_string_lossy().to_string());
    cmd.args(&["run", "python"]);
    cmd.arg(script_path.to_str().unwrap());
    append_impute_args(&mut cmd, &args);

    let start_time = std::time::Instant::now();

    match cmd.output() {
        Ok(output) => {
            let execution_time = start_time.elapsed().as_secs_f64();
            Ok(build_result(output, execution_time, &args.output_path))
        }
        Err(e) => {
            Ok(ImputeResult {
                success: false,
                message: format!("Failed to execute Python script: {}", e),
                output_file: None,
                execution_time: None,
            })
        }
    }
}
