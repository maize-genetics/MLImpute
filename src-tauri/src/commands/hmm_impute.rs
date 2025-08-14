use serde::{Deserialize, Serialize};
use tauri::command;
use std::process::Command;
use std::path::Path;

#[derive(Serialize, Deserialize)]
pub struct HmmImputeArgs {
    pub input_path: String,
    pub output_bed: String,
    pub global_weights: String,
    pub ps4g_file: String,
    pub diploid: bool,
}

#[derive(Serialize, Deserialize)]
pub struct HmmImputeResult {
    pub success: bool,
    pub message: String,
    pub output_file: Option<String>,
}

#[command]
pub async fn run_hmm_imputation(args: HmmImputeArgs) -> Result<HmmImputeResult, String> {
    // Navigate to project root (handling case where we're in src-tauri/)
    let current_dir = std::env::current_dir().map_err(|e| e.to_string())?;
    let project_root = if current_dir.file_name().unwrap_or_default() == "src-tauri" {
        current_dir.parent().unwrap().to_path_buf()
    } else {
        current_dir
    };

    // Check if pixi is available
    let use_pixi = Command::new("pixi")
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false);

    // Build the command
    let mut cmd = if use_pixi {
        let mut c = Command::new("pixi");
        c.arg("run").arg("python");
        c
    } else {
        Command::new("python")
    };

    // Add the script path and arguments
    cmd.arg("src/python/hmm/hmm_impute.py")
        .arg("--input-path").arg(&args.input_path)
        .arg("--output-bed").arg(&args.output_bed)
        .arg("--global-weights").arg(&args.global_weights)
        .arg("--ps4g-file").arg(&args.ps4g_file);

    if args.diploid {
        cmd.arg("--diploid");
    }

    // Set working directory to project root
    cmd.current_dir(&project_root);

    // Execute the command
    match cmd.output() {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            
            if output.status.success() {
                let output_exists = Path::new(&args.output_bed).exists();
                Ok(HmmImputeResult {
                    success: true,
                    message: format!("HMM imputation completed successfully.\nStdout: {}\nStderr: {}", stdout, stderr),
                    output_file: if output_exists { Some(args.output_bed) } else { None },
                })
            } else {
                Ok(HmmImputeResult {
                    success: false,
                    message: format!("HMM imputation failed.\nStdout: {}\nStderr: {}", stdout, stderr),
                    output_file: None,
                })
            }
        }
        Err(e) => Err(format!("Failed to execute command: {}", e)),
    }
}