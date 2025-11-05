use std::{path::PathBuf, process::Command, fs};
use tauri::{Manager, AppHandle};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct BootstrapStatus {
    pub initialized: bool,
    pub venv_path: Option<String>,
    pub python_path: Option<String>,
    pub error: Option<String>,
}

/// Bootstrap Python environment at first run
/// This creates a virtual environment using the bundled Python runtime
/// and installs dependencies from the bundled wheelhouse
#[tauri::command]
pub async fn bootstrap_python(app: AppHandle) -> Result<BootstrapStatus, String> {
    let app_data_dir = app.path().app_data_dir()
        .map_err(|e| format!("Failed to get app data directory: {}", e))?;
    
    let embedded_python = app.path().resource_dir()
        .map_err(|e| format!("Failed to get resource directory: {}", e))?
        .join("resources")
        .join("python-runtime");
    
    let wheelhouse = app.path().resource_dir()
        .map_err(|e| format!("Failed to get resource directory: {}", e))?
        .join("resources")
        .join("wheelhouse");
    
    let venv_dir = app_data_dir.join("python-env");
    
    // Debug logging
    eprintln!("DEBUG BOOTSTRAP: App data dir: {}", app_data_dir.display());
    eprintln!("DEBUG BOOTSTRAP: Embedded Python dir: {}", embedded_python.display());
    eprintln!("DEBUG BOOTSTRAP: Wheelhouse dir: {}", wheelhouse.display());
    eprintln!("DEBUG BOOTSTRAP: Venv dir: {}", venv_dir.display());
    
    // Check if already bootstrapped
    if venv_dir.exists() && is_venv_valid(&venv_dir) {
        return Ok(BootstrapStatus {
            initialized: true,
            venv_path: Some(venv_dir.to_string_lossy().to_string()),
            python_path: Some(get_venv_python(&venv_dir).to_string_lossy().to_string()),
            error: None,
        });
    }
    
    // Create app data directory if it doesn't exist
    if !app_data_dir.exists() {
        fs::create_dir_all(&app_data_dir)
            .map_err(|e| format!("Failed to create app data directory: {}", e))?;
    }
    
    // Get embedded Python executable path
    let embedded_python_exe = get_embedded_python_exe(&embedded_python)?;
    
    eprintln!("DEBUG BOOTSTRAP: Embedded Python exe: {}", embedded_python_exe.display());
    eprintln!("DEBUG BOOTSTRAP: Embedded Python exe exists: {}", embedded_python_exe.exists());
    
    if !embedded_python_exe.exists() {
        return Err(format!("Embedded Python not found at: {}", embedded_python_exe.display()));
    }
    
    // Remove existing venv if corrupted
    if venv_dir.exists() {
        fs::remove_dir_all(&venv_dir)
            .map_err(|e| format!("Failed to remove corrupted venv: {}", e))?;
    }
    
    // Create virtual environment
    let venv_status = Command::new(&embedded_python_exe)
        .args(["-m", "venv", venv_dir.to_str().unwrap()])
        .status()
        .map_err(|e| format!("Failed to execute venv command: {}", e))?;
    
    if !venv_status.success() {
        return Ok(BootstrapStatus {
            initialized: false,
            venv_path: None,
            python_path: None,
            error: Some("Failed to create virtual environment".to_string()),
        });
    }
    
    // Install dependencies from wheelhouse if available
    if wheelhouse.exists() {
        install_from_wheelhouse(&venv_dir, &wheelhouse, &app)?;
    }
    
    Ok(BootstrapStatus {
        initialized: true,
        venv_path: Some(venv_dir.to_string_lossy().to_string()),
        python_path: Some(get_venv_python(&venv_dir).to_string_lossy().to_string()),
        error: None,
    })
}

/// Get the status of the Python environment without bootstrapping
#[tauri::command]
pub async fn get_python_status(app: AppHandle) -> Result<BootstrapStatus, String> {
    let app_data_dir = app.path().app_data_dir()
        .map_err(|e| format!("Failed to get app data directory: {}", e))?;
    
    let venv_dir = app_data_dir.join("python-env");
    
    eprintln!("DEBUG STATUS: App data dir: {}", app_data_dir.display());
    eprintln!("DEBUG STATUS: Venv dir: {}", venv_dir.display());
    eprintln!("DEBUG STATUS: Venv exists: {}", venv_dir.exists());
    
    if venv_dir.exists() && is_venv_valid(&venv_dir) {
        Ok(BootstrapStatus {
            initialized: true,
            venv_path: Some(venv_dir.to_string_lossy().to_string()),
            python_path: Some(get_venv_python(&venv_dir).to_string_lossy().to_string()),
            error: None,
        })
    } else {
        Ok(BootstrapStatus {
            initialized: false,
            venv_path: None,
            python_path: None,
            error: None,
        })
    }
}

/// Reset/delete the Python environment to force re-bootstrap
#[tauri::command]
pub async fn reset_python_environment(app: AppHandle) -> Result<String, String> {
    let app_data_dir = app.path().app_data_dir()
        .map_err(|e| format!("Failed to get app data directory: {}", e))?;
    
    let venv_dir = app_data_dir.join("python-env");
    
    if venv_dir.exists() {
        fs::remove_dir_all(&venv_dir)
            .map_err(|e| format!("Failed to remove Python environment: {}", e))?;
        Ok(format!("Python environment reset. Deleted: {}", venv_dir.display()))
    } else {
        Ok("Python environment was not found, nothing to reset".to_string())
    }
}

/// Run a Python command using the bootstrapped environment
#[tauri::command]
pub async fn run_python_command(app: AppHandle, args: Vec<String>) -> Result<String, String> {
    let status = get_python_status(app.clone()).await?;
    
    if !status.initialized {
        let bootstrap_result = bootstrap_python(app.clone()).await?;
        if !bootstrap_result.initialized {
            return Err("Failed to bootstrap Python environment".to_string());
        }
    }
    
    let python_path = if let Some(path) = status.python_path {
        path
    } else {
        let bootstrap_result = bootstrap_python(app.clone()).await?;
        bootstrap_result.python_path
            .ok_or("Python path not available after bootstrap")?
    };
    
    let output = Command::new(python_path)
        .args(args)
        .output()
        .map_err(|e| format!("Failed to execute Python command: {}", e))?;
    
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).to_string())
    }
}

// Helper functions

fn get_embedded_python_exe(embedded_python: &PathBuf) -> Result<PathBuf, String> {
    #[cfg(target_os = "windows")]
    let python_exe = embedded_python.join("python.exe");
    
    #[cfg(not(target_os = "windows"))]
    let python_exe = embedded_python.join("bin").join("python3");
    
    Ok(python_exe)
}

fn get_venv_python(venv_dir: &PathBuf) -> PathBuf {
    #[cfg(target_os = "windows")]
    return venv_dir.join("Scripts").join("python.exe");
    
    #[cfg(not(target_os = "windows"))]
    return venv_dir.join("bin").join("python");
}

fn get_venv_pip(venv_dir: &PathBuf) -> PathBuf {
    #[cfg(target_os = "windows")]
    return venv_dir.join("Scripts").join("pip.exe");
    
    #[cfg(not(target_os = "windows"))]
    return venv_dir.join("bin").join("pip");
}

fn is_venv_valid(venv_dir: &PathBuf) -> bool {
    let python_exe = get_venv_python(venv_dir);
    python_exe.exists() && python_exe.is_file()
}

fn install_from_wheelhouse(venv_dir: &PathBuf, wheelhouse: &PathBuf, app: &AppHandle) -> Result<(), String> {
    let pip_exe = get_venv_pip(venv_dir);
    
    if !pip_exe.exists() {
        return Err("pip not found in virtual environment".to_string());
    }
    
    // Check if bundled requirements.txt exists and use it for installation
    let app_resource_dir = app.path().resource_dir()
        .map_err(|e| format!("Failed to get resource directory: {}", e))?;
    let bundled_requirements = app_resource_dir.join("_up_").join("requirements.txt");
    
    let status = if bundled_requirements.exists() {
        eprintln!("DEBUG BOOTSTRAP: Installing from bundled requirements: {}", bundled_requirements.display());
        // Install from requirements file
        Command::new(pip_exe)
            .args([
                "install",
                "--no-index",
                "--find-links",
                wheelhouse.to_str().unwrap(),
                "--force-reinstall",
                "-r",
                bundled_requirements.to_str().unwrap()
            ])
            .status()
            .map_err(|e| format!("Failed to execute pip install: {}", e))?
    } else {
        eprintln!("DEBUG BOOTSTRAP: No bundled requirements, installing key packages");
        // Fallback to installing key packages
        Command::new(pip_exe)
            .args([
                "install",
                "--no-index",
                "--find-links",
                wheelhouse.to_str().unwrap(),
                "--force-reinstall",
                // Install all required dependencies from requirements.txt
                "numpy", "pandas", "scikit-learn", "torch", "torchvision", 
                "transformers", "pytest", "pytest-cov", "lightning", "tqdm",
                "numba", "wandb"
            ])
            .status()
            .map_err(|e| format!("Failed to execute pip install: {}", e))?
    };
    
    if !status.success() {
        return Err("Failed to install packages from wheelhouse".to_string());
    }
    
    Ok(())
}