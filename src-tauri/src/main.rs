// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

// Learn more about Tauri commands at https://v1.tauri.app/v1/guides/features/command
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[tauri::command]
fn greet_py(name: &str) -> String {
    let output = std::process::Command::new("python3")
        .arg("../src/python/greet.py")  // adjust path if needed
        .arg(name)
        .output()
        .expect("Failed to execute Python script");

    if output.status.success() {
        String::from_utf8_lossy(&output.stdout).into_owned()
    } else {
        String::from_utf8_lossy(&output.stderr).into_owned()
    }
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![greet, greet_py])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
