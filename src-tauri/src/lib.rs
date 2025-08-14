pub mod commands;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  tauri::Builder::default()
    // Plugins
    .plugin(tauri_plugin_os::init())
    .plugin(tauri_plugin_shell::init())

    // Commands
    .invoke_handler(tauri::generate_handler![
      commands::greet::greet,
      commands::gpu::gpu_adapters,
      commands::greet_python::greet_py,
    ])
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
}