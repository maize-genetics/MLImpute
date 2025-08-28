pub mod commands;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  tauri::Builder::default()
    // Plugins
    .plugin(tauri_plugin_os::init())
    .plugin(tauri_plugin_shell::init())
    .plugin(tauri_plugin_dialog::init())

    // Commands
    .invoke_handler(tauri::generate_handler![
      commands::gpu::gpu_adapters,
      commands::visualization_data::run_imputation_visualization,
      commands::python_impute::run_python_imputation,
      commands::bed_parser::process_bed_file,
    ])
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
}