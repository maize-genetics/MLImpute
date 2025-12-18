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
      commands::python_bootstrap::bootstrap_python,
      commands::python_bootstrap::get_python_status,
      commands::python_bootstrap::run_python_command,
      commands::python_bootstrap::reset_python_environment,
      commands::ps4g_parser::parse_ps4g_file,
    ])
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
}