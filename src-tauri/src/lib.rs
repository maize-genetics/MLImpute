pub mod commands;

use commands::ps4g_parser::PS4GCache;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  tauri::Builder::default()
    // Plugins
    .plugin(tauri_plugin_os::init())
    .plugin(tauri_plugin_shell::init())
    .plugin(tauri_plugin_dialog::init())
    .plugin(tauri_plugin_fs::init())

    // Managed state for PS4G file caching
    .manage(PS4GCache::new())

    // Commands
    .invoke_handler(tauri::generate_handler![
      commands::gpu::gpu_adapters,
      commands::python_impute::run_python_imputation,
      commands::python_bootstrap::bootstrap_python,
      commands::python_bootstrap::get_python_status,
      commands::python_bootstrap::run_python_command,
      commands::python_bootstrap::reset_python_environment,
      commands::ps4g_parser::parse_ps4g_file,
      commands::ps4g_parser::get_chromosome_matrix,
      commands::ps4g_parser::get_chromosome_matrix_binary,
      commands::bed_parser::parse_bed_file,
      commands::bed_parser::get_bed_chromosome_matrix,
      commands::npy_overlay::load_npy_overlay,
    ])
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
}