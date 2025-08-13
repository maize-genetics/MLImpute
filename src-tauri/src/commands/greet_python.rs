use tauri::{Manager, path::BaseDirectory};
use tauri_plugin_shell::ShellExt;

#[tauri::command]
pub async fn greet_py(app: tauri::AppHandle, name: String) -> Result<String, String> {
  let shell = app.shell();
  let script = app
    .path()
    .resolve("../src/python/greet.py", BaseDirectory::Resource)
    .map_err(|e| e.to_string())?;

  let cmd = shell.command("python").args([script.to_string_lossy().as_ref(), &name]);

  let out = cmd.output().await.map_err(|e| e.to_string())?;
  if out.status.success() {
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
  } else {
    Err(String::from_utf8_lossy(&out.stderr).into_owned())
  }
}
