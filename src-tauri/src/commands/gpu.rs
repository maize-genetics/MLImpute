use serde::Serialize;

#[derive(Serialize)]
pub struct AdapterInfo {
  pub name: String,
  pub backend: String,
  pub device_type: String,
  pub vendor: u32,
  pub device: u32,
}

#[tauri::command]
pub fn gpu_adapters() -> Vec<AdapterInfo> {
  pollster::block_on(async {
    let instance = wgpu::Instance::default();
    instance
      .enumerate_adapters(wgpu::Backends::all())
      .into_iter()
      .map(|a| {
        let i = a.get_info();
        AdapterInfo {
          name: i.name,
          backend: format!("{:?}", i.backend),
          device_type: format!("{:?}", i.device_type),
          vendor: i.vendor,
          device: i.device,
        }
      })
      .collect()
  })
}
