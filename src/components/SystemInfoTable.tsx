export type SystemInfo = {
  platform?: string;
  arch?: string;
};

export type AdapterInfo = {
  name: string;
  backend: string;
  device_type: string;
  vendor: number;
  device: number;
};

export default function SystemInfoTable(
  {
    system,
    adapters,
  }: {
  system: SystemInfo;
  adapters: AdapterInfo[] | null;
}) {
  return (
    <div className="sysinfo-tables">
      <table className="table kv">
        <tbody>
        <tr>
          <th>OS Platform</th>
          <td>{system.platform ?? "…"}</td>
        </tr>
        <tr>
          <th>CPU Architecture</th>
          <td>{system.arch ?? "…"}</td>
        </tr>
        <tr>
          <th>GPU Adapter Count</th>
          <td>{adapters ? adapters.length : "…"}</td>
        </tr>
        </tbody>
      </table>

      {adapters && adapters.length > 0 && (
        <table className="table adapters">
          <thead>
          <tr>
            <th>#</th>
            <th>Name</th>
            <th>Type</th>
            <th>Backend</th>
            <th>Vendor</th>
            <th>Device</th>
          </tr>
          </thead>
          <tbody>
          {adapters.map((a, i) => (
            <tr key={`${a.name}-${i}`}>
              <td>{i + 1}</td>
              <td>{a.name}</td>
              <td>{a.device_type}</td>
              <td>{a.backend}</td>
              <td>{`0x${a.vendor.toString(16)}`}</td>
              <td>{`0x${a.device.toString(16)}`}</td>
            </tr>
          ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
