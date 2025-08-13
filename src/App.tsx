import "./App.css";

import D3Matrix from "./components/D3Matrix";
import SystemInfoTable, { type AdapterInfo } from "./components/SystemInfoTable";
import { generateRandomMatrix, generateRandomHighlights } from "./components/utils";

import { arch, platform } from "@tauri-apps/plugin-os";
import { invoke } from "@tauri-apps/api/core";
import {useEffect, useState} from "react";

type SystemState = {
  platform?: string;
  arch?: string;
};

function App() {
  const [system, setSystem] = useState<SystemState>({});
  const [adapters, setAdapters] = useState<AdapterInfo[] | null>(null);

  // demo matrix data (adjust sizes as you like)
  const { matrix: sampleMatrix, rowLabels: samples, colLabels: positions } =
    generateRandomMatrix(50, 1000);
  const highlights = generateRandomHighlights(samples, positions);

  useEffect(() => {
    let canceled = false;

    (async () => {
      try {
        const [p, a] = await Promise.all([platform(), arch()]);
        if (!canceled) setSystem({ platform: p, arch: a });
      } catch (err) {
        console.error("OS probe failed:", err);
      }

      try {
        const gpus = await invoke<AdapterInfo[]>("gpu_adapters");
        if (!canceled) setAdapters(gpus);
        console.table(gpus);
      } catch (err) {
        console.error("GPU probe failed:", err);
      }
    })();

    return () => {
      canceled = true;
    };
  }, []);

  return (
    <main className="container">
      <h1>ML Imputation App</h1>
      <h2>System info & matrix demo</h2>

      <SystemInfoTable system={system} adapters={adapters} />

      <D3Matrix
        data={sampleMatrix}
        rowLabels={samples}
        colLabels={positions}
        highlightData={highlights}
        margin={{ top: 40, right: 10, bottom: 10, left: 80 }}
        maxVisibleRows={20}
        maxVisibleCols={50}
      />
    </main>
  );
}

export default App;