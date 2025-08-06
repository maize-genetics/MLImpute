import reactLogo from "./assets/react.svg";
import D3Matrix from "./components/D3Matrix";
import "./App.css";

/**
 * Generates a random binary matrix and corresponding labels
 */
function generateRandomMatrix(
    numRows: number,
    numCols: number
): { matrix: number[][]; rowLabels: string[]; colLabels: string[] } {
  const matrix = Array.from({ length: numRows }, () =>
      Array.from({ length: numCols }, () => (Math.random() < 0.5 ? 1 : 0))
  );
  const rowLabels = Array.from({ length: numRows }, (_, i) => `Sample ${i + 1}`);
  const colLabels = Array.from({ length: numCols }, (_, j) => `Pos${j + 1}`);
  return { matrix, rowLabels, colLabels };
}

function App() {
  // dynamically generate a 100×50 matrix
  const { matrix: sampleMatrix, rowLabels: samples, colLabels: positions } =
      generateRandomMatrix(50, 5000);

  return (
      <main className="container">
        <h1>ML Imputation App</h1>

        <div className="row logos">
          <a href="https://vitejs.dev" target="_blank" rel="noopener noreferrer">
            <img src="/vite.svg" className="logo vite" alt="Vite logo" />
          </a>
          <a href="https://reactjs.org" target="_blank" rel="noopener noreferrer">
            <img src={reactLogo} className="logo react" alt="React logo" />
          </a>
          <a href="https://tauri.app" target="_blank" rel="noopener noreferrer">
            <img src="/tauri.svg" className="logo tauri" alt="Tauri logo" />
          </a>
        </div>

        <D3Matrix
            data={sampleMatrix}
            rowLabels={samples}
            colLabels={positions}
            margin={{ top: 40, right: 10, bottom: 10, left: 80 }}
            maxVisibleRows={20}
            maxVisibleCols={50}
        />
      </main>
  );
}

export default App;