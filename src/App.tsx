import D3Matrix from "./components/D3Matrix";
import "./App.css";
import { generateRandomMatrix, generateRandomHighlights } from "./components/utils.ts";


function App() {
  const { matrix: sampleMatrix, rowLabels: samples, colLabels: positions } =
    generateRandomMatrix(50, 1000);

  // Generate random highlights
  const highlights = generateRandomHighlights(samples, positions);

  return (
    <main className="container">
      <h1>ML Imputation App</h1>
      <h2>(React + D3 Tests)</h2>

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