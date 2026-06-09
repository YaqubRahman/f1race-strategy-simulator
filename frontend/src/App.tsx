import { useState } from "react";

const DRIVERS: Record<string, string> = {
  VER: "Max Verstappen",
  HAM: "Lewis Hamilton",
  LEC: "Charles Leclerc",
  NOR: "Lando Norris",
  SAI: "Carlos Sainz",
};

const CIRCUITS = [
  "Silverstone",
  "Monaco Grand Prix",
  "Italian Grand Prix",
  "Spanish Grand Prix",
  "Belgian Grand Prix",
];

const COMPOUND_COLOURS: Record<string, string> = {
  SOFT: "bg-red-500",
  MEDIUM: "bg-yellow-400",
  HARD: "bg-gray-300",
};

interface StrategyResult {
  best_strategy: number;
  best_compound1: string;
  best_compound2: string;
  best_time: string;
}

function CompoundBadge({ compound }: { compound: string }) {
  const colour = COMPOUND_COLOURS[compound] ?? "bg-gray-400";
  return (
    <span
      className={`inline-block px-3 py-1 rounded-full text-xs font-semibold text-black ${colour}`}
    >
      {compound}
    </span>
  );
}

export default function App() {
  const [driver, setDriver] = useState("VER");
  const [circuit, setCircuit] = useState("Silverstone");
  const [result, setResult] = useState<StrategyResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handlePredict() {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ driver, circuit }),
      });
      if (!res.ok) throw new Error("Something went wrong with the prediction.");
      const data: StrategyResult = await res.json();
      setResult(data);
    } catch (e) {
      setError(
        "Failed to get a prediction. Make sure the Flask server is running.",
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-gray-950 text-white font-mono flex flex-col items-center justify-center px-4 py-16">
      <div className="w-full max-w-xl">
        {/* Header */}
        <div className="mb-10 text-center">
          <p className="text-xs tracking-widest text-red-500 uppercase mb-2">
            Made by Yaqub
          </p>
          <h1 className="text-3xl font-bold tracking-tight">
            F1 Pitstop Predictor
          </h1>
          <p className="text-gray-400 text-sm mt-2">
            Select a driver and circuit to get the optimal one-stop strategy
          </p>
        </div>

        <div className="bg-gray-900 border border-gray-800 rounded-2xl p-6 mb-4">
          <div className="grid grid-cols-2 gap-4 mb-5">
            <div>
              <label className="block text-xs text-gray-400 mb-2 uppercase tracking-wider">
                Driver
              </label>
              <select
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-red-500"
                value={driver}
                onChange={(e) => setDriver(e.target.value)}
              >
                {Object.entries(DRIVERS).map(([code, name]) => (
                  <option key={code} value={code}>
                    {name}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-2 uppercase tracking-wider">
                Circuit
              </label>
              <select
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-red-500"
                value={circuit}
                onChange={(e) => setCircuit(e.target.value)}
              >
                {CIRCUITS.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <button
            onClick={handlePredict}
            disabled={loading}
            className="w-full bg-red-600 hover:bg-red-500 disabled:bg-gray-700 disabled:text-gray-500 transition-colors rounded-lg py-3 text-sm font-semibold tracking-wide"
          >
            {loading ? "Calculating strategy..." : "Predict strategy"}
          </button>
        </div>

        {error && (
          <div className="bg-red-950 border border-red-800 rounded-xl px-4 py-3 text-red-400 text-sm mb-4">
            {error}
          </div>
        )}

        {loading && (
          <div className="bg-gray-900 border border-gray-800 rounded-2xl p-6 animate-pulse">
            <div className="h-3 bg-gray-700 rounded w-1/3 mb-6"></div>
            <div className="grid grid-cols-3 gap-3 mb-6">
              {[0, 1, 2].map((i) => (
                <div key={i} className="bg-gray-800 rounded-xl h-20"></div>
              ))}
            </div>
            <div className="h-10 bg-gray-800 rounded-xl"></div>
          </div>
        )}

        {result && !loading && (
          <div className="bg-gray-900 border border-gray-800 rounded-2xl p-6">
            <p className="text-xs text-gray-500 uppercase tracking-widest mb-5">
              Optimal strategy — {DRIVERS[driver]} · {circuit}
            </p>

            <div className="grid grid-cols-3 gap-3 mb-6">
              <div className="bg-gray-800 rounded-xl p-4 text-center">
                <p className="text-xs text-gray-400 mb-1">Pit on lap</p>
                <p className="text-3xl font-bold text-white">
                  {result.best_strategy}
                </p>
              </div>
              <div className="bg-gray-800 rounded-xl p-4 text-center">
                <p className="text-xs text-gray-400 mb-2">Compounds</p>
                <div className="flex flex-col gap-1 items-center">
                  <CompoundBadge compound={result.best_compound1} />
                  <span className="text-gray-500 text-xs">→</span>
                  <CompoundBadge compound={result.best_compound2} />
                </div>
              </div>
              <div className="bg-gray-800 rounded-xl p-4 text-center">
                <p className="text-xs text-gray-400 mb-1">Race time</p>
                <p className="text-lg font-bold text-white leading-tight">
                  {result.best_time}
                </p>
              </div>
            </div>

            <div className="border-t border-gray-800 pt-4">
              <p className="text-xs text-gray-500 mb-3">Strategy breakdown</p>
              <div className="flex items-center gap-2 flex-wrap">
                <div className="bg-gray-800 rounded-lg px-3 py-2 text-xs text-gray-300">
                  Stint 1 · <CompoundBadge compound={result.best_compound1} />
                </div>
                <span className="text-gray-600 text-xs">
                  → pit lap {result.best_strategy} →
                </span>
                <div className="bg-gray-800 rounded-lg px-3 py-2 text-xs text-gray-300">
                  Stint 2 · <CompoundBadge compound={result.best_compound2} />
                </div>
              </div>
            </div>
          </div>
        )}

        <p className="text-center text-xs text-gray-600 mt-6">
          Based on historical race data · 2018–2024
        </p>
      </div>
    </div>
  );
}
