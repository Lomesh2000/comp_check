export default function ConfigPanel({ config, setConfig }) {
  return (
    <div className="bg-gray-800 p-4 rounded-lg space-y-4">
      <h2 className="text-lg font-semibold">Settings</h2>

      {/* Lambda Threshold */}
      <div>
        <label>Lambda Threshold: {config.lambda_thresh}</label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={config.lambda_thresh}
          onChange={(e) => {
            const val = parseFloat(e.target.value);
            setConfig({
                ...config,
                lambda_thresh: isNaN(val) ? 0 : val,
            });
            }}
          className="w-full"
        />
      </div>

      {/* Hop K */}
      <div>
        <label>Hop K:</label>
        <input
          type="number"
          value={config.hop_k}
          onChange={(e) => {
            const val = parseInt(e.target.value);
            setConfig({
                ...config,
                hop_k: isNaN(val) ? 1 : val,
            });
            }}
          className="w-full bg-gray-900 p-2 rounded"
        />
      </div>

      {/* Max Triples */}
      <div>
        <label>Max Triples:</label>
        <input
          type="number"
          value={config.max_triples}
          onChange={(e) => {
            const val = parseInt(e.target.value);
            setConfig({
                ...config,
                max_triples: isNaN(val) ? 60 : val,
            });
            }}
          className="w-full bg-gray-900 p-2 rounded"
        />
      </div>

      {/* Prefer Local */}
      <div>
        <label>
          <input
            type="checkbox"
            checked={config.prefer_local}
            onChange={(e) =>
              setConfig({
                ...config,
                prefer_local: e.target.checked,
              })
            }
          />
          Prefer Local Model
        </label>
      </div>
    </div>
  );
}