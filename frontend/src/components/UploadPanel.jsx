import { useState } from "react";
import { complianceCheck, uploadPdfChunks } from "../services/api";
 import ConfigPanel from "./ConfigPanel";

export default function UploadPanel({ onResult }) {
  const [text, setText] = useState("");
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);

  // Config (we’ll make UI for this later)

    const [config, setConfig] = useState({
    chunk_size: 500,
    chunk_overlap: 100,
    lambda_thresh: 0.75,
    hop_k: 1,
    max_triples: 60,
    prefer_local: false,
    });

  // 🔹 Handle text submission
  const handleTextSubmit = async () => {
    if (!text.trim()) return;

    setLoading(true);
    try {
      const res = await complianceCheck({
        text,
        ...config,
      });
      onResult({ type: "text", data: res });
    } catch (err) {
    console.error("FULL ERROR:", err);

    if (err.response) {
        console.error("Backend Error:", err.response.data);
        alert(JSON.stringify(err.response.data));
    } else {
        alert(err.message);
    }
    } finally {
      setLoading(false);
    }
  };

  // 🔹 Handle PDF upload
  const handleFileSubmit = async () => {
    if (!file) return;

    setLoading(true);
    try {
      const res = await uploadPdfChunks(file, config);
      onResult({ type: "pdf", data: res });
    } catch (err) {
    console.error("FULL ERROR:", err);

    if (err.response) {
        console.error("Backend Error:", err.response.data);
        alert(JSON.stringify(err.response.data));
    } else {
        alert(err.message);
    }
    }finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-gray-800 p-4 rounded-lg space-y-4">
      
      <ConfigPanel config={config} setConfig={setConfig} />

      {/* Text Input */}
      <div>
        <h2 className="text-lg font-semibold mb-2">Text Input</h2>
        <textarea
          className="w-full p-3 rounded bg-gray-900 border border-gray-700"
          rows={4}
          placeholder="Enter policy text..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <button
          onClick={handleTextSubmit}
          disabled={loading}
          className="bg-blue-600 hover:bg-blue-500 px-4 py-2 rounded-lg shadow-md transition"
        >
          {loading ? "Processing..." : "Run Check"}
        </button>
      </div>

      {/* PDF Upload */}
      <div>
        <h2 className="text-lg font-semibold mb-2">Upload PDF</h2>
        <input
          type="file"
          accept="application/pdf"
          onChange={(e) => setFile(e.target.files[0])}
        />
        <button
          onClick={handleFileSubmit}
          disabled={loading}
          className="mt-2 bg-green-600 hover:bg-green-500 px-4 py-2 rounded"
        >
          {loading ? "Uploading..." : "Analyze PDF"}
        </button>
      </div>

    </div>
  );
}