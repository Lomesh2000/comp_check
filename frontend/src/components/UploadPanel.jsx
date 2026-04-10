import { useState } from "react";
import { complianceCheck, uploadPdfChunks } from "../services/api";
import ConfigPanel from "./ConfigPanel";

export default function UploadPanel({ onResult }) {
  const [text, setText] = useState("");
  const [file, setFile] = useState(null);

  const [loadingText, setLoadingText] = useState(false);
  const [loadingPdf, setLoadingPdf] = useState(false);

  const [progress, setProgress] = useState(0);

  const [config, setConfig] = useState({
    chunk_size: 500,
    chunk_overlap: 100,
    lambda_thresh: 0.75,
    hop_k: 1,
    max_triples: 60,
    prefer_local: false,
  });

  // 🔹 TEXT HANDLER
  const handleTextSubmit = async () => {
    if (!text.trim() || loadingText) return;

    setLoadingText(true);

    try {
      const res = await complianceCheck({
        text,
        ...config,
      });

      onResult({ type: "text", data: res });

    } catch (err) {
      console.error(err);
      alert(err?.response?.data || err.message);
    } finally {
      setLoadingText(false);
    }
  };

  // 🔹 PDF HANDLER (FINAL FIXED)
  const handleFileSubmit = async () => {
    if (!file || loadingPdf) return;

    setLoadingPdf(true);
    setProgress(0);

    let progressValue = 0;

    const interval = setInterval(() => {
      progressValue += 5;
      if (progressValue <= 95) {
        setProgress(progressValue);
      }
    }, 100);

    try {
      // 🔥 run API in parallel
      const responsePromise = uploadPdfChunks(file, config);

      // 🔥 force minimum 2 sec progress
      await new Promise((resolve) => setTimeout(resolve, 2000));

      const response = await responsePromise;

      console.log("API RESPONSE:", response);

      // handle both cases safely
      const data = response?.data || response;

      clearInterval(interval);
      setProgress(100);

      // small delay for smooth UX
      setTimeout(() => {
        console.log("Sending to UI:", data);
        onResult({
        type: "pdf",
        data: data,
      });

        setLoadingPdf(false);
        setProgress(0);

      }, 500);

    } catch (err) {
      console.error("PDF ERROR:", err);

      clearInterval(interval);
      setLoadingPdf(false);
      setProgress(0);
    }
  };

  return (
    <div className="bg-gray-900 p-5 rounded-xl border border-gray-800 shadow-lg space-y-6">

      <ConfigPanel config={config} setConfig={setConfig} />

      {/* 🔹 TEXT INPUT */}
      <div>
        <h2 className="text-lg font-semibold mb-2">Text Input</h2>

        <textarea
          className="w-full p-3 rounded bg-gray-800 border border-gray-700"
          rows={4}
          placeholder="Enter policy text..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />

        <button
          onClick={handleTextSubmit}
          disabled={loadingText || !text.trim()}
          className={`mt-2 px-4 py-2 rounded-lg transition ${
            loadingText
              ? "bg-gray-600 cursor-not-allowed"
              : "bg-blue-600 hover:bg-blue-500"
          }`}
        >
          {loadingText ? "Processing..." : "Run Check"}
        </button>

        {loadingText && (
          <div className="text-blue-400 text-sm mt-2">
            ⏳ Processing text...
          </div>
        )}
      </div>

      {/* 🔹 PDF UPLOAD */}
      <div>
        <h2 className="text-lg font-semibold mb-2">Upload PDF</h2>

        <input
          type="file"
          accept="application/pdf"
          onChange={(e) => setFile(e.target.files[0])}
        />

        <button
          onClick={handleFileSubmit}
          disabled={loadingPdf || !file}
          className={`mt-2 px-4 py-2 rounded-lg transition ${
            loadingPdf
              ? "bg-gray-600 cursor-not-allowed"
              : "bg-green-600 hover:bg-green-500"
          }`}
        >
          {loadingPdf ? "Processing..." : "Analyze PDF"}
        </button>

        {/* 🔥 PROGRESS BAR */}
        {(loadingPdf || progress > 0) && (
          <div className="mt-3">
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div
                className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>

            <div className="text-sm text-blue-400 mt-1">
              {progress < 30 && "Extracting text..."}
              {progress >= 30 && progress < 60 && "Embedding chunks..."}
              {progress >= 60 && progress < 90 && "Running retrieval..."}
              {progress >= 90 && "Finalizing results..."} ({progress}%)
            </div>
          </div>
        )}
      </div>

    </div>
  );
}