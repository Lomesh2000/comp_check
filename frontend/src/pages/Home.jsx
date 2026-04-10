import ChunkList from "../components/ChunkList";
import { useState } from "react";
import UploadPanel from "../components/UploadPanel";

export default function Home() {
  const [result, setResult] = useState(null);
  const [selectedChunk, setSelectedChunk] = useState(null);
  let failedChunks = [];
    let passedChunks = [];

    if (result && result.type === "pdf") {
    failedChunks = result.data.chunks.filter(c => c.verdict === "fail");
    passedChunks = result.data.chunks.filter(c => c.verdict === "pass");
    }

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6 space-y-6">

      {/* Title */}
      <h1 className="text-3xl font-bold text-blue-400">
        LexGuard Compliance Dashboard
      </h1>

      {/* Upload Panel */}
      <UploadPanel onResult={setResult} />

      {/* Result Section */}
      {result && (
        <div className="bg-gray-800 p-6 rounded-xl shadow-lg border border-gray-700">
          
          <h2 className="text-xl font-semibold mb-3">
            Result
          </h2>

          {/* TEXT RESULT */}
          {result.type === "text" && (
            <div>
              <p>
                <strong>Verdict:</strong>{" "}
                <span className="text-yellow-400">
                  {result.data.verdict}
                </span>
              </p>

              <p className="mt-2">
                <strong>LLM Reply:</strong>
              </p>
              <div className="bg-gray-900 p-3 rounded mt-1">
                {result.data.llm_reply || "No response"}
              </div>
            </div>
          )}

          {/* PDF RESULT */}
          {result.type === "pdf" && (
            <div className="space-y-4">

                <p>
                <strong>Summary Verdict:</strong>{" "}
                <span className="text-yellow-400">
                    {result.data.summary_verdict}
                </span>
                </p>

                <p>Total Chunks: {result.data.total_chunks}</p>

                {/* 🔥 Chunk List */}
                <ChunkList
                chunks={result.data.chunks}
                onSelect={setSelectedChunk}
                />

                {/* 🔥 Selected Chunk Details */}
                {selectedChunk && (
                    <div className="bg-gray-900 p-5 rounded-xl border border-gray-700 shadow-md space-y-4">
                        
                        <h3 className="text-lg font-bold">
                        Chunk {selectedChunk.chunk_id}
                        </h3>

                        <p>
                        <strong>Verdict:</strong>{" "}
                        <span className={
                            selectedChunk.verdict === "fail"
                            ? "text-red-400"
                            : "text-green-400"
                        }>
                            {selectedChunk.verdict}
                        </span>
                        </p>

                        {/* Text */}
                        <div>
                        <p className="font-semibold">Text</p>
                        <div className="bg-black p-2 rounded text-sm">
                            {selectedChunk.text_preview}
                        </div>
                        </div>

                        {/* Evidence */}
                        <div>
                        <p className="font-semibold">Evidence</p>
                        <pre className="bg-black p-2 rounded text-xs overflow-auto">
                            {JSON.stringify(selectedChunk.evidence, null, 2)}
                        </pre>
                        </div>

                        {/* LLM Reply */}
                        <div>
                        <p className="font-semibold">LLM Reply</p>
                        <div className="bg-black p-2 rounded">
                            {selectedChunk.llm_reply || "No reply"}
                        </div>
                        </div>

                        {/* 🔥 Graph Insights */}
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">

                        {/* Hits */}
                        <div>
                            <p className="font-semibold text-blue-400">Hits</p>
                            <div className="bg-black p-2 rounded text-xs max-h-40 overflow-auto">
                            {selectedChunk.hits?.map((h, i) => (
                                <div key={i}>{h}</div>
                            ))}
                            </div>
                        </div>

                        {/* Positive */}
                        <div>
                            <p className="font-semibold text-green-400">P (Positive)</p>
                            <div className="bg-black p-2 rounded text-xs max-h-40 overflow-auto">
                            {selectedChunk.P?.map((p, i) => (
                                <div key={i}>{p}</div>
                            ))}
                            </div>
                        </div>

                        {/* Negative */}
                        <div>
                            <p className="font-semibold text-red-400">N (Negative)</p>
                            <div className="bg-black p-2 rounded text-xs max-h-40 overflow-auto">
                            {selectedChunk.N?.map((n, i) => (
                                <div key={i}>{n}</div>
                            ))}
                            </div>
                        </div>

                        </div>

                    </div>
                    )}

            </div>
            )}

        </div>
      )}

    </div>
  );
}