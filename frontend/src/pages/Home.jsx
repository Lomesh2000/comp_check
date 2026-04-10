import ChunkList from "../components/ChunkList";
import { useState } from "react";
import UploadPanel from "../components/UploadPanel";

export default function Home() {
  const [result, setResult] = useState(null);
  const [selectedChunk, setSelectedChunk] = useState(null);

  let failCount = 0;
  let passCount = 0;
  let total = 0;
  let riskScore = 0;

  if (result && result.type === "pdf") {
    total = result.data.chunks.length;
    failCount = result.data.chunks.filter(c => c.verdict === "fail").length;
    passCount = result.data.chunks.filter(c => c.verdict === "pass").length;
    riskScore = total > 0 ? Math.round((failCount / total) * 100) : 0;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-black text-white">

      <div className="max-w-7xl mx-auto p-6 space-y-6">

        {/* 🔥 HEADER */}
        <div className="flex justify-between items-center">
          <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent animate-pulse">
            Policylens
          </h1>
          <span className="text-gray-400 text-sm">
            AI Compliance Engine
          </span>
        </div>

        {/* 🔥 STATS */}
        {result && result.type === "pdf" && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">

            {/* FAIL */}
            <div className="backdrop-blur-xl bg-red-500/10 border border-red-500/30 p-5 rounded-xl shadow-lg hover:scale-[1.03] transition">
              <p className="text-sm text-red-300">Failed</p>
              <p className="text-3xl font-bold">{failCount}</p>
            </div>

            {/* PASS */}
            <div className="backdrop-blur-xl bg-green-500/10 border border-green-500/30 p-5 rounded-xl shadow-lg hover:scale-[1.03] transition">
              <p className="text-sm text-green-300">Passed</p>
              <p className="text-3xl font-bold">{passCount}</p>
            </div>

            {/* RISK */}
            <div className="backdrop-blur-xl bg-white/5 border border-white/10 p-5 rounded-xl shadow-lg hover:scale-[1.03] transition">
              <p className="text-sm text-gray-300">Risk Score</p>
              <p className={`text-3xl font-bold ${
                riskScore > 50
                  ? "text-red-400"
                  : riskScore > 20
                  ? "text-yellow-400"
                  : "text-green-400"
              }`}>
                {riskScore}%
              </p>
            </div>

          </div>
        )}

        {/* 🔥 MAIN GRID */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">

          {/* LEFT */}
          <div className="lg:col-span-1">
            <div className="sticky top-6 backdrop-blur-xl bg-white/5 border border-white/10 rounded-xl shadow-xl">
              <UploadPanel onResult={setResult} />
            </div>
          </div>

          {/* RIGHT */}
          <div className="lg:col-span-3 space-y-6">

            {/* SUMMARY */}
            {result && (
              <div className="backdrop-blur-xl bg-white/5 border border-white/10 p-6 rounded-xl shadow-xl">

                <h2 className="text-xs text-gray-400 uppercase mb-2 tracking-wide">
                  Summary
                </h2>

                <div className="flex justify-between items-center">
                  <div className="text-3xl font-bold">
                    <span className={
                      result.data.summary_verdict === "fail"
                        ? "text-red-400"
                        : result.data.summary_verdict === "pass"
                        ? "text-green-400"
                        : "text-yellow-400"
                    }>
                      {result.data.summary_verdict}
                    </span>
                  </div>

                  <div className="text-right text-sm text-gray-400">
                    <div>{result.data.total_chunks}</div>
                    <div>chunks analyzed</div>
                  </div>
                </div>

              </div>
            )}

            {/* 🔥 CONTENT */}
            {result && result.type === "pdf" && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

                {/* LIST */}
                <div className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-xl shadow-xl p-4">
                  <ChunkList
                    chunks={result.data.chunks}
                    onSelect={setSelectedChunk}
                    selectedChunk={selectedChunk}
                  />
                </div>

                {/* DETAILS */}
                <div className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-xl shadow-xl p-5">

                  {selectedChunk ? (
                    <div className="space-y-4">

                      <div className="flex justify-between items-center">
                        <h3 className="text-lg font-bold">
                          Chunk {selectedChunk.chunk_id}
                        </h3>

                        <span className={`px-2 py-1 rounded text-xs ${
                          selectedChunk.verdict === "fail"
                            ? "bg-red-500/20 text-red-400"
                            : "bg-green-500/20 text-green-400"
                        }`}>
                          {selectedChunk.verdict}
                        </span>
                      </div>

                      <div>
                        <p className="text-xs text-gray-400 mb-1">Text</p>
                        <div className="bg-black/60 p-3 rounded text-sm">
                          {selectedChunk.text_preview}
                        </div>
                      </div>

                      <div>
                        <p className="text-xs text-gray-400 mb-1">LLM Reply</p>
                        <div className="bg-black/60 p-3 rounded text-sm">
                          {selectedChunk.llm_reply || "No reply"}
                        </div>
                      </div>

                      {/* GRAPH */}
                      <div className="grid grid-cols-3 gap-3 text-xs">

                        <div>
                          <p className="text-blue-400 mb-1">Hits</p>
                          <div className="bg-black/60 p-2 rounded max-h-28 overflow-auto">
                            {selectedChunk.hits?.map((h, i) => (
                              <div key={i}>{h}</div>
                            ))}
                          </div>
                        </div>

                        <div>
                          <p className="text-green-400 mb-1">P</p>
                          <div className="bg-black/60 p-2 rounded max-h-28 overflow-auto">
                            {selectedChunk.P?.map((p, i) => (
                              <div key={i}>{p}</div>
                            ))}
                          </div>
                        </div>

                        <div>
                          <p className="text-red-400 mb-1">N</p>
                          <div className="bg-black/60 p-2 rounded max-h-28 overflow-auto">
                            {selectedChunk.N?.map((n, i) => (
                              <div key={i}>{n}</div>
                            ))}
                          </div>
                        </div>

                      </div>

                    </div>
                  ) : (
                    <div className="flex items-center justify-center h-full text-gray-500">
                      <div className="text-center">
                        <div className="text-2xl mb-2">✨</div>
                        <p>Select a chunk to explore insights</p>
                      </div>
                    </div>
                  )}

                </div>

              </div>
            )}

          </div>
        </div>

      </div>
    </div>
  );
}