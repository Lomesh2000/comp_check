export default function ChunkList({ chunks, onSelect, selectedChunk }) {
  return (
    <div className="bg-gray-900 p-5 rounded-xl border border-gray-800 shadow-lg">

      {/* Header */}
      <div className="flex justify-between items-center mb-3">
        <h2 className="text-lg font-semibold">Chunks</h2>
        <span className="text-xs text-gray-400">
          {chunks.length} items
        </span>
      </div>

      {/* Scrollable List */}
      <div className="max-h-[500px] overflow-y-auto space-y-2 pr-1">

        {chunks.map((chunk) => {
          const isSelected = selectedChunk?.chunk_id === chunk.chunk_id;

          return (
            <div
              key={chunk.chunk_id}
              onClick={() => onSelect(chunk)}
              className={`p-4 rounded-lg cursor-pointer border transition-all duration-200 ease-in-out
                ${isSelected ? "ring-2 ring-blue-500" : ""}
                ${
                  chunk.verdict === "fail"
                    ? "bg-red-900/30 border-red-500 hover:bg-red-800/50"
                    : chunk.verdict === "pass"
                    ? "bg-green-900/30 border-green-500 hover:bg-green-800/50"
                    : "bg-gray-800 border-gray-700 hover:bg-gray-700"
                }
              `}
            >

              {/* Top Row */}
              <div className="flex justify-between items-center">

                <span className="text-sm font-medium">
                  Chunk {chunk.chunk_id}
                </span>

                {/* Verdict Badge */}
                <span className={`px-2 py-1 rounded text-xs font-semibold ${
                  chunk.verdict === "fail"
                    ? "bg-red-500/20 text-red-400"
                    : chunk.verdict === "pass"
                    ? "bg-green-500/20 text-green-400"
                    : "bg-gray-500/20 text-gray-300"
                }`}>
                  {chunk.verdict}
                </span>

              </div>

              {/* Preview */}
              <p className="text-xs mt-2 text-gray-400 line-clamp-2">
                {chunk.text_preview}
              </p>

            </div>
          );
        })}

      </div>
    </div>
  );
}