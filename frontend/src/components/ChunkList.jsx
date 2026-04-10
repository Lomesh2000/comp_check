export default function ChunkList({ chunks, onSelect }) {
  return (
    <div className="bg-gray-800 p-4 rounded-lg">
      <h2 className="text-xl font-semibold mb-3">Chunks</h2>

      <div className="space-y-2 max-h-80 overflow-auto">
        {chunks.map((chunk) => (
          <div
            key={chunk.chunk_id}
            onClick={() => onSelect(chunk)}
            className={`p-4 rounded-lg cursor-pointer border transition-all duration-200 hover:scale-[1.02] ${
                chunk.verdict === "fail"
                ? "bg-red-900/40 border-red-500 hover:bg-red-800/60"
                : chunk.verdict === "pass"
                ? "bg-green-900/40 border-green-500 hover:bg-green-800/60"
                : "bg-gray-700 border-gray-600"
            }`}
            >
            <div className="flex justify-between">
              <span>Chunk {chunk.chunk_id}</span>
              <span className="font-bold">{chunk.verdict}</span>
            </div>

            <p className="text-sm mt-1 text-gray-300">
              {chunk.text_preview}...
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}