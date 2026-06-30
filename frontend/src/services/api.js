import axios from "axios";

// 🔹 Backend URL (FastAPI)
// const API_BASE_URL = "https://miniature-enigma-gppx95gxw4xc99vw-8000.app.github.dev/";
// const API_BASE_URL = "https://vigilant-space-capybara-7vxj9xvj5xj93r99g-8000.app.github.dev";
const API_BASE_URL = "";

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
});

// 🔹 Health check
export const checkHealth = async () => {
  const res = await api.get("/health");
  return res.data;
};

// 🔹 Compliance check (text input)
export const complianceCheck = async (payload) => {
  const res = await api.post("/compliance-check", payload);
  return res.data;
};

// 🔹 Upload PDF (chunk analysis)
export const uploadPdfChunks = async (file, config) => {
  const formData = new FormData();

  formData.append("file", file);

  // Config params
  formData.append("chunk_size", config.chunk_size || 500);
  formData.append("chunk_overlap", config.chunk_overlap || 100);
  formData.append("lambda_thresh", config.lambda_thresh || 0.75);
  formData.append("hop_k", config.hop_k || 1);
  formData.append("max_triples", config.max_triples || 60);
  formData.append("prefer_local", config.prefer_local || false);

  const res = await api.post("/upload-pdf-chunks", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });

  return res.data;
};

export default api;