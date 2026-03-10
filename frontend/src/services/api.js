import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: BASE_URL,
  headers: { "Content-Type": "application/json" },
});

export const getHealth = () => api.get("/health").then((r) => r.data);
export const getModelInfo = () => api.get("/model/info").then((r) => r.data);
export const getSample = () => api.get("/sample").then((r) => r.data);

export const predictSingle = (customer) =>
  api.post("/predict", customer).then((r) => r.data);

export const predictBatchJson = (customers) =>
  api.post("/predict/batch/json", customers).then((r) => r.data);

export const explainPrediction = (customer) =>
  api.post("/explain", customer).then((r) => r.data);

export const predictBatchCsv = async (file) => {
  const formData = new FormData();
  formData.append("file", file);
  const res = await api.post("/predict/batch/csv", formData, {
    headers: { "Content-Type": "multipart/form-data" },
    responseType: "blob",
  });
  return res.data;
};

export default api;
