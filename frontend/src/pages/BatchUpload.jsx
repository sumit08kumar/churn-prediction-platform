import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { toast } from "react-hot-toast";
import { Upload, FileText, Download, CheckCircle, X, AlertTriangle } from "lucide-react";
import { predictBatchCsv } from "../services/api";

const RISK_COLORS = {
  Low: "#10b981", Medium: "#f59e0b", High: "#f97316", Critical: "#ef4444"
};

export default function BatchUpload() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [resultBlob, setResultBlob] = useState(null);
  const [preview, setPreview] = useState(null); // parsed result rows for display

  const onDrop = useCallback((accepted) => {
    if (accepted.length) {
      setFile(accepted[0]);
      setResultBlob(null);
      setPreview(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "text/csv": [".csv"] },
    maxFiles: 1,
    maxSize: 5 * 1024 * 1024,
  });

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    try {
      const blob = await predictBatchCsv(file);
      setResultBlob(blob);

      // Parse blob for preview
      const text = await blob.text();
      const rows = text.split("\n").filter(Boolean);
      const headers = rows[0].split(",");
      const data = rows.slice(1, 11).map((r) => {
        const vals = r.split(",");
        return Object.fromEntries(headers.map((h, i) => [h.trim(), vals[i]?.trim()]));
      });
      setPreview(data);
      toast.success(`Predictions complete for ${rows.length - 1} customers!`);
    } catch (err) {
      toast.error(err?.response?.data?.detail || "Batch prediction failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (!resultBlob) return;
    const url = URL.createObjectURL(resultBlob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "churn_predictions.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Batch Prediction</h1>
        <p className="page-subtitle">Upload a CSV of customers to get churn predictions for all of them at once</p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px", alignItems: "start" }}>
        {/* Upload Panel */}
        <div>
          <div className="card">
            <div className="card-header">
              <div className="card-title">Upload CSV</div>
            </div>

            {/* Dropzone */}
            <div
              {...getRootProps()}
              style={{
                border: `2px dashed ${isDragActive ? "#3b82f6" : "#2a3347"}`,
                borderRadius: "var(--radius)",
                padding: "48px 24px",
                textAlign: "center",
                cursor: "pointer",
                background: isDragActive ? "var(--accent-glow)" : "var(--bg-base)",
                transition: "all 0.2s",
                marginBottom: "20px",
              }}
            >
              <input {...getInputProps()} />
              <Upload size={32} color={isDragActive ? "#3b82f6" : "#4a5568"} style={{ margin: "0 auto 12px" }} />
              {isDragActive ? (
                <p style={{ color: "#3b82f6", fontWeight: "600" }}>Drop it here!</p>
              ) : (
                <>
                  <p style={{ color: "var(--text-secondary)", marginBottom: "4px" }}>
                    Drag & drop your CSV file here
                  </p>
                  <p style={{ color: "var(--text-muted)", fontSize: "12px" }}>
                    or click to browse · Max 5MB · .csv only
                  </p>
                </>
              )}
            </div>

            {/* Selected File */}
            {file && (
              <div style={{
                display: "flex", alignItems: "center", gap: "10px",
                background: "var(--bg-elevated)", borderRadius: "var(--radius-sm)",
                padding: "12px 16px", marginBottom: "16px",
                border: "1px solid var(--border)",
              }}>
                <FileText size={18} color="#3b82f6" />
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: "13px", fontWeight: "600" }}>{file.name}</div>
                  <div style={{ fontSize: "11px", color: "var(--text-muted)" }}>
                    {(file.size / 1024).toFixed(1)} KB
                  </div>
                </div>
                <button
                  onClick={(e) => { e.stopPropagation(); setFile(null); setResultBlob(null); setPreview(null); }}
                  style={{ background: "none", border: "none", cursor: "pointer", color: "var(--text-muted)" }}
                >
                  <X size={16} />
                </button>
              </div>
            )}

            <button
              className="btn btn-primary"
              onClick={handleUpload}
              disabled={!file || loading}
              style={{ width: "100%" }}
            >
              {loading ? (
                <><span className="spinner" /> Processing...</>
              ) : (
                <><Upload size={16} /> Run Batch Prediction</>
              )}
            </button>

            {resultBlob && (
              <button
                className="btn btn-outline"
                onClick={handleDownload}
                style={{ width: "100%", marginTop: "10px" }}
              >
                <Download size={16} /> Download Results CSV
              </button>
            )}
          </div>

          {/* Format Guide */}
          <div className="card" style={{ marginTop: "16px" }}>
            <div className="card-header">
              <div className="card-title">CSV Format Guide</div>
            </div>
            <p style={{ fontSize: "13px", color: "var(--text-secondary)", marginBottom: "12px" }}>
              Your CSV must have these column headers (same as the Telco dataset):
            </p>
            <div style={{
              background: "var(--bg-base)",
              borderRadius: "var(--radius-sm)",
              padding: "12px",
              fontFamily: "var(--font-mono)",
              fontSize: "11px",
              color: "#10b981",
              lineHeight: "1.8",
              border: "1px solid var(--border)",
              overflowX: "auto",
            }}>
              customerID, gender, SeniorCitizen, Partner, Dependents,
              tenure, PhoneService, MultipleLines, InternetService,
              OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
              StreamingTV, StreamingMovies, Contract, PaperlessBilling,
              PaymentMethod, MonthlyCharges, TotalCharges
            </div>
            <p style={{ fontSize: "12px", color: "var(--text-muted)", marginTop: "10px" }}>
              💡 Download the{" "}
              <a
                href="https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
                target="_blank"
                rel="noreferrer"
                style={{ color: "var(--accent)" }}
              >
                Telco Customer Churn dataset
              </a>{" "}
              from Kaggle to test batch predictions.
            </p>
          </div>
        </div>

        {/* Preview Results */}
        <div>
          {!preview ? (
            <div className="card">
              <div className="empty-state">
                <div className="empty-state-icon">📊</div>
                <div style={{ color: "var(--text-secondary)", marginBottom: "8px" }}>
                  No predictions yet
                </div>
                <p style={{ fontSize: "13px", color: "var(--text-muted)" }}>
                  Upload a CSV and run predictions to see results here.
                </p>
              </div>
            </div>
          ) : (
            <div className="card">
              <div className="card-header">
                <div>
                  <div className="card-title">Results Preview</div>
                  <div className="card-subtitle">Showing first 10 rows — download for full results</div>
                </div>
                <button className="btn btn-outline" onClick={handleDownload} style={{ padding: "6px 12px", fontSize: "12px" }}>
                  <Download size={13} /> Download
                </button>
              </div>

              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12px" }}>
                  <thead>
                    <tr>
                      {["Customer ID", "Churn Prob", "Risk Level", "Risk Score"].map((h) => (
                        <th key={h} style={{
                          textAlign: "left", padding: "8px 10px",
                          color: "var(--text-muted)", fontWeight: "600",
                          borderBottom: "1px solid var(--border)",
                          fontSize: "11px", letterSpacing: "0.5px",
                          textTransform: "uppercase",
                        }}>
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {preview.map((row, i) => {
                      const risk = row.risk_level || "Low";
                      const prob = parseFloat(row.churn_probability || 0);
                      return (
                        <tr key={i} style={{ borderBottom: "1px solid var(--border-light)" }}>
                          <td style={{ padding: "10px 10px", fontFamily: "var(--font-mono)", fontSize: "11px", color: "var(--text-secondary)" }}>
                            {row.customerID || `#${i + 1}`}
                          </td>
                          <td style={{ padding: "10px 10px" }}>
                            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                              <div style={{
                                width: "50px", height: "5px", borderRadius: "3px",
                                background: "var(--bg-elevated)", overflow: "hidden",
                              }}>
                                <div style={{
                                  width: `${prob * 100}%`, height: "100%",
                                  background: RISK_COLORS[risk],
                                }} />
                              </div>
                              <span style={{ fontFamily: "var(--font-mono)", color: RISK_COLORS[risk] }}>
                                {(prob * 100).toFixed(0)}%
                              </span>
                            </div>
                          </td>
                          <td style={{ padding: "10px 10px" }}>
                            <span className={`risk-badge risk-${risk}`}>{risk}</span>
                          </td>
                          <td style={{ padding: "10px 10px", fontFamily: "var(--font-mono)", color: RISK_COLORS[risk] }}>
                            {row.risk_score}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
