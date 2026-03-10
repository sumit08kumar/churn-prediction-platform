import { useEffect, useState } from "react";
import { getModelInfo, getHealth } from "../services/api";
import { FlaskConical, CheckCircle, AlertCircle, RefreshCw, Database, Cpu, Layers } from "lucide-react";

export default function ModelInfo() {
  const [info, setInfo] = useState(null);
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchData = async () => {
    setLoading(true);
    try {
      const [infoData, healthData] = await Promise.all([getModelInfo(), getHealth()]);
      setInfo(infoData);
      setHealth(healthData);
    } catch {
      setInfo(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchData(); }, []);

  const metrics = info?.metrics || {};

  return (
    <div>
      <div className="page-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div>
          <h1 className="page-title">Model Registry</h1>
          <p className="page-subtitle">Trained model details, performance metrics, and API status</p>
        </div>
        <button className="btn btn-outline" onClick={fetchData} disabled={loading}>
          <RefreshCw size={14} className={loading ? "spinning" : ""} /> Refresh
        </button>
      </div>

      {loading ? (
        <div className="card" style={{ textAlign: "center", padding: "60px" }}>
          <span className="spinner" style={{ width: "32px", height: "32px" }} />
          <p style={{ marginTop: "16px", color: "var(--text-muted)" }}>Connecting to API...</p>
        </div>
      ) : !info ? (
        <div className="card">
          <div className="empty-state">
            <AlertCircle size={40} color="#ef4444" style={{ margin: "0 auto 12px" }} />
            <div style={{ color: "var(--text-secondary)", marginBottom: "8px" }}>API Unavailable</div>
            <p style={{ fontSize: "13px", color: "var(--text-muted)" }}>
              Make sure the FastAPI backend is running on port 8000.
            </p>
          </div>
        </div>
      ) : (
        <>
          {/* API Health */}
          <div className="stat-grid" style={{ marginBottom: "24px" }}>
            {[
              {
                icon: health?.status === "healthy" ? CheckCircle : AlertCircle,
                color: health?.status === "healthy" ? "#10b981" : "#ef4444",
                label: "API Status",
                value: health?.status === "healthy" ? "Healthy" : "Degraded",
                sub: `v${health?.version || "—"}`,
              },
              {
                icon: Cpu,
                color: "#3b82f6",
                label: "Model",
                value: "XGBoost",
                sub: "Tuned with Optuna",
              },
              {
                icon: Database,
                color: "#a855f7",
                label: "Dataset",
                value: "Telco Churn",
                sub: "7,043 rows · 21 features",
              },
              {
                icon: Layers,
                color: "#f59e0b",
                label: "Registry Status",
                value: info?.status === "production" ? "Production" : "Staging",
                sub: `Version ${info?.version || "1.0.0"}`,
              },
            ].map((s) => (
              <div className="stat-card" key={s.label} style={{ "--accent": s.color }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "12px" }}>
                  <div className="stat-label">{s.label}</div>
                  <s.icon size={16} color={s.color} opacity={0.8} />
                </div>
                <div className="stat-value" style={{ fontSize: "20px", color: s.color }}>{s.value}</div>
                <div className="stat-change">{s.sub}</div>
              </div>
            ))}
          </div>

          <div className="grid-2">
            {/* Model Details */}
            <div className="card">
              <div className="card-header">
                <div className="card-title">Model Details</div>
                <span className="tag">{info.status}</span>
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: "0" }}>
                {[
                  { label: "Model Name", value: info.model_name },
                  { label: "Model Type", value: info.model_type },
                  { label: "Version", value: info.version },
                  { label: "Dataset", value: info.dataset },
                  { label: "Training Date", value: info.training_date },
                  { label: "Status", value: info.status },
                ].map(({ label, value }) => (
                  <div key={label} style={{
                    display: "flex", justifyContent: "space-between",
                    padding: "12px 0", borderBottom: "1px solid var(--border-light)",
                    fontSize: "13px",
                  }}>
                    <span style={{ color: "var(--text-secondary)" }}>{label}</span>
                    <span style={{ fontWeight: "500", fontFamily: label === "Version" ? "var(--font-mono)" : "inherit" }}>
                      {value || "—"}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Performance Metrics */}
            <div className="card">
              <div className="card-header">
                <div className="card-title">Evaluation Metrics</div>
                <div className="card-subtitle">Test set performance</div>
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
                {[
                  { key: "roc_auc", label: "AUC-ROC", color: "#3b82f6", desc: "Area under the ROC curve" },
                  { key: "f1", label: "F1 Score", color: "#10b981", desc: "Harmonic mean of precision & recall" },
                  { key: "precision", label: "Precision", color: "#f59e0b", desc: "True positives / predicted positives" },
                  { key: "recall", label: "Recall", color: "#a855f7", desc: "True positives / actual positives" },
                  { key: "accuracy", label: "Accuracy", color: "#06b6d4", desc: "Overall correct predictions" },
                ].map(({ key, label, color, desc }) => {
                  const val = metrics[key] ?? 0;
                  return (
                    <div key={key}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "6px" }}>
                        <div>
                          <span style={{ fontSize: "13px", fontWeight: "600" }}>{label}</span>
                          <span style={{ fontSize: "11px", color: "var(--text-muted)", marginLeft: "8px" }}>{desc}</span>
                        </div>
                        <span style={{
                          fontFamily: "var(--font-mono)", fontSize: "14px",
                          fontWeight: "700", color,
                        }}>
                          {val ? `${(val * 100).toFixed(2)}%` : "—"}
                        </span>
                      </div>
                      <div className="progress-bar" style={{ height: "8px" }}>
                        <div
                          className="progress-fill"
                          style={{ width: val ? `${val * 100}%` : "0%", background: color }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Pipeline Architecture */}
          <div className="card" style={{ marginTop: "20px" }}>
            <div className="card-header">
              <div className="card-title">Pipeline Architecture</div>
              <div className="card-subtitle">End-to-end ML pipeline stages</div>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: "0", overflowX: "auto", paddingBottom: "4px" }}>
              {[
                { stage: "Raw CSV", desc: "Telco dataset", color: "#4a5568" },
                { stage: "Feature Eng.", desc: "5 new features", color: "#3b82f6" },
                { stage: "SMOTE", desc: "Balance classes", color: "#a855f7" },
                { stage: "Preprocessor", desc: "Scale + Encode", color: "#f59e0b" },
                { stage: "Optuna Tuning", desc: "50 trials", color: "#f97316" },
                { stage: "XGBoost", desc: "Best model", color: "#10b981" },
                { stage: "MLflow", desc: "Registry", color: "#06b6d4" },
                { stage: "FastAPI", desc: "REST endpoint", color: "#ec4899" },
              ].map((s, i, arr) => (
                <>
                  <div key={s.stage} style={{
                    background: "var(--bg-elevated)", border: `1px solid ${s.color}33`,
                    borderRadius: "var(--radius-sm)", padding: "10px 14px",
                    textAlign: "center", minWidth: "90px", flexShrink: 0,
                  }}>
                    <div style={{ fontSize: "12px", fontWeight: "700", color: s.color, marginBottom: "2px" }}>{s.stage}</div>
                    <div style={{ fontSize: "10px", color: "var(--text-muted)" }}>{s.desc}</div>
                  </div>
                  {i < arr.length - 1 && (
                    <div key={`arrow-${i}`} style={{ color: "var(--text-muted)", fontSize: "16px", flexShrink: 0, padding: "0 4px" }}>→</div>
                  )}
                </>
              ))}
            </div>
          </div>
        </>
      )}

      <style>{`.spinning { animation: spin 1s linear infinite; } @keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
