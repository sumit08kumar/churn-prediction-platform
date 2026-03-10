import { useState } from "react";
import { toast } from "react-hot-toast";
import { Brain, Loader, Info, ArrowRight, RefreshCw } from "lucide-react";
import { predictSingle, explainPrediction, getSample } from "../services/api";

// ─── Form field definitions ───────────────────────────────────────────────────
const FIELDS = [
  // Demographics
  {
    group: "Demographics",
    fields: [
      { name: "gender", label: "Gender", type: "select", options: ["Male", "Female"] },
      { name: "SeniorCitizen", label: "Senior Citizen", type: "select", options: [{ label: "No", value: 0 }, { label: "Yes", value: 1 }] },
      { name: "Partner", label: "Has Partner", type: "select", options: ["Yes", "No"] },
      { name: "Dependents", label: "Has Dependents", type: "select", options: ["Yes", "No"] },
    ],
  },
  // Account
  {
    group: "Account Information",
    fields: [
      { name: "tenure", label: "Tenure (months)", type: "number", min: 0, max: 72, step: 1 },
      { name: "Contract", label: "Contract Type", type: "select", options: ["Month-to-month", "One year", "Two year"] },
      { name: "PaperlessBilling", label: "Paperless Billing", type: "select", options: ["Yes", "No"] },
      { name: "PaymentMethod", label: "Payment Method", type: "select", options: ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"] },
      { name: "MonthlyCharges", label: "Monthly Charges ($)", type: "number", min: 0, step: 0.01 },
      { name: "TotalCharges", label: "Total Charges ($)", type: "number", min: 0, step: 0.01 },
    ],
  },
  // Phone
  {
    group: "Phone Services",
    fields: [
      { name: "PhoneService", label: "Phone Service", type: "select", options: ["Yes", "No"] },
      { name: "MultipleLines", label: "Multiple Lines", type: "select", options: ["Yes", "No", "No phone service"] },
    ],
  },
  // Internet
  {
    group: "Internet Services",
    fields: [
      { name: "InternetService", label: "Internet Service", type: "select", options: ["DSL", "Fiber optic", "No"] },
      { name: "OnlineSecurity", label: "Online Security", type: "select", options: ["Yes", "No", "No internet service"] },
      { name: "OnlineBackup", label: "Online Backup", type: "select", options: ["Yes", "No", "No internet service"] },
      { name: "DeviceProtection", label: "Device Protection", type: "select", options: ["Yes", "No", "No internet service"] },
      { name: "TechSupport", label: "Tech Support", type: "select", options: ["Yes", "No", "No internet service"] },
      { name: "StreamingTV", label: "Streaming TV", type: "select", options: ["Yes", "No", "No internet service"] },
      { name: "StreamingMovies", label: "Streaming Movies", type: "select", options: ["Yes", "No", "No internet service"] },
    ],
  },
];

const DEFAULT_VALUES = {
  gender: "Male", SeniorCitizen: 0, Partner: "Yes", Dependents: "No",
  tenure: 12, Contract: "Month-to-month", PaperlessBilling: "Yes",
  PaymentMethod: "Electronic check", MonthlyCharges: 65.0, TotalCharges: 780.0,
  PhoneService: "Yes", MultipleLines: "No", InternetService: "Fiber optic",
  OnlineSecurity: "No", OnlineBackup: "Yes", DeviceProtection: "No",
  TechSupport: "No", StreamingTV: "Yes", StreamingMovies: "Yes",
};

// ─── Risk Gauge Component ─────────────────────────────────────────────────────
function RiskGauge({ probability, risk }) {
  const pct = Math.round(probability * 100);
  const color = {
    Low: "#10b981", Medium: "#f59e0b", High: "#f97316", Critical: "#ef4444"
  }[risk] || "#3b82f6";

  const circumference = 2 * Math.PI * 54;
  const dashOffset = circumference * (1 - probability);

  return (
    <div className="gauge-container">
      <div style={{ position: "relative", width: "140px", height: "140px" }}>
        <svg width="140" height="140" style={{ transform: "rotate(-90deg)" }}>
          <circle cx="70" cy="70" r="54" fill="none" stroke="#1e2a3a" strokeWidth="12" />
          <circle
            cx="70" cy="70" r="54" fill="none"
            stroke={color} strokeWidth="12"
            strokeDasharray={circumference}
            strokeDashoffset={dashOffset}
            strokeLinecap="round"
            style={{ transition: "stroke-dashoffset 1s ease, stroke 0.4s ease", filter: `drop-shadow(0 0 8px ${color})` }}
          />
        </svg>
        <div style={{
          position: "absolute", inset: 0, display: "flex",
          flexDirection: "column", alignItems: "center", justifyContent: "center",
        }}>
          <div style={{ fontSize: "28px", fontWeight: "800", color, lineHeight: 1 }}>{pct}%</div>
          <div style={{ fontSize: "10px", color: "var(--text-muted)", letterSpacing: "1px", textTransform: "uppercase", marginTop: "4px" }}>Risk Score</div>
        </div>
      </div>
      <div className={`risk-badge risk-${risk}`} style={{ fontSize: "14px", padding: "6px 18px" }}>
        {risk} Risk
      </div>
    </div>
  );
}

// ─── SHAP Panel ───────────────────────────────────────────────────────────────
function ShapPanel({ factors }) {
  if (!factors?.length) return null;
  const maxImpact = Math.max(...factors.map((f) => f.impact));

  return (
    <div>
      <div style={{ fontSize: "12px", color: "var(--text-muted)", marginBottom: "14px", textTransform: "uppercase", letterSpacing: "1px" }}>
        Top Factors (SHAP)
      </div>
      {factors.map((f, i) => (
        <div key={i} className="shap-item">
          <div className="shap-name" title={f.feature}>{f.feature}</div>
          <div className="shap-bar-track">
            <div
              className="shap-bar-fill"
              style={{
                width: `${(f.impact / maxImpact) * 100}%`,
                background: f.direction === "increases" ? "#ef4444" : "#10b981",
              }}
            />
          </div>
          <div className="shap-val" style={{ color: f.direction === "increases" ? "#ef4444" : "#10b981" }}>
            {f.direction === "increases" ? "+" : "-"}{f.impact.toFixed(1)}
          </div>
        </div>
      ))}
      <div style={{ display: "flex", gap: "20px", marginTop: "12px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "11px", color: "var(--text-muted)" }}>
          <div style={{ width: "10px", height: "4px", background: "#ef4444", borderRadius: "2px" }} />
          Increases risk
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "11px", color: "var(--text-muted)" }}>
          <div style={{ width: "10px", height: "4px", background: "#10b981", borderRadius: "2px" }} />
          Decreases risk
        </div>
      </div>
    </div>
  );
}

// ─── Main Predictor Page ──────────────────────────────────────────────────────
export default function Predictor() {
  const [formData, setFormData] = useState(DEFAULT_VALUES);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value, type } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === "number" ? parseFloat(value) || 0 : value,
    }));
  };

  const handleSelectChange = (name, value) => {
    const parsed = value === "0" ? 0 : value === "1" ? 1 : isNaN(Number(value)) ? value : value;
    setFormData((prev) => ({ ...prev, [name]: parsed }));
  };

  const handleSubmit = async () => {
    setLoading(true);
    setResult(null);
    try {
      const data = await predictSingle(formData);
      setResult(data);
      toast.success("Prediction complete!");
    } catch (err) {
      toast.error(err?.response?.data?.detail || "Prediction failed. Is the API running?");
    } finally {
      setLoading(false);
    }
  };

  const handleLoadSample = async () => {
    try {
      const sample = await getSample();
      setFormData(sample);
      toast.success("Sample loaded!");
    } catch {
      setFormData(DEFAULT_VALUES);
      toast.success("Sample loaded (default).");
    }
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Churn Predictor</h1>
        <p className="page-subtitle">Enter customer details to get a real-time churn risk prediction with SHAP explanations</p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: result ? "1fr 340px" : "1fr", gap: "24px", alignItems: "start" }}>
        {/* ── Form ── */}
        <div>
          <div style={{ display: "flex", gap: "10px", marginBottom: "24px" }}>
            <button className="btn btn-outline" onClick={handleLoadSample}>
              <RefreshCw size={14} /> Load Sample
            </button>
          </div>

          {FIELDS.map(({ group, fields }) => (
            <div key={group} className="card" style={{ marginBottom: "16px" }}>
              <div className="card-header" style={{ marginBottom: "16px" }}>
                <div className="card-title">{group}</div>
              </div>
              <div className="form-grid">
                {fields.map((field) => (
                  <div className="form-group" key={field.name}>
                    <label className="form-label">{field.label}</label>
                    {field.type === "select" ? (
                      <select
                        className="form-select"
                        name={field.name}
                        value={formData[field.name]}
                        onChange={(e) => handleSelectChange(field.name, e.target.value)}
                      >
                        {field.options.map((opt) =>
                          typeof opt === "object" ? (
                            <option key={opt.value} value={opt.value}>{opt.label}</option>
                          ) : (
                            <option key={opt} value={opt}>{opt}</option>
                          )
                        )}
                      </select>
                    ) : (
                      <input
                        className="form-input"
                        type="number"
                        name={field.name}
                        value={formData[field.name]}
                        onChange={handleChange}
                        min={field.min}
                        max={field.max}
                        step={field.step}
                      />
                    )}
                  </div>
                ))}
              </div>
            </div>
          ))}

          <button
            className="btn btn-primary btn-lg"
            onClick={handleSubmit}
            disabled={loading}
            style={{ width: "100%", marginTop: "8px" }}
          >
            {loading ? (
              <><span className="spinner" /> Predicting...</>
            ) : (
              <><Brain size={18} /> Predict Churn Risk <ArrowRight size={16} /></>
            )}
          </button>
        </div>

        {/* ── Result Panel ── */}
        {result && (
          <div style={{ position: "sticky", top: "24px" }}>
            <div className="card" style={{ marginBottom: "16px" }}>
              <div className="card-header">
                <div className="card-title">Prediction Result</div>
              </div>

              <RiskGauge probability={result.churn_probability} risk={result.risk_level} />

              <hr className="divider" />

              <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
                {[
                  { label: "Probability", value: `${(result.churn_probability * 100).toFixed(1)}%` },
                  { label: "Prediction", value: result.churn_prediction ? "Will Churn" : "Will Stay" },
                  { label: "Confidence", value: result.confidence },
                  { label: "Risk Score", value: `${result.risk_score}/100` },
                ].map(({ label, value }) => (
                  <div key={label} style={{ display: "flex", justifyContent: "space-between", fontSize: "13px" }}>
                    <span style={{ color: "var(--text-secondary)" }}>{label}</span>
                    <span style={{ fontWeight: "600" }}>{value}</span>
                  </div>
                ))}
              </div>

              <hr className="divider" />

              <div style={{
                background: "var(--bg-elevated)",
                border: "1px solid var(--border)",
                borderRadius: "var(--radius-sm)",
                padding: "12px",
                fontSize: "12px",
                color: "var(--text-secondary)",
                lineHeight: "1.6",
              }}>
                <div style={{ display: "flex", gap: "6px", marginBottom: "4px", color: "var(--accent)" }}>
                  <Info size={13} /> Recommendation
                </div>
                {result.recommendation}
              </div>
            </div>

            {result.top_factors?.length > 0 && (
              <div className="card">
                <ShapPanel factors={result.top_factors} />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
