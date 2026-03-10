import { useEffect, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line, Legend,
} from "recharts";
import { TrendingUp, Users, AlertTriangle, CheckCircle, Activity } from "lucide-react";
import { getModelInfo } from "../services/api";

// ─── Mock analytics data (replace with real data from your API) ───────────────
const churnByContract = [
  { name: "Month-to-month", churnRate: 42.7, total: 3875 },
  { name: "One year", churnRate: 11.3, total: 1473 },
  { name: "Two year", churnRate: 2.8, total: 1695 },
];

const churnByInternet = [
  { name: "Fiber Optic", churnRate: 41.9, color: "#ef4444" },
  { name: "DSL", churnRate: 19.0, color: "#f59e0b" },
  { name: "No Internet", churnRate: 7.4, color: "#10b981" },
];

const monthlyTrend = [
  { month: "Jan", churn: 24, retained: 76 },
  { month: "Feb", churn: 27, retained: 73 },
  { month: "Mar", churn: 22, retained: 78 },
  { month: "Apr", churn: 31, retained: 69 },
  { month: "May", churn: 28, retained: 72 },
  { month: "Jun", churn: 26, retained: 74 },
];

const riskDistribution = [
  { name: "Low Risk", value: 44, color: "#10b981" },
  { name: "Medium Risk", value: 28, color: "#f59e0b" },
  { name: "High Risk", value: 18, color: "#f97316" },
  { name: "Critical", value: 10, color: "#ef4444" },
];

const TOOLTIP_STYLE = {
  backgroundColor: "#1a2035",
  border: "1px solid #2a3347",
  borderRadius: "8px",
  color: "#e2e8f0",
  fontFamily: "'Space Grotesk', sans-serif",
  fontSize: "13px",
};

export default function Dashboard() {
  const [modelInfo, setModelInfo] = useState(null);

  useEffect(() => {
    getModelInfo().then(setModelInfo).catch(() => {});
  }, []);

  const metrics = modelInfo?.metrics || {};

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Analytics Dashboard</h1>
        <p className="page-subtitle">
          Overview of churn patterns and model performance metrics
        </p>
      </div>

      {/* ── KPI Cards ── */}
      <div className="stat-grid">
        {[
          {
            label: "Overall Churn Rate",
            value: "26.5%",
            sub: "7,043 customers analysed",
            icon: TrendingUp,
            color: "#ef4444",
          },
          {
            label: "High Risk Customers",
            value: "1,408",
            sub: "Require immediate action",
            icon: AlertTriangle,
            color: "#f97316",
          },
          {
            label: "Model AUC-ROC",
            value: metrics.roc_auc ? `${(metrics.roc_auc * 100).toFixed(1)}%` : "—",
            sub: "XGBoost tuned with Optuna",
            icon: Activity,
            color: "#3b82f6",
          },
          {
            label: "Predicted Retainable",
            value: "~$240K",
            sub: "Est. annual revenue saved",
            icon: CheckCircle,
            color: "#10b981",
          },
        ].map((s) => (
          <div className="stat-card" key={s.label} style={{ "--accent": s.color }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "12px" }}>
              <div className="stat-label">{s.label}</div>
              <s.icon size={16} color={s.color} opacity={0.7} />
            </div>
            <div className="stat-value" style={{ color: s.color }}>
              {s.value}
            </div>
            <div className="stat-change">{s.sub}</div>
          </div>
        ))}
      </div>

      {/* ── Charts Row 1 ── */}
      <div className="grid-2" style={{ marginBottom: "20px" }}>
        {/* Churn by Contract */}
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Churn Rate by Contract Type</div>
              <div className="card-subtitle">Month-to-month is the highest risk segment</div>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={churnByContract} barSize={36}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" vertical={false} />
              <XAxis dataKey="name" tick={{ fill: "#8892a4", fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: "#8892a4", fontSize: 11 }} axisLine={false} tickLine={false} unit="%" />
              <Tooltip
                contentStyle={TOOLTIP_STYLE}
                formatter={(v) => [`${v}%`, "Churn Rate"]}
              />
              <Bar dataKey="churnRate" radius={[6, 6, 0, 0]}>
                {churnByContract.map((_, i) => (
                  <Cell key={i} fill={i === 0 ? "#ef4444" : i === 1 ? "#f59e0b" : "#10b981"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Risk Distribution Pie */}
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Risk Distribution</div>
              <div className="card-subtitle">Customer segmentation by churn risk level</div>
            </div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "24px" }}>
            <ResponsiveContainer width="55%" height={220}>
              <PieChart>
                <Pie
                  data={riskDistribution}
                  cx="50%" cy="50%"
                  innerRadius={55} outerRadius={85}
                  paddingAngle={3}
                  dataKey="value"
                >
                  {riskDistribution.map((entry, i) => (
                    <Cell key={i} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v) => [`${v}%`, ""]} />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ flex: 1 }}>
              {riskDistribution.map((d) => (
                <div key={d.name} style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "10px" }}>
                  <div style={{ width: "10px", height: "10px", borderRadius: "2px", background: d.color, flexShrink: 0 }} />
                  <div style={{ flex: 1, fontSize: "12px", color: "var(--text-secondary)" }}>{d.name}</div>
                  <div style={{ fontSize: "13px", fontWeight: "600", color: d.color }}>{d.value}%</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* ── Charts Row 2 ── */}
      <div className="grid-2">
        {/* Monthly Trend */}
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Monthly Churn Trend</div>
              <div className="card-subtitle">Churn vs retention rate over 6 months</div>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={monthlyTrend}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" vertical={false} />
              <XAxis dataKey="month" tick={{ fill: "#8892a4", fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: "#8892a4", fontSize: 11 }} axisLine={false} tickLine={false} unit="%" />
              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v) => [`${v}%`]} />
              <Legend wrapperStyle={{ fontSize: "12px", color: "#8892a4" }} />
              <Line type="monotone" dataKey="churn" stroke="#ef4444" strokeWidth={2} dot={{ r: 3, fill: "#ef4444" }} />
              <Line type="monotone" dataKey="retained" stroke="#10b981" strokeWidth={2} dot={{ r: 3, fill: "#10b981" }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Model Performance */}
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Model Performance</div>
              <div className="card-subtitle">XGBoost evaluation metrics on test set</div>
            </div>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: "14px", marginTop: "8px" }}>
            {[
              { label: "AUC-ROC", value: metrics.roc_auc || 0.891, color: "#3b82f6" },
              { label: "F1 Score", value: metrics.f1 || 0.637, color: "#10b981" },
              { label: "Precision", value: metrics.precision || 0.681, color: "#f59e0b" },
              { label: "Recall", value: metrics.recall || 0.598, color: "#a855f7" },
              { label: "Accuracy", value: metrics.accuracy || 0.812, color: "#06b6d4" },
            ].map((m) => (
              <div key={m.label}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px" }}>
                  <span style={{ fontSize: "12px", color: "var(--text-secondary)" }}>{m.label}</span>
                  <span style={{ fontSize: "13px", fontWeight: "700", color: m.color, fontFamily: "var(--font-mono)" }}>
                    {(m.value * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{ width: `${m.value * 100}%`, background: m.color }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
