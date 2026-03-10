import { useState } from "react";
import { BrowserRouter, Routes, Route, NavLink, useLocation } from "react-router-dom";
import { Toaster } from "react-hot-toast";
import {
  LayoutDashboard, BrainCircuit, Upload,
  FlaskConical, Activity, ChevronRight, Zap,
} from "lucide-react";

import Dashboard from "./pages/Dashboard";
import Predictor from "./pages/Predictor";
import BatchUpload from "./pages/BatchUpload";
import ModelInfo from "./pages/ModelInfo";

import "./index.css";

const NAV_ITEMS = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard, end: true },
  { to: "/predict", label: "Predictor", icon: BrainCircuit },
  { to: "/batch", label: "Batch Upload", icon: Upload },
  { to: "/model", label: "Model Info", icon: FlaskConical },
];

function Sidebar() {
  const location = useLocation();
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <div className="logo-icon">
          <Zap size={18} strokeWidth={2.5} />
        </div>
        <div>
          <div className="logo-name">ChurnIQ</div>
          <div className="logo-sub">ML Platform</div>
        </div>
      </div>

      <nav className="sidebar-nav">
        {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) =>
              `nav-item ${isActive ? "nav-item--active" : ""}`
            }
          >
            <Icon size={17} />
            <span>{label}</span>
            <ChevronRight size={14} className="nav-chevron" />
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="status-dot" />
        <div>
          <div className="status-label">API Status</div>
          <div className="status-value">Live · v1.0.0</div>
        </div>
      </div>
    </aside>
  );
}

function Layout({ children }) {
  return (
    <div className="app-layout">
      <Sidebar />
      <main className="app-main">{children}</main>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Toaster
        position="top-right"
        toastOptions={{
          style: {
            background: "#1a1f2e",
            color: "#e2e8f0",
            border: "1px solid #2d3748",
            borderRadius: "10px",
            fontFamily: "'Space Grotesk', sans-serif",
          },
        }}
      />
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/predict" element={<Predictor />} />
          <Route path="/batch" element={<BatchUpload />} />
          <Route path="/model" element={<ModelInfo />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}
