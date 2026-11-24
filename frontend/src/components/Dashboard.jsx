import React, { useEffect, useState } from 'react';
import { 
  BarChart3, TrendingUp, AlertOctagon, CheckCircle2, Clock, 
  RefreshCw, PieChart, Activity, ShieldAlert
} from 'lucide-react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  PointElement,
  LineElement
} from 'chart.js';
import { Bar, Doughnut, Line } from 'react-chartjs-2';
import { api } from '../services/api';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  PointElement,
  LineElement
);

export default function Dashboard({ onNavigateToStudio }) {
  const [analytics, setAnalytics] = useState(null);
  const [modelMetrics, setModelMetrics] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchDashboardData = async () => {
    setLoading(true);
    try {
      const [anData, mmData] = await Promise.all([
        api.getAnalytics(),
        api.getModelMetrics()
      ]);
      setAnalytics(anData);
      setModelMetrics(mmData);
    } catch (err) {
      console.error('Failed to load dashboard data:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
  }, []);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] text-slate-400 space-y-3 font-mono text-xs">
        <RefreshCw className="w-8 h-8 animate-spin text-cyan-400" />
        <p>Loading Quality Executive Analytics...</p>
      </div>
    );
  }

  // Bar Chart: Defect Categories
  const categoryLabels = Object.keys(analytics?.category_distribution || {});
  const categoryValues = Object.values(analytics?.category_distribution || {});
  const categoryBarData = {
    labels: categoryLabels.map(l => l.replace('_', ' ').toUpperCase()),
    datasets: [
      {
        label: 'Detected Count',
        data: categoryValues,
        backgroundColor: [
          'rgba(56, 189, 248, 0.7)',
          'rgba(244, 63, 94, 0.7)',
          'rgba(245, 158, 11, 0.7)',
          'rgba(168, 85, 247, 0.7)',
          'rgba(16, 185, 129, 0.7)',
          'rgba(234, 179, 8, 0.7)',
        ],
        borderColor: [
          '#38BDF8',
          '#F43F5E',
          '#F59E0B',
          '#A855F7',
          '#10B981',
          '#EAB308',
        ],
        borderWidth: 1,
        borderRadius: 6,
      },
    ],
  };

  // Doughnut Chart: Severity Breakdown
  const severityData = {
    labels: ['Critical Flaws', 'Medium Flaws', 'Low/Cosmetic'],
    datasets: [
      {
        data: [
          analytics?.severity_distribution?.CRITICAL || 0,
          analytics?.severity_distribution?.MEDIUM || 0,
          analytics?.severity_distribution?.LOW || 0,
        ],
        backgroundColor: [
          'rgba(239, 68, 68, 0.85)',
          'rgba(245, 158, 11, 0.85)',
          'rgba(59, 130, 246, 0.85)',
        ],
        borderColor: '#111827',
        borderWidth: 2,
      },
    ],
  };

  // Line Chart: Inspection Timeline
  const timeline = analytics?.recent_verdict_trend || [];
  const lineData = {
    labels: timeline.map((t, idx) => t.time || `#${idx + 1}`),
    datasets: [
      {
        label: 'Anomaly Score %',
        data: timeline.map(t => (t.anomaly_score * 100).toFixed(1)),
        borderColor: '#06B6D4',
        backgroundColor: 'rgba(6, 182, 212, 0.15)',
        tension: 0.35,
        fill: true,
      },
      {
        label: 'Latency (ms)',
        data: timeline.map(t => t.latency),
        borderColor: '#A855F7',
        backgroundColor: 'rgba(168, 85, 247, 0.1)',
        tension: 0.35,
        fill: false,
      }
    ],
  };

  const chartOptionsDark = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: {
          color: '#94A3B8',
          font: { family: 'JetBrains Mono', size: 10 }
        }
      }
    },
    scales: {
      x: {
        grid: { color: 'rgba(51, 65, 85, 0.3)' },
        ticks: { color: '#94A3B8', font: { family: 'JetBrains Mono', size: 9 } }
      },
      y: {
        grid: { color: 'rgba(51, 65, 85, 0.3)' },
        ticks: { color: '#94A3B8', font: { family: 'JetBrains Mono', size: 9 } }
      }
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 bg-industrial-900 border border-industrial-800 rounded-xl p-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-cyan-400" />
            Executive Quality & Defect Analytics
          </h1>
          <p className="text-xs text-slate-400 font-mono">
            Factory Yield Metrics, Defect Pareto Distribution & Edge Inferencing Telemetry
          </p>
        </div>
        <button
          onClick={fetchDashboardData}
          className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-industrial-950 border border-industrial-700 text-xs text-slate-300 hover:text-white font-mono hover:bg-industrial-850"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Refresh Data</span>
        </button>
      </div>

      {/* 4 Primary KPI Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 font-mono">
        {/* Total Inspected */}
        <div className="bg-industrial-900 border border-industrial-800 rounded-xl p-4 relative overflow-hidden">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs uppercase tracking-wider">Total Inspected</span>
            <Activity className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-3xl font-bold text-white tracking-tight">
            {analytics?.total_inspected || 0}
          </div>
          <div className="text-[11px] text-slate-400 mt-2 flex items-center gap-1 font-sans">
            <span className="text-cyan-400 font-mono font-bold">100%</span> inspected via Dual-Stage AI
          </div>
        </div>

        {/* First-Pass Yield Rate */}
        <div className="bg-industrial-900 border border-industrial-800 rounded-xl p-4 relative overflow-hidden">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs uppercase tracking-wider">First-Pass Yield</span>
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-3xl font-bold text-emerald-400 tracking-tight">
            {analytics?.yield_rate_percentage || 0}%
          </div>
          <div className="text-[11px] text-slate-400 mt-2 font-sans">
            {analytics?.pass_count || 0} units cleared with zero defects
          </div>
        </div>

        {/* Defect Rate */}
        <div className="bg-industrial-900 border border-industrial-800 rounded-xl p-4 relative overflow-hidden">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs uppercase tracking-wider">Defect Rate</span>
            <AlertOctagon className="w-4 h-4 text-rose-400" />
          </div>
          <div className="text-3xl font-bold text-rose-400 tracking-tight">
            {analytics?.defect_rate_percentage || 0}%
          </div>
          <div className="text-[11px] text-slate-400 mt-2 font-sans">
            {analytics?.defective_count || 0} quarantined ({analytics?.fail_count || 0} Failed, {analytics?.review_count || 0} Review)
          </div>
        </div>

        {/* Avg Latency */}
        <div className="bg-industrial-900 border border-industrial-800 rounded-xl p-4 relative overflow-hidden">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs uppercase tracking-wider">Avg Latency</span>
            <Clock className="w-4 h-4 text-purple-400" />
          </div>
          <div className="text-3xl font-bold text-purple-400 tracking-tight">
            {analytics?.avg_latency_ms || 0} ms
          </div>
          <div className="text-[11px] text-slate-400 mt-2 font-sans">
            Throughput: ~{(1000 / (analytics?.avg_latency_ms || 50)).toFixed(1)} FPS
          </div>
        </div>
      </div>

      {/* Interactive Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Defect Category Pareto Bar Chart (7 Cols) */}
        <div className="lg:col-span-7 bg-industrial-900 border border-industrial-800 rounded-xl p-5">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-bold text-slate-200">Defect Category Distribution</h3>
              <p className="text-xs text-slate-400 font-mono">Localized flaw categories identified by YOLOv8</p>
            </div>
            <span className="text-xs text-cyan-400 font-mono px-2 py-0.5 rounded bg-cyan-500/10 border border-cyan-500/20">
              Pareto Breakdown
            </span>
          </div>

          <div className="h-64">
            {categoryLabels.length > 0 ? (
              <Bar data={categoryBarData} options={chartOptionsDark} />
            ) : (
              <div className="h-full flex items-center justify-center text-xs font-mono text-slate-500">
                No defect categories logged yet. Inspect products in the Studio.
              </div>
            )}
          </div>
        </div>

        {/* Right: Severity Breakdown Doughnut Chart (5 Cols) */}
        <div className="lg:col-span-5 bg-industrial-900 border border-industrial-800 rounded-xl p-5">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-bold text-slate-200">Severity Risk Distribution</h3>
              <p className="text-xs text-slate-400 font-mono">Critical vs Medium vs Cosmetic/Low</p>
            </div>
            <ShieldAlert className="w-4 h-4 text-amber-400" />
          </div>

          <div className="h-64 flex items-center justify-center">
            {analytics?.total_inspected > 0 ? (
              <Doughnut
                data={severityData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                    legend: {
                      position: 'bottom',
                      labels: { color: '#94A3B8', font: { family: 'JetBrains Mono', size: 10 } }
                    }
                  }
                }}
              />
            ) : (
              <div className="text-xs font-mono text-slate-500">
                Awaiting defect data for severity breakdown.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Inspection Trend Line Chart */}
      <div className="bg-industrial-900 border border-industrial-800 rounded-xl p-5">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-sm font-bold text-slate-200">Recent Inspection Timeline & Anomaly Scores</h3>
            <p className="text-xs text-slate-400 font-mono">Sequential trend of deep anomaly scores and inference latencies</p>
          </div>
          <span className="text-xs text-slate-400 font-mono">Last {timeline.length} Units</span>
        </div>

        <div className="h-56">
          {timeline.length > 0 ? (
            <Line data={lineData} options={chartOptionsDark} />
          ) : (
            <div className="h-full flex items-center justify-center text-xs font-mono text-slate-500">
              No timeline points available. Execute inspections to plot data.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
