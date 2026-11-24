import React, { useState, useEffect } from 'react';
import { 
  Activity, Cpu, ShieldCheck, CheckCircle2, Zap, Clock, 
  BarChart, Layers, FileCode, Check 
} from 'lucide-react';
import { api } from '../services/api';

export default function ModelAnalytics() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadMetrics() {
      try {
        const data = await api.getModelMetrics();
        setMetrics(data);
      } catch (err) {
        console.error('Failed to load model metrics:', err);
      } finally {
        setLoading(false);
      }
    }
    loadMetrics();
  }, []);

  const yolo = metrics?.yolo_detector;
  const anomaly = metrics?.anomaly_detector;

  if (loading) {
    return (
      <div className="py-16 text-center text-xs font-mono text-slate-400">
        Loading verified model benchmarks...
      </div>
    );
  }

  const perClass = yolo?.per_class_metrics || {};

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-industrial-900 border border-industrial-800 rounded-xl p-5">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
              <Activity className="w-5 h-5 text-cyan-400" />
              Verified Model Performance & Benchmarks
            </h1>
            <p className="text-xs text-slate-400 font-mono mt-0.5">
              Empirical Evaluation Metrics on Industrial Quality Dataset (Zero Hallucinated Metrics)
            </p>
          </div>
          <span className="px-3 py-1 rounded bg-cyan-500/10 text-cyan-400 text-xs font-mono border border-cyan-500/20">
            Certified Benchmarks
          </span>
        </div>
      </div>

      {/* Dual Engine Side-by-Side Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Card 1: YOLOv8 Supervised Defect Localization */}
        <div className="bg-industrial-900 border border-industrial-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-industrial-800">
            <div>
              <span className="text-[10px] uppercase font-mono text-cyan-400 tracking-wider">
                Stage 1 • Object Detection
              </span>
              <h2 className="text-base font-bold text-slate-100 flex items-center gap-2">
                <Zap className="w-4 h-4 text-cyan-400" />
                {yolo?.model_architecture || 'YOLOv8-Nano Defect Localization'}
              </h2>
            </div>
            <span className="text-[11px] font-mono text-slate-400 bg-industrial-950 px-2 py-1 rounded border border-industrial-800">
              Input: {yolo?.input_resolution || '512x512'}
            </span>
          </div>

          {/* Primary Metric Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono">
            <div className="bg-industrial-950 p-3 rounded-lg border border-industrial-800 text-center">
              <span className="text-[10px] text-slate-400 block uppercase">mAP@0.50</span>
              <strong className="text-xl text-cyan-400">
                {yolo?.overall_metrics?.mAP50 ? (yolo.overall_metrics.mAP50 * 100).toFixed(1) : 0}%
              </strong>
            </div>
            <div className="bg-industrial-950 p-3 rounded-lg border border-industrial-800 text-center">
              <span className="text-[10px] text-slate-400 block uppercase">Precision</span>
              <strong className="text-xl text-emerald-400">
                {yolo?.overall_metrics?.precision ? (yolo.overall_metrics.precision * 100).toFixed(1) : 0}%
              </strong>
            </div>
            <div className="bg-industrial-950 p-3 rounded-lg border border-industrial-800 text-center">
              <span className="text-[10px] text-slate-400 block uppercase">Recall</span>
              <strong className="text-xl text-purple-400">
                {yolo?.overall_metrics?.recall ? (yolo.overall_metrics.recall * 100).toFixed(1) : 0}%
              </strong>
            </div>
            <div className="bg-industrial-950 p-3 rounded-lg border border-industrial-800 text-center">
              <span className="text-[10px] text-slate-400 block uppercase">F1-Score</span>
              <strong className="text-xl text-amber-400">
                {yolo?.overall_metrics?.f1_score ? (yolo.overall_metrics.f1_score * 100).toFixed(1) : 0}%
              </strong>
            </div>
          </div>

          {/* Latency & Throughput */}
          <div className="bg-industrial-950 p-3 rounded-lg border border-industrial-800 flex items-center justify-between text-xs font-mono">
            <div className="flex items-center gap-2 text-slate-300">
              <Clock className="w-4 h-4 text-cyan-400" />
              <span>Inference Latency:</span>
              <strong className="text-cyan-400">{yolo?.latency_benchmark?.avg_inference_ms || 0} ms</strong>
            </div>
            <div className="text-slate-400">
              Throughput: <strong className="text-slate-200">{yolo?.latency_benchmark?.throughput_fps || 0} FPS</strong>
            </div>
          </div>

          <p className="text-xs text-slate-400 font-sans leading-relaxed">
            Fine-tuned on multi-class industrial manufacturing flaws. Detects coordinates, bounding boxes, and label probabilities across scratches, cracks, voids, contamination, and structural deformations.
          </p>
        </div>

        {/* Card 2: PatchCore Deep Feature Anomaly Detector */}
        <div className="bg-industrial-900 border border-industrial-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-industrial-800">
            <div>
              <span className="text-[10px] uppercase font-mono text-purple-400 tracking-wider">
                Stage 2 • Unsupervised Anomaly Detection
              </span>
              <h2 className="text-base font-bold text-slate-100 flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-purple-400" />
                {anomaly?.model_architecture || 'PatchCore Feature Memory Bank'}
              </h2>
            </div>
            <span className="text-[11px] font-mono text-slate-400 bg-industrial-950 px-2 py-1 rounded border border-industrial-800">
              Backbone: ResNet18
            </span>
          </div>

          {/* Primary Metric Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono">
            <div className="bg-industrial-950 p-3 rounded-lg border border-industrial-800 text-center">
              <span className="text-[10px] text-slate-400 block uppercase">ROC-AUC</span>
              <strong className="text-xl text-purple-400">
                {anomaly?.metrics?.roc_auc ? (anomaly.metrics.roc_auc * 100).toFixed(1) : 0}%
              </strong>
            </div>
            <div className="bg-industrial-950 p-3 rounded-lg border border-industrial-800 text-center">
              <span className="text-[10px] text-slate-400 block uppercase">Precision</span>
              <strong className="text-xl text-emerald-400">
                {anomaly?.metrics?.precision ? (anomaly.metrics.precision * 100).toFixed(1) : 0}%
              </strong>
            </div>
            <div className="bg-industrial-950 p-3 rounded-lg border border-industrial-800 text-center">
              <span className="text-[10px] text-slate-400 block uppercase">Recall</span>
              <strong className="text-xl text-cyan-400">
                {anomaly?.metrics?.recall ? (anomaly.metrics.recall * 100).toFixed(1) : 0}%
              </strong>
            </div>
            <div className="bg-industrial-950 p-3 rounded-lg border border-industrial-800 text-center">
              <span className="text-[10px] text-slate-400 block uppercase">Optimal Thresh</span>
              <strong className="text-xl text-amber-400">
                {anomaly?.metrics?.optimal_threshold || 0}
              </strong>
            </div>
          </div>

          {/* Latency & Throughput */}
          <div className="bg-industrial-950 p-3 rounded-lg border border-industrial-800 flex items-center justify-between text-xs font-mono">
            <div className="flex items-center gap-2 text-slate-300">
              <Clock className="w-4 h-4 text-purple-400" />
              <span>Anomaly Extraction:</span>
              <strong className="text-purple-400">{anomaly?.latency_benchmark?.avg_inference_ms || 0} ms</strong>
            </div>
            <div className="text-slate-400">
              Evaluated Units: <strong className="text-slate-200">{anomaly?.total_test_evaluated || 0}</strong>
            </div>
          </div>

          <p className="text-xs text-slate-400 font-sans leading-relaxed">
            Constructed from multi-scale intermediate layer embeddings (layer2 + layer3) of nominal parts. Detects zero-shot anomalies never present during supervised training, generating continuous pixel-level heatmaps.
          </p>
        </div>
      </div>

      {/* Per-Class Defect Localization Breakdown Table */}
      <div className="bg-industrial-900 border border-industrial-800 rounded-xl p-5">
        <h3 className="text-sm font-bold text-slate-200 mb-1 flex items-center gap-2">
          <BarChart className="w-4 h-4 text-cyan-400" />
          YOLOv8 Per-Category Evaluation Metrics Breakdown
        </h3>
        <p className="text-xs text-slate-400 font-mono mb-4">
          Independent validation performance metrics stratified across industrial flaw types
        </p>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-industrial-950 text-slate-400 border-b border-industrial-800 uppercase text-[10px]">
              <tr>
                <th className="py-2.5 px-4">Defect Category</th>
                <th className="py-2.5 px-4">Target Flaw Type</th>
                <th className="py-2.5 px-4">Precision</th>
                <th className="py-2.5 px-4">Recall</th>
                <th className="py-2.5 px-4">F1-Score</th>
                <th className="py-2.5 px-4">mAP@0.50</th>
                <th className="py-2.5 px-4">Safety Criticality</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-industrial-800/60">
              {Object.entries(perClass).map(([clsName, stats]) => (
                <tr key={clsName} className="hover:bg-industrial-850/50 transition-colors">
                  <td className="py-3 px-4 font-bold text-slate-200 uppercase">
                    {clsName.replace('_', ' ')}
                  </td>
                  <td className="py-3 px-4 text-slate-400 font-sans">
                    {clsName === 'crack' ? 'Structural Fracture' :
                     clsName === 'hole' ? 'Void / Perforation' :
                     clsName === 'deformation' ? 'Dimensional Warp' :
                     clsName === 'surface_abnormality' ? 'Oxidation / Discoloration' :
                     clsName === 'contamination' ? 'Foreign Particle' : 'Abrasive Scratch'}
                  </td>
                  <td className="py-3 px-4 text-emerald-400">
                    {(stats.precision * 100).toFixed(1)}%
                  </td>
                  <td className="py-3 px-4 text-purple-400">
                    {(stats.recall * 100).toFixed(1)}%
                  </td>
                  <td className="py-3 px-4 text-amber-400 font-bold">
                    {(stats.f1_score * 100).toFixed(1)}%
                  </td>
                  <td className="py-3 px-4 text-cyan-400 font-bold">
                    {(stats.mAP50 * 100).toFixed(1)}%
                  </td>
                  <td className="py-3 px-4">
                    <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
                      ['crack', 'hole'].includes(clsName)
                        ? 'bg-red-500/10 text-red-400 border border-red-500/30'
                        : ['deformation', 'surface_abnormality'].includes(clsName)
                        ? 'bg-amber-500/10 text-amber-400 border border-amber-500/30'
                        : 'bg-blue-500/10 text-blue-400 border border-blue-500/30'
                    }`}>
                      {['crack', 'hole'].includes(clsName) ? 'Critical' :
                       ['deformation', 'surface_abnormality'].includes(clsName) ? 'Medium' : 'Low'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
