import React, { useState, useEffect, useRef } from 'react';
import { 
  Upload, Sparkles, CheckCircle2, AlertTriangle, XCircle, 
  Layers, Sliders, Clock, Crosshair, ZoomIn, RefreshCw, FileText
} from 'lucide-react';
import { api } from '../services/api';

const PRODUCT_TYPES = [
  { id: 'metal_sheet', label: 'Metal Sheet (Brushed/Machined)' },
  { id: 'bottle', label: 'Glass Container / Bottle' },
  { id: 'capsule', label: 'Pharmaceutical Capsule' },
  { id: 'pcb', label: 'Printed Circuit Board (PCB)' },
  { id: 'ceramic_tile', label: 'Ceramic Floor/Wall Tile' },
];

export default function InspectionStudio({ onInspectionComplete }) {
  const [productType, setProductType] = useState('metal_sheet');
  const [confThreshold, setConfThreshold] = useState(0.25);
  const [alpha, setAlpha] = useState(0.45);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [inspecting, setInspecting] = useState(false);
  const [result, setResult] = useState(null);
  const [viewMode, setViewMode] = useState('annotated'); // 'annotated', 'heatmap', 'side_by_side', 'original'
  const [samples, setSamples] = useState([]);
  const [loadingSamples, setLoadingSamples] = useState(false);
  const fileInputRef = useRef(null);

  // Load sample images from backend
  useEffect(() => {
    async function fetchSamples() {
      setLoadingSamples(true);
      try {
        const data = await api.getSamples();
        setSamples(data);
      } catch (err) {
        console.error('Failed to load sample images:', err);
      } finally {
        setLoadingSamples(false);
      }
    }
    fetchSamples();
  }, []);

  const handleFileSelect = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
    }
  };

  const handleSampleClick = async (sample) => {
    try {
      // Convert base64 sample full image to Blob/File
      const res = await fetch(sample.full_image_url);
      const blob = await res.blob();
      const file = new File([blob], sample.filename, { type: 'image/jpeg' });
      setSelectedFile(file);
      setPreviewUrl(sample.full_image_url);
      setProductType(sample.product_type || 'metal_sheet');
      setResult(null);
    } catch (err) {
      console.error('Error loading sample image:', err);
    }
  };

  const runInspection = async () => {
    if (!selectedFile) return;
    setInspecting(true);
    try {
      const data = await api.inspectImage(selectedFile, productType, confThreshold, alpha);
      setResult(data);
      if (onInspectionComplete) {
        onInspectionComplete(data);
      }
    } catch (err) {
      console.error('Inspection failed:', err);
      alert('Inspection error: ' + (err.response?.data?.detail || err.message));
    } finally {
      setInspecting(false);
    }
  };

  // Helper for active displayed image
  const getDisplayImage = () => {
    if (!result) return previewUrl;
    if (viewMode === 'original') return result.images.original;
    if (viewMode === 'heatmap') return result.images.heatmap;
    if (viewMode === 'side_by_side') return result.images.side_by_side;
    return result.images.annotated;
  };

  const verdictMeta = result?.verdict;
  const isPass = verdictMeta?.verdict === 'PASS';
  const isReview = verdictMeta?.verdict === 'REVIEW';
  const isFail = verdictMeta?.verdict === 'FAIL';

  return (
    <div className="space-y-6">
      {/* Top Banner / Breadcrumb */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 bg-industrial-900 border border-industrial-800 rounded-xl p-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
            <Crosshair className="w-5 h-5 text-cyan-400" />
            Industrial Inspection Studio
          </h1>
          <p className="text-xs text-slate-400 font-mono">
            Dual-Stage Computer Vision: YOLOv8 Defect Localization & PatchCore Deep Anomaly Heatmap
          </p>
        </div>

        {/* Inspection Parameter Controls */}
        <div className="flex flex-wrap items-center gap-3">
          <div>
            <label className="block text-[11px] font-mono text-slate-400 uppercase">Substrate Target</label>
            <select
              value={productType}
              onChange={(e) => setProductType(e.target.value)}
              className="bg-industrial-950 border border-industrial-700 text-xs text-slate-200 rounded-lg px-2.5 py-1.5 focus:outline-none focus:border-cyan-500 font-mono"
            >
              {PRODUCT_TYPES.map((pt) => (
                <option key={pt.id} value={pt.id}>{pt.label}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-[11px] font-mono text-slate-400 uppercase">
              Confidence: {(confThreshold * 100).toFixed(0)}%
            </label>
            <input
              type="range"
              min="0.10"
              max="0.80"
              step="0.05"
              value={confThreshold}
              onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
              className="w-24 accent-cyan-400 cursor-pointer"
            />
          </div>

          <div>
            <label className="block text-[11px] font-mono text-slate-400 uppercase">
              Heatmap Alpha: {(alpha * 100).toFixed(0)}%
            </label>
            <input
              type="range"
              min="0.20"
              max="0.80"
              step="0.05"
              value={alpha}
              onChange={(e) => setAlpha(parseFloat(e.target.value))}
              className="w-24 accent-cyan-400 cursor-pointer"
            />
          </div>

          <button
            onClick={runInspection}
            disabled={!selectedFile || inspecting}
            className={`flex items-center space-x-2 px-5 py-2.5 rounded-lg text-xs font-semibold uppercase tracking-wider transition-all font-mono ${
              !selectedFile || inspecting
                ? 'bg-industrial-800 text-slate-500 cursor-not-allowed border border-industrial-700'
                : 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white hover:from-cyan-400 hover:to-blue-500 shadow-md shadow-cyan-500/20 active:scale-95'
            }`}
          >
            {inspecting ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span>Analyzing...</span>
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4" />
                <span>Execute Inspection</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Main Grid: Left Image Area, Right Telemetry & Results */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Image Viewer (7 Cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="bg-industrial-900 border border-industrial-800 rounded-xl p-4 flex flex-col h-full">
            {/* View Layer Toggles */}
            <div className="flex items-center justify-between pb-3 border-b border-industrial-800 mb-3">
              <div className="flex items-center space-x-1.5 text-xs font-mono text-slate-300">
                <Layers className="w-4 h-4 text-cyan-400" />
                <span>LAYER VIEW:</span>
              </div>
              <div className="flex items-center space-x-1 bg-industrial-950 p-1 rounded-lg border border-industrial-800">
                {[
                  { id: 'annotated', label: 'Annotated BBox' },
                  { id: 'heatmap', label: 'Anomaly Heatmap' },
                  { id: 'side_by_side', label: 'Side-by-Side' },
                  { id: 'original', label: 'Raw Original' },
                ].map((mode) => (
                  <button
                    key={mode.id}
                    onClick={() => setViewMode(mode.id)}
                    disabled={!result && mode.id !== 'original'}
                    className={`px-2.5 py-1 rounded text-[11px] font-mono transition-all ${
                      viewMode === mode.id
                        ? 'bg-industrial-800 text-cyan-400 font-bold border border-cyan-500/30'
                        : 'text-slate-400 hover:text-slate-200 disabled:opacity-40 disabled:hover:text-slate-400'
                    }`}
                  >
                    {mode.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Display Canvas / Image Frame */}
            <div className="relative flex-1 min-h-[380px] bg-industrial-950 rounded-lg border border-industrial-850 flex items-center justify-center overflow-hidden">
              {getDisplayImage() ? (
                <div className="relative max-h-[520px] w-full flex items-center justify-center p-2">
                  <img
                    src={getDisplayImage()}
                    alt="Industrial Inspection View"
                    className="max-h-[500px] w-auto object-contain rounded shadow-lg border border-industrial-800"
                  />
                  {/* Floating Layer Tag */}
                  <div className="absolute top-4 left-4 bg-industrial-950/85 backdrop-blur-md border border-industrial-700/60 px-2.5 py-1 rounded text-[10px] font-mono uppercase text-cyan-400">
                    Mode: {viewMode}
                  </div>
                </div>
              ) : (
                <div
                  onClick={() => fileInputRef.current?.click()}
                  className="flex flex-col items-center justify-center p-8 text-center cursor-pointer hover:bg-industrial-900/50 transition-colors w-full h-full border-2 border-dashed border-industrial-750 rounded-lg"
                >
                  <div className="p-4 rounded-full bg-industrial-900 text-cyan-400 mb-3 border border-industrial-800">
                    <Upload className="w-8 h-8" />
                  </div>
                  <p className="text-sm font-semibold text-slate-200">
                    Upload Industrial Inspection Image
                  </p>
                  <p className="text-xs text-slate-400 mt-1 max-w-sm">
                    Drag and drop manufacturing image or choose from test gallery below (JPG, PNG, TIFF)
                  </p>
                  <span className="mt-4 px-3 py-1.5 rounded-lg bg-industrial-850 text-cyan-400 text-xs font-mono border border-cyan-500/20">
                    Browse Machine Images
                  </span>
                </div>
              )}

              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                className="hidden"
              />
            </div>

            {/* Latency & Telemetry Footer */}
            {result && (
              <div className="mt-3 pt-3 border-t border-industrial-800 flex flex-wrap items-center justify-between text-xs font-mono text-slate-400">
                <div className="flex items-center space-x-4">
                  <span className="flex items-center gap-1.5">
                    <Clock className="w-3.5 h-3.5 text-cyan-400" />
                    Total: <strong className="text-slate-200">{result.latency.total_ms} ms</strong>
                  </span>
                  <span>
                    YOLO: <strong className="text-slate-200">{result.latency.yolo_ms} ms</strong>
                  </span>
                  <span>
                    PatchCore: <strong className="text-slate-200">{result.latency.anomaly_ms} ms</strong>
                  </span>
                </div>
                <div>
                  Anomaly Score: <strong className="text-amber-400">{(result.anomaly.score * 100).toFixed(1)}%</strong>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Decision, Severity, Defect Breakdown (5 Cols) */}
        <div className="lg:col-span-5 space-y-4">
          {/* Quality Decision Verdict Card */}
          <div className="bg-industrial-900 border border-industrial-800 rounded-xl p-5">
            <h2 className="text-xs font-mono uppercase text-slate-400 tracking-wider mb-2">
              Automated Inspection Verdict
            </h2>

            {result ? (
              <div className="space-y-4">
                <div
                  className={`p-4 rounded-xl border flex items-start space-x-3.5 transition-all ${
                    isPass
                      ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                      : isReview
                      ? 'bg-amber-500/10 border-amber-500/30 text-amber-400'
                      : 'bg-red-500/10 border-red-500/30 text-red-400'
                  }`}
                >
                  {isPass && <CheckCircle2 className="w-7 h-7 flex-shrink-0 mt-0.5" />}
                  {isReview && <AlertTriangle className="w-7 h-7 flex-shrink-0 mt-0.5" />}
                  {isFail && <XCircle className="w-7 h-7 flex-shrink-0 mt-0.5" />}

                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-2xl font-black font-mono tracking-tight">
                        {verdictMeta.verdict}
                      </span>
                      <span className="text-xs uppercase font-mono px-2 py-0.5 rounded-full bg-black/30 border border-current">
                        Quality Gate
                      </span>
                    </div>
                    <p className="text-xs mt-1 text-slate-300 leading-relaxed font-sans">
                      {verdictMeta.status_text}
                    </p>
                  </div>
                </div>

                {/* KPI Summary Pill Counters */}
                <div className="grid grid-cols-4 gap-2 text-center font-mono">
                  <div className="bg-industrial-950 p-2 rounded-lg border border-industrial-800">
                    <span className="text-[10px] text-slate-400 block uppercase">Total</span>
                    <strong className="text-base text-slate-100">{result.defects.length}</strong>
                  </div>
                  <div className="bg-industrial-950 p-2 rounded-lg border border-industrial-800">
                    <span className="text-[10px] text-red-400 block uppercase">Critical</span>
                    <strong className="text-base text-red-400">{verdictMeta.critical_defects}</strong>
                  </div>
                  <div className="bg-industrial-950 p-2 rounded-lg border border-industrial-800">
                    <span className="text-[10px] text-amber-400 block uppercase">Medium</span>
                    <strong className="text-base text-amber-400">{verdictMeta.medium_defects}</strong>
                  </div>
                  <div className="bg-industrial-950 p-2 rounded-lg border border-industrial-800">
                    <span className="text-[10px] text-blue-400 block uppercase">Low</span>
                    <strong className="text-base text-blue-400">{verdictMeta.low_defects}</strong>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-8 border border-dashed border-industrial-800 rounded-lg">
                <Crosshair className="w-8 h-8 text-slate-600 mx-auto mb-2" />
                <p className="text-xs text-slate-400 font-mono">
                  Awaiting inspection execution...
                </p>
                <p className="text-[11px] text-slate-500 mt-1">
                  Upload an image or pick a test sample to evaluate defects.
                </p>
              </div>
            )}
          </div>

          {/* Defect Breakdown List Table */}
          <div className="bg-industrial-900 border border-industrial-800 rounded-xl p-5">
            <h3 className="text-xs font-mono uppercase text-slate-400 tracking-wider mb-3 flex items-center justify-between">
              <span>Detected Defect Items</span>
              <span className="text-cyan-400 text-[11px]">
                {result?.defects?.length || 0} Identified
              </span>
            </h3>

            {result?.defects && result.defects.length > 0 ? (
              <div className="space-y-2.5 max-h-[300px] overflow-y-auto pr-1">
                {result.defects.map((d, idx) => (
                  <div
                    key={d.id || idx}
                    className="p-3 rounded-lg bg-industrial-950 border border-industrial-800 hover:border-industrial-700 transition-colors"
                  >
                    <div className="flex items-center justify-between mb-1.5">
                      <div className="flex items-center space-x-2">
                        <span className="text-xs font-mono font-bold text-slate-200 uppercase">
                          {d.class_name.replace('_', ' ')}
                        </span>
                        <span className={`text-[10px] font-mono px-2 py-0.5 rounded border uppercase ${d.badge_class}`}>
                          {d.severity_level}
                        </span>
                      </div>
                      <span className="text-xs font-mono text-cyan-400 font-bold">
                        {(d.confidence * 100).toFixed(1)}% Conf
                      </span>
                    </div>

                    <div className="grid grid-cols-3 gap-2 text-[11px] font-mono text-slate-400 mt-2 bg-industrial-900/60 p-2 rounded">
                      <div>
                        <span className="text-slate-500 block text-[9px]">AREA</span>
                        <span className="text-slate-300">{d.area_pixels} px ({d.area_percentage}%)</span>
                      </div>
                      <div>
                        <span className="text-slate-500 block text-[9px]">RISK SCORE</span>
                        <span className="text-slate-300">{(d.severity_score * 100).toFixed(0)} / 100</span>
                      </div>
                      <div>
                        <span className="text-slate-500 block text-[9px]">BBOX</span>
                        <span className="text-slate-300">[{d.bbox.join(', ')}]</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : result ? (
              <div className="text-center py-6 text-xs text-slate-400 font-mono">
                No localized defect anomalies detected above confidence threshold.
              </div>
            ) : (
              <div className="text-center py-6 text-xs text-slate-500 font-mono">
                Defect breakdown will appear following analysis.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Bottom Section: Sample Inspection Gallery */}
      <div className="bg-industrial-900 border border-industrial-800 rounded-xl p-5">
        <div className="flex items-center justify-between mb-3">
          <div>
            <h3 className="text-sm font-bold text-slate-200 flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-cyan-400" />
              1-Click Sample Test Gallery
            </h3>
            <p className="text-xs text-slate-400 font-mono">
              Pre-loaded industrial substrates with certified ground-truth defect variations & nominal items
            </p>
          </div>
          <span className="text-xs text-slate-400 font-mono">
            {samples.length} Samples Ready
          </span>
        </div>

        {loadingSamples ? (
          <div className="py-8 text-center text-xs font-mono text-slate-400">
            Loading sample gallery...
          </div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-3">
            {samples.map((s, idx) => (
              <div
                key={idx}
                onClick={() => handleSampleClick(s)}
                className="group cursor-pointer bg-industrial-950 border border-industrial-800 hover:border-cyan-500/50 rounded-lg p-2 transition-all hover:scale-105"
              >
                <div className="aspect-square w-full rounded overflow-hidden bg-industrial-900 border border-industrial-850 mb-1.5">
                  <img
                    src={s.thumbnail_url}
                    alt={s.filename}
                    className="w-full h-full object-cover group-hover:opacity-90"
                  />
                </div>
                <div className="text-[10px] font-mono truncate text-slate-300 font-semibold uppercase">
                  {s.expected_class.replace('_', ' ')}
                </div>
                <div className="text-[9px] font-mono text-slate-500 truncate">
                  {s.product_type}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
