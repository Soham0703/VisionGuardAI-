import React, { useState, useEffect, useRef } from 'react';
import { 
  Video, Play, Pause, Camera, Radio, AlertTriangle, 
  CheckCircle2, RefreshCw, XCircle, Sliders 
} from 'lucide-react';
import { api } from '../services/api';

export default function LiveCameraModal() {
  const [streamSource, setStreamSource] = useState('simulated'); // 'webcam' or 'simulated'
  const [isRunning, setIsRunning] = useState(false);
  const [currentFrameUrl, setCurrentFrameUrl] = useState(null);
  const [fps, setFps] = useState(0);
  const [latency, setLatency] = useState(0);
  const [verdict, setVerdict] = useState(null);
  const [defects, setDefects] = useState([]);
  const [renderMode, setRenderMode] = useState('annotated');
  const [confThreshold, setConfThreshold] = useState(0.25);
  const [enableAnomaly, setEnableAnomaly] = useState(true);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const intervalRef = useRef(null);
  const samplesRef = useRef([]);

  // Load samples for simulation loop
  useEffect(() => {
    async function loadSamples() {
      try {
        const samples = await api.getSamples();
        samplesRef.current = samples;
      } catch (err) {
        console.error('Error loading simulation samples:', err);
      }
    }
    loadSamples();
  }, []);

  // WebCam Stream Setup
  useEffect(() => {
    let stream = null;
    if (streamSource === 'webcam' && isRunning) {
      navigator.mediaDevices?.getUserMedia({ video: { width: 640, height: 480 } })
        .then((s) => {
          stream = s;
          if (videoRef.current) {
            videoRef.current.srcObject = s;
            videoRef.current.play();
          }
        })
        .catch((err) => {
          console.warn('Webcam access not granted or unavailable:', err);
          alert('Webcam hardware unavailable or permission denied. Switching to conveyor simulator.');
          setStreamSource('simulated');
        });
    }

    return () => {
      if (stream) {
        stream.getTracks().forEach(t => t.stop());
      }
    };
  }, [streamSource, isRunning]);

  // Frame Capture & Processing Loop
  useEffect(() => {
    if (!isRunning) {
      if (intervalRef.current) clearInterval(intervalRef.current);
      return;
    }

    let sampleIndex = 0;

    const captureAndProcess = async () => {
      let b64Image = null;

      if (streamSource === 'webcam' && videoRef.current && canvasRef.current) {
        const canvas = canvasRef.current;
        const video = videoRef.current;
        if (video.readyState === 4) {
          canvas.width = video.videoWidth || 640;
          canvas.height = video.videoHeight || 480;
          const ctx = canvas.getContext('2d');
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          b64Image = canvas.toDataURL('image/jpeg', 0.8);
        }
      } else if (samplesRef.current.length > 0) {
        // High-Speed Conveyor Simulation: cycle through samples
        const sample = samplesRef.current[sampleIndex % samplesRef.current.length];
        sampleIndex++;
        b64Image = sample.full_image_url;
      }

      if (!b64Image) return;

      try {
        const data = await api.inspectStreamFrame(
          b64Image,
          confThreshold,
          enableAnomaly,
          renderMode
        );
        setCurrentFrameUrl(data.image_base64);
        setFps(data.fps || 0);
        setLatency(data.latency_ms || 0);
        setVerdict(data.verdict);
        setDefects(data.defects || []);
      } catch (err) {
        console.error('Frame processing error:', err);
      }
    };

    // 100ms interval (~10 FPS frame stream)
    intervalRef.current = setInterval(captureAndProcess, 140);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isRunning, streamSource, confThreshold, enableAnomaly, renderMode]);

  return (
    <div className="space-y-6">
      {/* Stream Controls Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 bg-industrial-900 border border-industrial-800 rounded-xl p-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
            <Radio className="w-5 h-5 text-rose-500 animate-pulse" />
            Live Conveyor & Machine Camera Stream
          </h1>
          <p className="text-xs text-slate-400 font-mono">
            High-Speed Edge Quality Assurance with Optical Surface Defect Alerting
          </p>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Source Toggle */}
          <div className="flex items-center space-x-1 bg-industrial-950 p-1 rounded-lg border border-industrial-800 text-xs font-mono">
            <button
              onClick={() => { setIsRunning(false); setStreamSource('simulated'); }}
              className={`px-3 py-1 rounded transition-all ${
                streamSource === 'simulated'
                  ? 'bg-industrial-800 text-cyan-400 font-bold border border-cyan-500/30'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Factory Conveyor Sim
            </button>
            <button
              onClick={() => { setIsRunning(false); setStreamSource('webcam'); }}
              className={`px-3 py-1 rounded transition-all ${
                streamSource === 'webcam'
                  ? 'bg-industrial-800 text-cyan-400 font-bold border border-cyan-500/30'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Hardware Webcam
            </button>
          </div>

          {/* Render Mode */}
          <select
            value={renderMode}
            onChange={(e) => setRenderMode(e.target.value)}
            className="bg-industrial-950 border border-industrial-700 text-xs text-slate-200 rounded-lg px-2.5 py-1.5 focus:outline-none focus:border-cyan-500 font-mono"
          >
            <option value="annotated">Annotated BBoxes</option>
            <option value="heatmap">Thermal Heatmap</option>
            <option value="side_by_side">Side-by-Side HUD</option>
          </select>

          {/* Run / Stop Button */}
          <button
            onClick={() => setIsRunning(!isRunning)}
            className={`flex items-center space-x-2 px-5 py-2 rounded-lg text-xs font-bold font-mono tracking-wider transition-all uppercase ${
              isRunning
                ? 'bg-rose-500/20 text-rose-400 border border-rose-500/40 hover:bg-rose-500/30'
                : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 hover:bg-emerald-500/30'
            }`}
          >
            {isRunning ? (
              <>
                <Pause className="w-4 h-4" />
                <span>Halt Inspection</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                <span>Start Machine Stream</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Main View Area */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Stream Canvas (8 Cols) */}
        <div className="lg:col-span-8 bg-industrial-900 border border-industrial-800 rounded-xl p-4 flex flex-col">
          <div className="relative flex-1 min-h-[460px] bg-industrial-950 rounded-lg border border-industrial-850 flex items-center justify-center overflow-hidden">
            {currentFrameUrl ? (
              <img
                src={currentFrameUrl}
                alt="Live Inspection Stream"
                className="max-h-[500px] w-auto object-contain rounded"
              />
            ) : (
              <div className="text-center p-8 text-slate-500 font-mono text-xs flex flex-col items-center gap-2">
                <Camera className="w-10 h-10 text-slate-700" />
                <span>Camera feed inactive. Click "Start Machine Stream" to initiate inspection.</span>
              </div>
            )}

            {/* Video & Canvas (Hidden for webcam processing) */}
            <video ref={videoRef} className="hidden" playsInline muted />
            <canvas ref={canvasRef} className="hidden" />

            {/* Stream HUD Bar */}
            <div className="absolute top-4 right-4 bg-industrial-950/85 backdrop-blur-md border border-industrial-700/60 px-3 py-1.5 rounded-lg text-xs font-mono flex items-center space-x-3">
              <span className="flex items-center gap-1.5 text-cyan-400">
                <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping"></span>
                FPS: <strong>{fps}</strong>
              </span>
              <span className="text-slate-500">|</span>
              <span className="text-slate-300">
                Latency: <strong>{latency} ms</strong>
              </span>
            </div>
          </div>
        </div>

        {/* Real-Time Telemetry & Defect Counters (4 Cols) */}
        <div className="lg:col-span-4 space-y-4">
          {/* Live Status Card */}
          <div className="bg-industrial-900 border border-industrial-800 rounded-xl p-5">
            <h3 className="text-xs font-mono uppercase text-slate-400 tracking-wider mb-3">
              Current Unit Status
            </h3>

            {verdict ? (
              <div className={`p-4 rounded-xl border flex items-center space-x-3 ${
                verdict.verdict === 'PASS'
                  ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                  : verdict.verdict === 'REVIEW'
                  ? 'bg-amber-500/10 border-amber-500/30 text-amber-400'
                  : 'bg-red-500/10 border-red-500/30 text-red-400'
              }`}>
                {verdict.verdict === 'PASS' && <CheckCircle2 className="w-8 h-8 flex-shrink-0" />}
                {verdict.verdict === 'REVIEW' && <AlertTriangle className="w-8 h-8 flex-shrink-0" />}
                {verdict.verdict === 'FAIL' && <XCircle className="w-8 h-8 flex-shrink-0" />}
                <div>
                  <div className="text-xl font-bold font-mono">{verdict.verdict}</div>
                  <div className="text-[11px] text-slate-300">{verdict.status_text}</div>
                </div>
              </div>
            ) : (
              <div className="p-4 rounded-lg bg-industrial-950 text-slate-500 font-mono text-xs text-center">
                Awaiting active frames...
              </div>
            )}
          </div>

          {/* Active Defects in Field-Of-View */}
          <div className="bg-industrial-900 border border-industrial-800 rounded-xl p-5">
            <h3 className="text-xs font-mono uppercase text-slate-400 tracking-wider mb-3 flex items-center justify-between">
              <span>FOV Anomalies</span>
              <span className="text-rose-400 text-[11px] font-bold">
                {defects.length} Present
              </span>
            </h3>

            {defects.length > 0 ? (
              <div className="space-y-2 max-h-[220px] overflow-y-auto">
                {defects.map((d, i) => (
                  <div key={i} className="p-2.5 rounded bg-industrial-950 border border-industrial-800 flex items-center justify-between text-xs font-mono">
                    <div className="flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-rose-500"></span>
                      <span className="text-slate-200 uppercase font-bold">{d.class_name}</span>
                    </div>
                    <span className="text-cyan-400">{(d.confidence * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="py-6 text-center text-xs font-mono text-slate-500">
                Zero active defect triggers in current optical frame.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
