import React from 'react';
import { ShieldCheck, Activity, Eye, History, BarChart3, Video, Cpu } from 'lucide-react';

export default function Navbar({ activeTab, setActiveTab, healthInfo }) {
  const tabs = [
    { id: 'studio', label: 'Inspection Studio', icon: Eye },
    { id: 'dashboard', label: 'Executive Dashboard', icon: BarChart3 },
    { id: 'history', label: 'Inspection History', icon: History },
    { id: 'analytics', label: 'Model Benchmarks', icon: Activity },
    { id: 'camera', label: 'Live Camera / Video Feed', icon: Video },
  ];

  return (
    <header className="border-b border-industrial-800 bg-industrial-900/90 backdrop-blur-md sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand */}
          <div className="flex items-center space-x-3 cursor-pointer" onClick={() => setActiveTab('studio')}>
            <div className="p-2 rounded-lg bg-gradient-to-br from-cyan-500/20 to-blue-600/30 border border-cyan-500/40 text-cyan-400">
              <ShieldCheck className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-bold text-lg tracking-tight bg-gradient-to-r from-cyan-400 via-sky-300 to-indigo-300 bg-clip-text text-transparent">
                  VisionGuard AI
                </span>
                <span className="text-[10px] uppercase font-mono font-bold px-1.5 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                  v1.0
                </span>
              </div>
              <p className="text-[11px] text-slate-400 font-mono tracking-wider">
                Industrial Defect Detection & Quality Inspection
              </p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="hidden md:flex space-x-1">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center space-x-2 px-3 py-2 rounded-lg text-xs font-medium transition-all ${
                    isActive
                      ? 'bg-industrial-800 text-cyan-400 border border-cyan-500/30 shadow-sm shadow-cyan-500/10'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-industrial-850'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </nav>

          {/* Telemetry Pill */}
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2 px-3 py-1.5 rounded-full bg-industrial-950 border border-industrial-800 text-[11px] font-mono">
              <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></div>
              <span className="text-slate-300">DUAL-ENGINE ACTIVE</span>
              <span className="text-slate-500">|</span>
              <span className="text-cyan-400 flex items-center gap-1">
                <Cpu className="w-3 h-3" />
                {healthInfo?.device?.toUpperCase() || 'CPU'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
