import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import InspectionStudio from './components/InspectionStudio';
import Dashboard from './components/Dashboard';
import InspectionHistory from './components/InspectionHistory';
import ModelAnalytics from './components/ModelAnalytics';
import LiveCameraModal from './components/LiveCameraModal';
import { api } from './services/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('studio');
  const [healthInfo, setHealthInfo] = useState(null);

  useEffect(() => {
    async function checkHealth() {
      try {
        const data = await api.getHealth();
        setHealthInfo(data);
      } catch (err) {
        console.warn('Backend API connection pending:', err);
      }
    }
    checkHealth();
  }, []);

  return (
    <div className="min-h-screen bg-industrial-950 text-slate-100 flex flex-col font-sans">
      {/* Top Navbar */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        healthInfo={healthInfo}
      />

      {/* Main Body Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === 'studio' && (
          <InspectionStudio onInspectionComplete={() => {}} />
        )}
        {activeTab === 'dashboard' && (
          <Dashboard onNavigateToStudio={() => setActiveTab('studio')} />
        )}
        {activeTab === 'history' && (
          <InspectionHistory />
        )}
        {activeTab === 'analytics' && (
          <ModelAnalytics />
        )}
        {activeTab === 'camera' && (
          <LiveCameraModal />
        )}
      </main>

      {/* Industrial Footer */}
      <footer className="border-t border-industrial-850 bg-industrial-900/60 py-4 px-6 text-center text-xs font-mono text-slate-500">
        VisionGuard AI – Real-Time Industrial Defect Detection & Quality Inspection Platform • PyTorch + YOLOv8 + PatchCore
      </footer>
    </div>
  );
}
