import React, { useState, useEffect } from 'react';
import { 
  History, Download, Trash2, Filter, Search, CheckCircle2, 
  AlertTriangle, XCircle, RefreshCw, Eye
} from 'lucide-react';
import { api } from '../services/api';

export default function InspectionHistory({ onViewRecord }) {
  const [historyItems, setHistoryItems] = useState([]);
  const [totalCount, setTotalCount] = useState(0);
  const [filterVerdict, setFilterVerdict] = useState('ALL');
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(true);

  const fetchHistory = async () => {
    setLoading(true);
    try {
      const data = await api.getHistory(100, 0, filterVerdict);
      setHistoryItems(data.items || []);
      setTotalCount(data.total || 0);
    } catch (err) {
      console.error('Failed to load history:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, [filterVerdict]);

  const handleClear = async () => {
    if (!window.confirm('Are you sure you want to clear all inspection records?')) return;
    try {
      await api.clearHistory();
      fetchHistory();
    } catch (err) {
      console.error('Failed to clear history:', err);
    }
  };

  const filteredItems = historyItems.filter(item => {
    if (!searchQuery) return true;
    const q = searchQuery.toLowerCase();
    return (
      item.product_type?.toLowerCase().includes(q) ||
      item.verdict?.toLowerCase().includes(q) ||
      item.verdict_reason?.toLowerCase().includes(q) ||
      item.id?.toString().includes(q)
    );
  });

  return (
    <div className="space-y-6">
      {/* Header and Controls */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 bg-industrial-900 border border-industrial-800 rounded-xl p-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
            <History className="w-5 h-5 text-cyan-400" />
            Inspection Audit Log & History
          </h1>
          <p className="text-xs text-slate-400 font-mono">
            Persistent Quality Traceability Log with Defect Items, Timestamps & Verdicts
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          {/* Verdict Filter */}
          <div className="flex items-center space-x-1 bg-industrial-950 p-1 rounded-lg border border-industrial-800 text-xs font-mono">
            {['ALL', 'PASS', 'REVIEW', 'FAIL'].map((v) => (
              <button
                key={v}
                onClick={() => setFilterVerdict(v)}
                className={`px-2.5 py-1 rounded transition-all ${
                  filterVerdict === v
                    ? 'bg-industrial-800 text-cyan-400 font-bold border border-cyan-500/30'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {v}
              </button>
            ))}
          </div>

          {/* Download CSV */}
          <a
            href={api.getExportCsvUrl()}
            download
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-industrial-950 border border-industrial-700 text-xs text-slate-300 hover:text-white font-mono hover:bg-industrial-850"
          >
            <Download className="w-3.5 h-3.5 text-cyan-400" />
            <span>Export CSV</span>
          </a>

          {/* Clear Button */}
          <button
            onClick={handleClear}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-rose-500/10 border border-rose-500/30 text-xs text-rose-400 hover:bg-rose-500/20 font-mono"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Clear Log</span>
          </button>
        </div>
      </div>

      {/* Search Input */}
      <div className="relative">
        <Search className="w-4 h-4 text-slate-500 absolute left-3.5 top-3" />
        <input
          type="text"
          placeholder="Search by ID, substrate product, flaw, or verdict..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full bg-industrial-900 border border-industrial-800 rounded-xl pl-10 pr-4 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 font-mono placeholder:text-slate-500"
        />
      </div>

      {/* History Data Table */}
      <div className="bg-industrial-900 border border-industrial-800 rounded-xl overflow-hidden">
        {loading ? (
          <div className="py-12 text-center text-xs font-mono text-slate-400 flex flex-col items-center gap-2">
            <RefreshCw className="w-6 h-6 animate-spin text-cyan-400" />
            <span>Retrieving inspection audit logs...</span>
          </div>
        ) : filteredItems.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead className="bg-industrial-950 text-slate-400 border-b border-industrial-800 uppercase text-[10px]">
                <tr>
                  <th className="py-3 px-4">Thumb</th>
                  <th className="py-3 px-4">Record ID</th>
                  <th className="py-3 px-4">Timestamp</th>
                  <th className="py-3 px-4">Product Substrate</th>
                  <th className="py-3 px-4">Quality Verdict</th>
                  <th className="py-3 px-4">Flaws Detected</th>
                  <th className="py-3 px-4">Anomaly Score</th>
                  <th className="py-3 px-4">Latency</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-industrial-800/60">
                {filteredItems.map((item) => {
                  const isPass = item.verdict === 'PASS';
                  const isRev = item.verdict === 'REVIEW';
                  return (
                    <tr key={item.id} className="hover:bg-industrial-850/50 transition-colors">
                      <td className="py-2 px-4">
                        <div className="w-10 h-10 rounded overflow-hidden bg-industrial-950 border border-industrial-800">
                          {item.thumbnail_base64 ? (
                            <img
                              src={item.thumbnail_base64}
                              alt={`Unit #${item.id}`}
                              className="w-full h-full object-cover"
                            />
                          ) : (
                            <div className="w-full h-full flex items-center justify-center text-[9px] text-slate-600">
                              N/A
                            </div>
                          )}
                        </div>
                      </td>
                      <td className="py-3 px-4 font-bold text-slate-200">
                        #{item.id}
                      </td>
                      <td className="py-3 px-4 text-slate-400">
                        {item.timestamp ? new Date(item.timestamp).toLocaleTimeString() : 'N/A'}
                      </td>
                      <td className="py-3 px-4 text-slate-300 font-sans">
                        {item.product_type}
                      </td>
                      <td className="py-3 px-4">
                        <span
                          className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[11px] font-bold ${
                            isPass
                              ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                              : isRev
                              ? 'bg-amber-500/10 text-amber-400 border border-amber-500/30'
                              : 'bg-red-500/10 text-red-400 border border-red-500/30'
                          }`}
                        >
                          {isPass && <CheckCircle2 className="w-3 h-3" />}
                          {isRev && <AlertTriangle className="w-3 h-3" />}
                          {!isPass && !isRev && <XCircle className="w-3 h-3" />}
                          {item.verdict}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-slate-300">
                        {item.total_defects > 0 ? (
                          <span>
                            {item.total_defects} ({item.critical_defects} Crit, {item.medium_defects} Med)
                          </span>
                        ) : (
                          <span className="text-emerald-400">0 (Clean)</span>
                        )}
                      </td>
                      <td className="py-3 px-4 text-slate-400">
                        {(item.global_anomaly_score * 100).toFixed(1)}%
                      </td>
                      <td className="py-3 px-4 text-slate-400">
                        {item.latency_ms?.toFixed(1)} ms
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="py-12 text-center text-xs font-mono text-slate-500">
            No inspection records match filter criteria.
          </div>
        )}
      </div>
    </div>
  );
}
