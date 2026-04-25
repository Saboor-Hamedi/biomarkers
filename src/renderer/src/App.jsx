import { useState, useEffect, useMemo, useCallback } from 'react'
import Sidebar from './components/Sidebar'
import Header from './components/Header'
import ArtifactPicker from './components/ArtifactPicker'
import StatCard from './components/StatCard'
import ForensicInput from './components/ForensicInput'
import CommitteeReview from './components/CommitteeReview'
import VisualAnalytics from './components/VisualAnalytics'
import PatientDetailModal from './components/PatientDetailModal'
import ChatBot from './components/ChatBot'
import SettingsModal from './components/SettingsModal'
import { Activity, TrendingUp, ShieldAlert, Search } from 'lucide-react'
import { cn } from './lib/utils'

function App() {
  const [activeTab, setActiveTab] = useState('dashboard')
  const [artifacts, setArtifacts] = useState([])
  const [prediction, setPrediction] = useState(null)
  const [tsneData, setTsneData] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [importanceData, setImportanceData] = useState(null)
  const [distributionData, setDistributionData] = useState(null)
  const [trajectoryData, setTrajectoryData] = useState(null)
  const [shapData, setShapData] = useState(null)
  const [boundariesData, setBoundariesData] = useState(null)
  const [heatmapData, setHeatmapData] = useState(null)
  const [counterfactualData, setCounterfactualData] = useState(null)
  const [topPatients, setTopPatients] = useState([])
  const [visibleCount, setVisibleCount] = useState(10)
  const [selectedPatient, setSelectedPatient] = useState(null)
  const [engineStatus, setEngineStatus] = useState('waiting')
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)
  const [artifactFiles, setArtifactFiles] = useState([])
  const [loading, setLoading] = useState(false)
  const [auditHistory, setAuditHistory] = useState([])
  const [inputs, setInputs] = useState({
    AFP_pg_per_ml: 0,
    CA125_U_per_ml: 0,
    PSA_pg_per_ml: 0
  })
  const [resetKey, setResetKey] = useState(0)

  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/audit')
      if (!response.ok) throw new Error('Server Offline')
      const data = await response.json()
      setArtifacts(data.artifacts || [])

      // Also fetch top patients if engine is ready
      if (data.status === 'ready') {
        const topRes = await fetch('http://127.0.0.1:8000/top-patients')
        const topData = await topRes.json()
        if (Array.isArray(topData)) {
          setTopPatients(topData)
        } else if (topData.error) {
          console.error('Ranking fetch failed:', topData.error)
        }
      }
    } catch (err) {
      console.error('Status fetch failed', err)
    }
  }, [])

  const fetchVisuals = useCallback(async () => {
    try {
      const [tsneRes, metricsRes, importanceRes, distRes, boundRes, heatRes] = await Promise.all([
        fetch('http://127.0.0.1:8000/tsne'),
        fetch('http://127.0.0.1:8000/metrics'),
        fetch('http://127.0.0.1:8000/importance'),
        fetch('http://127.0.0.1:8000/distributions'),
        fetch('http://127.0.0.1:8000/boundaries'),
        fetch('http://127.0.0.1:8000/heatmap')
      ])
      const tsne = await tsneRes.json()
      const metricsData = await metricsRes.json()
      const importance = await importanceRes.json()
      const distributions = await distRes.json()
      const bounds = await boundRes.json()
      const heat = await heatRes.json()
      setTsneData(tsne)
      setMetrics(metricsData)
      setImportanceData(importance)
      setDistributionData(distributions)
      setBoundariesData(bounds)
      setHeatmapData(heat)
    } catch (err) {
      console.error('Visuals fetch failed', err)
    }
  }, [])

  useEffect(() => {
    fetchStatus()
    const interval = setInterval(fetchStatus, 2000)
    return () => clearInterval(interval)
  }, [fetchStatus])

  useEffect(() => {
    if (['trajectory', 'shap', 'boundaries', 'heatmap', 'counterfactual', 'calibration', 'roc', 'pr', 'cm', 'tsne', 'importance', 'distribution'].includes(activeTab)) {
      fetchVisuals()
    }
  }, [activeTab, fetchVisuals])

  const handlePredict = async () => {
    setLoading(true)
    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features: inputs })
      })
      const data = await response.json()
      if (data.error) throw new Error(data.error)
      
      try {
        const [trajRes, shapRes, cfRes] = await Promise.all([
          fetch('http://127.0.0.1:8000/trajectory', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ features: inputs }) }),
          fetch('http://127.0.0.1:8000/shap', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ features: inputs }) }),
          fetch('http://127.0.0.1:8000/counterfactual', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ features: inputs }) })
        ])
        const trajData = await trajRes.json()
        const shapDataJson = await shapRes.json()
        const cfData = await cfRes.json()
        setTrajectoryData(trajData)
        setShapData(shapDataJson)
        setCounterfactualData(cfData)
      } catch (e) {
        console.error('Advanced endpoints fetch failed', e)
      }

      setPrediction(data)
      setAuditHistory((prev) =>
        [
          {
            id: Date.now(),
            timestamp: new Date().toLocaleTimeString(),
            score: data.risk_score,
            verdict: data.prediction
          },
          ...prev
        ].slice(0, 20)
      )
    } catch (err) {
      alert('Analysis Error: ' + err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleReset = async () => {
    if (confirm('Reset forensic storage?')) {
      await window.electron.ipcRenderer.invoke('reset-artifacts')
      setPrediction(null)
      setAuditHistory([])
      setResetKey((prev) => prev + 1)
      fetchStatus()
    }
  }

  const handleInputChange = useCallback((key, value) => {
    setInputs((prev) => ({ ...prev, [key]: value }))
  }, [])

  const stats = useMemo(
    () => [
      {
        label: 'Risk Score',
        value: prediction ? prediction.risk_score : '---',
        icon: ShieldAlert,
        color: prediction?.risk_score > 0.5 ? 'text-red-500' : 'text-green-500',
        accent: 'border-gray-800'
      },
      {
        label: 'Prediction',
        value: prediction ? prediction.prediction : 'Standby',
        icon: Search,
        color: 'text-white',
        accent: 'border-gray-800'
      },
      {
        label: 'Consensus',
        value: prediction ? prediction.consensus : '0.0%',
        icon: TrendingUp,
        color: 'text-blue-400',
        accent: 'border-gray-800'
      }
    ],
    [prediction]
  )

  // Analytic tabs mapping
  const isAnalyticTab = ['trajectory', 'shap', 'boundaries', 'heatmap', 'counterfactual', 'calibration', 'influence', 'tsne', 'distribution', 'metrics', 'roc', 'pr', 'cm', 'importance'].includes(
    activeTab
  )

  return (
    <div className="flex h-screen bg-[#06080a] text-white overflow-hidden font-sans">
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} onOpenSettings={() => setIsSettingsOpen(true)} />
      <div className="flex-1 flex flex-col min-w-0">
        <Header />

        <main className="flex-1 overflow-y-auto p-8 custom-scrollbar">
          <div className="max-w-6xl mx-auto">
            {activeTab === 'dashboard' && (
              <div className="space-y-8 animate-in fade-in duration-500 max-w-5xl mx-auto pb-20">
                <header className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <Activity className="text-blue-500" size={18} />
                    <h1 className="text-lg font-black tracking-tight text-white uppercase italic">
                      Forensic Overview
                    </h1>
                  </div>
                  <button
                    onClick={handleReset}
                    className="px-4 py-1.5 border border-red-900/30 text-red-500 text-[8px] font-bold uppercase tracking-widest rounded hover:bg-red-500 hover:text-white transition-all"
                  >
                    Reset System
                  </button>
                </header>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  {stats.map((stat, i) => (
                    <StatCard key={i} {...stat} />
                  ))}
                </div>

                <div className="space-y-8">
                  <ArtifactPicker
                    onSync={fetchStatus}
                    files={artifactFiles}
                    setFiles={setArtifactFiles}
                  />
                  <ForensicInput
                    inputs={inputs}
                    onInputChange={handleInputChange}
                    onPredict={handlePredict}
                    loading={loading}
                    disabled={artifacts.length === 0}
                  />
                </div>
              </div>
            )}

            {activeTab === 'committee' && (
              <div className="space-y-8 animate-in slide-in-from-right-4 duration-500">
                <div className="flex items-center justify-between">
                  <h2 className="text-lg font-black uppercase tracking-tight italic">
                    Committee Performance Audit
                  </h2>
                  <p className="text-[10px] text-gray-500 font-bold uppercase tracking-widest">
                    Model Consensus Comparison
                  </p>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                  <CommitteeReview artifacts={artifacts} prediction={prediction} />
                  <div className="bg-[#0d1117] border border-gray-800 rounded-lg p-8">
                    <h3 className="text-[10px] font-bold uppercase tracking-[0.2em] text-blue-500 mb-8 flex items-center gap-2">
                      <Activity size={14} className="text-blue-500" />
                      Global Biomarker Influence
                    </h3>
                    <div className="space-y-6">
                      {importanceData && !importanceData.error ? (
                        Object.entries(
                          Object.values(importanceData).reduce((acc, curr) => {
                            Object.entries(curr).forEach(([k, v]) => {
                              acc[k] = (acc[k] || 0) + v / Object.keys(importanceData).length
                            })
                            return acc
                          }, {})
                        )
                          .sort((a, b) => b[1] - a[1])
                          .map(([feature, weight], i) => (
                            <div key={i} className="space-y-2">
                              <div className="flex justify-between text-[9px] font-black uppercase tracking-widest">
                                <span className="text-gray-400">{feature.replace(/_/g, ' ')}</span>
                                <span className="text-white font-mono">
                                  {(weight * 100).toFixed(1)}%
                                </span>
                              </div>
                              <div className="h-1 bg-gray-900 rounded-full overflow-hidden">
                                <div
                                  className="h-full bg-blue-600 shadow-[0_0_10px_rgba(37,99,235,0.4)] transition-all duration-1000"
                                  style={{ width: `${weight * 100}%` }}
                                />
                              </div>
                            </div>
                          ))
                      ) : (
                        <div className="py-12 flex flex-col items-center justify-center text-center opacity-30">
                          <Activity size={32} className="text-gray-600 mb-2" />
                          <p className="text-[8px] font-bold text-gray-500 uppercase tracking-widest">
                            Audit Required
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'ranking' && (
              <div className="space-y-8 animate-in slide-in-from-right-4 duration-500">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <TrendingUp className="text-blue-500" size={18} />
                    <h2 className="text-lg font-black uppercase tracking-tight italic">
                      Patient Risk Ranking (Top 50)
                    </h2>
                  </div>
                  <span className="text-[9px] bg-blue-500/10 text-blue-500 px-2 py-1 rounded border border-blue-500/20 font-black uppercase tracking-widest">
                    Global Audit Scan
                  </span>
                </div>
                <div className="bg-[#0d1117] border border-gray-800 rounded-lg overflow-hidden">
                  <table className="w-full text-left border-collapse">
                    <thead className="bg-black/50">
                      <tr>
                        <th className="p-4 text-[9px] font-bold text-gray-500 uppercase tracking-widest border-b border-gray-800 text-center w-16">
                          Rank
                        </th>
                        <th className="p-4 text-[9px] font-bold text-gray-500 uppercase tracking-widest border-b border-gray-800">
                          Sample ID
                        </th>
                        <th className="p-4 text-[9px] font-bold text-gray-500 uppercase tracking-widest border-b border-gray-800">
                          AFP
                        </th>
                        <th className="p-4 text-[9px] font-bold text-gray-500 uppercase tracking-widest border-b border-gray-800">
                          CA125
                        </th>
                        <th className="p-4 text-[9px] font-bold text-gray-500 uppercase tracking-widest border-b border-gray-800">
                          PSA
                        </th>
                        <th className="p-4 text-[9px] font-bold text-gray-500 uppercase tracking-widest border-b border-gray-800">
                          Neural Score
                        </th>
                        <th className="p-4 text-[9px] font-bold text-gray-500 uppercase tracking-widest border-b border-gray-800">
                          Status
                        </th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-800/50">
                      {Array.isArray(topPatients) &&
                        topPatients.slice(0, visibleCount).map((pt, i) => (
                          <tr
                            key={pt.id}
                            onClick={() => setSelectedPatient(pt)}
                            className="hover:bg-blue-500/5 transition-colors group cursor-pointer"
                          >
                            <td className="p-4 text-[9px] font-mono text-gray-600 text-center">
                              {i + 1}
                            </td>
                            <td className="p-4 text-[10px] font-bold text-gray-300 group-hover:text-blue-400">
                              {pt.id}
                            </td>
                            <td className="p-4 text-[10px] font-mono text-gray-400">{pt.AFP}</td>
                            <td className="p-4 text-[10px] font-mono text-gray-400">{pt.CA125}</td>
                            <td className="p-4 text-[10px] font-mono text-gray-400">{pt.PSA}</td>
                            <td className="p-4 text-[10px] font-mono font-bold text-blue-400">
                              {pt.score}
                            </td>
                            <td className="p-4">
                              <span
                                className={cn(
                                  'text-[8px] font-black uppercase px-2 py-1 rounded-full border',
                                  pt.status === 'Urgent'
                                    ? 'text-red-500 border-red-500/30 bg-red-500/10'
                                    : pt.status === 'Critical'
                                      ? 'text-orange-500 border-orange-500/30 bg-orange-500/10'
                                      : pt.status === 'Moderate'
                                        ? 'text-yellow-500 border-yellow-500/30 bg-yellow-500/10'
                                        : 'text-green-500 border-green-500/30 bg-green-500/10'
                                )}
                              >
                                {pt.status}
                              </span>
                            </td>
                          </tr>
                        ))}
                      {topPatients.length === 0 && (
                        <tr>
                          <td
                            colSpan="6"
                            className="p-20 text-center text-[10px] font-bold text-gray-600 uppercase tracking-widest"
                          >
                            {artifacts.length === 0
                              ? 'Engine Offline: Calibrate artifacts to begin audit'
                              : 'Scanning Population Database...'}
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>

                {Array.isArray(topPatients) && topPatients.length > 0 && (
                  <div className="flex justify-center items-center gap-12 pt-6">
                    {visibleCount < topPatients.length && (
                      <button
                        onClick={() => setVisibleCount((prev) => prev + 20)}
                        className="group flex flex-col items-center gap-2"
                      >
                        <span className="text-[8px] font-black uppercase tracking-[0.3em] text-gray-500 group-hover:text-blue-500 transition-colors">
                          Show More
                        </span>
                        <div className="w-12 h-1 bg-gray-800 rounded-full overflow-hidden">
                          <div className="w-0 group-hover:w-full h-full bg-blue-500 transition-all duration-500" />
                        </div>
                      </button>
                    )}

                    {visibleCount > 10 && (
                      <button
                        onClick={() => setVisibleCount(10)}
                        className="group flex flex-col items-center gap-2"
                      >
                        <span className="text-[8px] font-black uppercase tracking-[0.3em] text-gray-500 group-hover:text-red-500 transition-colors">
                          Show Less
                        </span>
                        <div className="w-12 h-1 bg-gray-800 rounded-full overflow-hidden">
                          <div className="w-0 group-hover:w-full h-full bg-red-500 transition-all duration-500" />
                        </div>
                      </button>
                    )}
                  </div>
                )}
              </div>
            )}

            {activeTab === 'registry' && (
              <div className="space-y-8 animate-in slide-in-from-right-4 duration-500">
                <div className="flex items-center justify-between">
                  <h2 className="text-lg font-black uppercase tracking-tight italic">
                    Audit Registry Archive
                  </h2>
                  <span className="text-[9px] bg-green-500/10 text-green-500 px-2 py-1 rounded border border-green-500/20">
                    {auditHistory.length} Entries Total
                  </span>
                </div>
                <div className="bg-[#0d1117] border border-gray-800 rounded-lg overflow-hidden">
                  <table className="w-full text-left border-collapse">
                    <thead className="bg-black/50">
                      <tr>
                        <th className="p-4 text-[9px] font-bold text-gray-500 uppercase tracking-widest border-b border-gray-800 text-center w-16">
                          #
                        </th>
                        <th className="p-4 text-[9px] font-bold text-gray-500 uppercase tracking-widest border-b border-gray-800">
                          Timestamp
                        </th>
                        <th className="p-4 text-[9px] font-bold text-gray-500 uppercase tracking-widest border-b border-gray-800">
                          Neural Score
                        </th>
                        <th className="p-4 text-[9px] font-bold text-gray-500 uppercase tracking-widest border-b border-gray-800">
                          Verdict
                        </th>
                        <th className="p-4 text-[9px] font-bold text-gray-500 uppercase tracking-widest border-b border-gray-800 text-right">
                          Scope
                        </th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-800/50">
                      {auditHistory.map((item, i) => (
                        <tr key={item.id} className="hover:bg-blue-500/5 transition-colors group">
                          <td className="p-4 text-[9px] font-mono text-gray-600 text-center">
                            {auditHistory.length - i}
                          </td>
                          <td className="p-4 text-[10px] font-bold text-gray-300">
                            {item.timestamp}
                          </td>
                          <td className="p-4 text-[10px] font-mono font-bold text-blue-400">
                            {item.score}
                          </td>
                          <td className="p-4">
                            <span
                              className={cn(
                                'text-[8px] font-black uppercase px-2 py-1 rounded-full border',
                                item.score > 0.5
                                  ? 'text-red-500 border-red-500/30 bg-red-500/10'
                                  : 'text-green-500 border-green-500/30 bg-green-500/10'
                              )}
                            >
                              {item.verdict}
                            </span>
                          </td>
                          <td className="p-4 text-[9px] text-gray-600 text-right uppercase font-bold group-hover:text-gray-400 transition-colors italic">
                            Clinical Artifact
                          </td>
                        </tr>
                      ))}
                      {auditHistory.length === 0 && (
                        <tr>
                          <td
                            colSpan="5"
                            className="p-20 text-center text-[10px] font-bold text-gray-600 uppercase tracking-widest"
                          >
                            Registry Database Empty
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {isAnalyticTab && (
              <VisualAnalytics
                activeTab={activeTab}
                prediction={prediction}
                tsneData={tsneData}
                metrics={metrics}
                importanceData={importanceData}
                distributionData={distributionData}
                trajectoryData={trajectoryData}
                shapData={shapData}
                boundariesData={boundariesData}
                heatmapData={heatmapData}
                counterfactualData={counterfactualData}
                inputs={inputs}
              />
            )}

            {selectedPatient && (
              <PatientDetailModal
                patient={selectedPatient}
                onClose={() => setSelectedPatient(null)}
              />
            )}
          </div>
        </main>
      </div>
      <ChatBot />
      <SettingsModal isOpen={isSettingsOpen} onClose={() => setIsSettingsOpen(false)} />
    </div>
  )
}

export default App
