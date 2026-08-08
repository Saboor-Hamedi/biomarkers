import React from 'react'
import AnalyticView from './AnalyticView'
import { Target } from 'lucide-react'

const Heatmap = ({ activeTab, heatmapData }) => {
  // Data shape from /heatmap: {x: model, y: metric, value: score}
  // Rows = metrics (left column), Columns = models (top row)
  const models = heatmapData ? Array.from(new Set(heatmapData.map(d => d.x))) : []
  const metrics = heatmapData ? Array.from(new Set(heatmapData.map(d => d.y))) : []

  const cellColor = (val) => {
    if (val >= 95) return 'bg-blue-500/80'
    if (val >= 90) return 'bg-blue-500/60'
    if (val >= 80) return 'bg-blue-500/40'
    if (val >= 70) return 'bg-blue-500/20'
    return 'bg-gray-800'
  }

  const fmt = (val) => (Number.isInteger(val) ? `${val}.00` : val.toFixed(2))

  return (
    <div className={activeTab === 'heatmap' ? 'block' : 'hidden'}>
      <AnalyticView
        title="Model Performance Heatmap"
        icon={Target}
        explanation="Cross-references each model against its evaluation metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC). Values are percentages from the 5-fold cross-validation evaluation."
      >
        <div className="h-[400px] flex items-center justify-center p-4">
          {heatmapData && models.length > 0 ? (
            <div className="w-full max-w-3xl overflow-x-auto rounded-lg border border-gray-800 bg-[#0d1117] shadow-xl">
              {/* Header Row: metric label + model names */}
              <div className="grid" style={{ gridTemplateColumns: `120px repeat(${models.length}, 1fr)` }}>
                <div className="p-3 border-b border-r border-gray-800 bg-gray-900/50 text-[10px] font-bold text-gray-500">Metric</div>
                {models.map(m => (
                  <div key={`header-${m}`} className="p-3 text-[10px] font-bold text-gray-400 text-center border-b border-r border-gray-800 bg-gray-900/50 truncate">
                    {m}
                  </div>
                ))}
              </div>

              {/* Data Rows: one per metric */}
              {metrics.map(metric => (
                <div key={`row-${metric}`} className="grid" style={{ gridTemplateColumns: `120px repeat(${models.length}, 1fr)` }}>
                  <div className="p-3 text-[10px] font-bold text-gray-400 border-b border-r border-gray-800 bg-gray-900/50 flex items-center">
                    {metric}
                  </div>
                  {models.map(model => {
                    const cell = heatmapData.find(d => d.x === model && d.y === metric)
                    const val = cell ? cell.value : 0
                    return (
                      <div
                        key={`${model}-${metric}`}
                        className={`p-4 border-b border-r border-gray-800 flex items-center justify-center text-xs font-mono font-bold transition-all duration-300 hover:brightness-150 hover:scale-[1.02] cursor-default ${cellColor(val)} text-white`}
                        title={`${model} ${metric}: ${val}%`}
                      >
                        {fmt(val)}%
                      </div>
                    )
                  })}
                </div>
              ))}
            </div>
          ) : (
            <div className="text-gray-500 text-[10px] font-bold tracking-widest animate-pulse">
              {models.length === 0 ? 'INITIALIZING MODEL PERFORMANCE HEATMAP...' : 'LOADING...'}
            </div>
          )}
        </div>
      </AnalyticView>
    </div>
  )
}

export default Heatmap
