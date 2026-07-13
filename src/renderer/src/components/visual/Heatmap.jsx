import React from 'react'
import AnalyticView from './AnalyticView'
import { Target } from 'lucide-react'

const Heatmap = ({ activeTab, heatmapData }) => {
  const models = heatmapData ? Array.from(new Set(heatmapData.map(d => d.x))) : [];

  return (
    <div className={activeTab === 'heatmap' ? 'block' : 'hidden'}>
      <AnalyticView
        title="Model Consensus Matrix"
        icon={Target}
        explanation="A correlation heatmap cross-referencing model agreement across the entire cohort. High values indicate models that share identical decision logic."
      >
        <div className="h-[400px] flex items-center justify-center p-4">
          {heatmapData ? (
            <div className="w-full max-w-2xl overflow-hidden rounded-lg border border-gray-800 bg-[#0d1117] shadow-xl">
              {/* Header Row */}
              <div className="grid" style={{ gridTemplateColumns: `120px repeat(${models.length}, 1fr)` }}>
                <div className="p-3 border-b border-r border-gray-800 bg-gray-900/50"></div>
                {models.map(m => (
                  <div key={`header-${m}`} className="p-3 text-[10px] font-bold text-gray-400 text-center border-b border-r border-gray-800 bg-gray-900/50 truncate">
                    {m}
                  </div>
                ))}
              </div>
              
              {/* Data Rows */}
              {models.map(mY => (
                <div key={`row-${mY}`} className="grid" style={{ gridTemplateColumns: `120px repeat(${models.length}, 1fr)` }}>
                  <div className="p-3 text-[10px] font-bold text-gray-400 border-b border-r border-gray-800 bg-gray-900/50 flex items-center">
                    {mY}
                  </div>
                  {models.map(mX => {
                    const cell = heatmapData.find(d => d.x === mX && d.y === mY);
                    const val = cell ? cell.value : 0;
                    
                    let bgClass = "bg-gray-800";
                    let textClass = "text-white";
                    
                    if (val === 100) bgClass = "bg-blue-500/80";
                    else if (val >= 90) bgClass = "bg-blue-500/60";
                    else if (val >= 80) bgClass = "bg-blue-500/40";
                    else if (val >= 70) bgClass = "bg-blue-500/20";
                    else bgClass = "bg-gray-800";
                    
                    return (
                      <div 
                        key={`${mX}-${mY}`} 
                        className={`p-4 border-b border-r border-gray-800 flex items-center justify-center text-xs font-mono font-bold transition-all duration-300 hover:brightness-150 hover:scale-[1.02] cursor-default ${bgClass} ${textClass}`}
                        title={`${mY} vs ${mX}: ${val}% Agreement`}
                      >
                        {val}%
                      </div>
                    )
                  })}
                </div>
              ))}
            </div>
          ) : (
            <div className="text-gray-500 text-[10px] font-bold tracking-widest animate-pulse">
              INITIALIZING CONSENSUS MATRIX...
            </div>
          )}
        </div>
      </AnalyticView>
    </div>
  )
}

export default Heatmap
