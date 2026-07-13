import React from 'react'
import AnalyticView from './AnalyticView'


import { Zap } from 'lucide-react'

const Importance = ({ activeTab, importancePrimaryModel, importanceTableData, importanceChartData }) => {
  return (
<div className={activeTab === 'importance' ? 'block' : 'hidden'}>
        <AnalyticView
        title="Biomarker Influence"
        icon={Zap}
        explanation="Feature Importance quantifies the contribution of each biomarker to the final neural verdict. Higher values indicate greater diagnostic weight."
        tableData={importanceTableData}
        columns={['Model', 'Biomarker', 'Neural Weight']}
      >
        <div className="h-[450px] overflow-y-auto custom-scrollbar pr-4 space-y-6">
          {importanceChartData.sort((a, b) => b.value - a.value).map(({ name, value }, i) => (
            <div key={i} className="space-y-3">
              <div className="flex justify-between text-[11px] font-black tracking-widest">
                <span className="text-gray-300">{name.replace(/_/g, ' ')}</span>
                <span className="text-blue-400 font-mono">
                  {(value * 100).toFixed(1)}% Weight
                </span>
              </div>
              <div className="h-3 bg-gray-900 rounded-full overflow-hidden border border-gray-800">
                <div
                  className="h-full bg-blue-500 shadow-[0_0_12px_rgba(59,130,246,0.6)] transition-all duration-1000 rounded-full"
                  style={{ width: `${value * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </AnalyticView>
      </div>
  )
}

export default Importance
