import React from 'react'
import AnalyticView from './AnalyticView'
import { Target, ShieldCheck } from 'lucide-react'


const Counterfactual = ({ activeTab, counterfactualData }) => {
  return (
<div className={activeTab === 'counterfactual' ? 'block' : 'hidden'}>
        <AnalyticView
        title="What-If Engine"
        icon={ShieldCheck}
        explanation="Simulates counterfactual scenarios to determine the exact biomarker shifts required to alter the neural network's decision boundary."
      >
        <div className="flex flex-col items-center justify-center h-[400px] bg-blue-900/10 border border-blue-500/20 rounded-lg p-10 text-center">
          <Target size={48} className="text-blue-500 mb-6" />
          <h3 className="text-xl font-black tracking-widest text-white mb-4">
            Counterfactual Projection
          </h3>
          <p className="text-sm text-gray-300 max-w-2xl leading-relaxed">
            {counterfactualData?.statement ||
              'Run an audit to generate counterfactual projections.'}
          </p>
        </div>
      </AnalyticView>
      </div>
  )
}

export default Counterfactual
