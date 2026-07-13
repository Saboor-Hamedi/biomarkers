import React from 'react'
import AnalyticView from './AnalyticView'
import { ShieldCheck } from 'lucide-react'


const Cm = ({ activeTab, metrics, cmTableData }) => {
  return (
<div className={activeTab === 'cm' ? 'block' : 'hidden'}>
      {(() => {
        const cm = metrics?.cm || [
          [0, 0],
          [0, 0]
        ]
        return (
          <AnalyticView
        title="Confusion Matrix"
        icon={ShieldCheck}
        explanation="The Confusion Matrix identifies classification errors. TN and TP are successes, while FP and FN represent misdiagnoses requiring further audit."
        tableData={cmTableData}
        columns={['Parameter', 'Audit Count', 'Clinical Impact']}
      >
        <div className="flex flex-col items-center gap-4 py-10">
          <div className="flex gap-4 w-full max-w-[500px]">
            <div className="flex-1 aspect-square bg-blue-600/10 border-2 border-blue-500/20 flex flex-col items-center justify-center rounded-xl shadow-inner">
              <span className="text-5xl font-black text-white">{cm[0][0]}</span>
              <span className="text-[10px] font-black text-blue-500 mt-2">
                True Negative
              </span>
            </div>
            <div className="flex-1 aspect-square bg-red-600/5 border-2 border-red-500/10 flex flex-col items-center justify-center rounded-xl">
              <span className="text-5xl font-black text-white">{cm[0][1]}</span>
              <span className="text-[10px] font-black text-red-500 mt-2">
                False Positive
              </span>
            </div>
          </div>
          <div className="flex gap-4 w-full max-w-[500px]">
            <div className="flex-1 aspect-square bg-red-600/5 border-2 border-red-500/10 flex flex-col items-center justify-center rounded-xl">
              <span className="text-5xl font-black text-white">{cm[1][0]}</span>
              <span className="text-[10px] font-black text-red-500 mt-2">
                False Negative
              </span>
            </div>
            <div className="flex-1 aspect-square bg-green-600/20 border-2 border-green-500/30 flex flex-col items-center justify-center rounded-xl shadow-[0_0_20px_rgba(34,197,94,0.1)]">
              <span className="text-5xl font-black text-white">{cm[1][1]}</span>
              <span className="text-[10px] font-black text-green-500 mt-2">
                True Positive
              </span>
            </div>
          </div>
        </div>
        <div className="mt-8 grid grid-cols-2 gap-4 text-[8px] font-bold tracking-widest text-gray-500 max-w-[500px] mx-auto">
          <div className="text-center">Actual Negative</div>
          <div className="text-center">Actual Positive</div>
        </div>
      </AnalyticView>
        );
      })()}
      </div>
  )
}

export default Cm
