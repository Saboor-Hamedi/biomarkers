import { CheckCircle2, XCircle, BarChart3 } from 'lucide-react'
import { cn } from '../lib/utils'

const CommitteeReview = ({ artifacts, prediction }) => {
  return (
    <div className="bg-[#0d1117] border border-gray-800 rounded-lg p-5">
      <h2 className="text-[10px] font-bold uppercase tracking-[0.2em] text-gray-400 mb-6 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <BarChart3 size={14} className="text-indigo-400" />
          AI Committee
        </div>
        <span className="text-[8px] bg-indigo-500/10 text-indigo-400 px-1.5 py-0.5 rounded border border-indigo-500/20">{artifacts.length} Active</span>
      </h2>
      
      <div className="space-y-2">
        {artifacts.length > 0 ? (
          artifacts.filter(a => a.includes('_model')).map((artifact, i) => {
            const modelName = artifact.replace('_model.pkl', '').replace(/^\w/, c => c.toUpperCase());
            const scoreValue = prediction?.models?.[modelName];
            const score = scoreValue || '---';
            const isHigh = scoreValue && parseFloat(scoreValue) > 50;
            
            return (
              <div key={i} className="flex items-center justify-between p-3 bg-black/40 rounded-md border border-gray-800/50">
                <div className="flex items-center gap-2">
                  <CheckCircle2 size={12} className={prediction ? (isHigh ? "text-red-500" : "text-green-500") : "text-gray-600"} />
                  <span className="text-[10px] font-bold text-gray-300">{modelName}</span>
                </div>
                <span className="text-[10px] font-black text-white font-mono">{score}</span>
              </div>
            )
          })
        ) : (
          <div className="py-12 flex flex-col items-center justify-center text-center opacity-30">
            <XCircle size={32} className="text-gray-600 mb-2" />
            <p className="text-[8px] font-bold text-gray-500 uppercase tracking-widest">Committee Offline</p>
          </div>
        )}
      </div>
    </div>
  )
}

export default CommitteeReview
