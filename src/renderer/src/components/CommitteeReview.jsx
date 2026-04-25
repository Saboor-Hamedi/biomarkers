import { CheckCircle2, XCircle, BarChart3 } from 'lucide-react'
import { cn } from '../lib/utils'

const CommitteeReview = ({ artifacts, prediction }) => {
  // Extract models from prediction or artifacts
  const models = prediction?.models 
    ? Object.entries(prediction.models).map(([name, score]) => ({ name, score, scoreValue: parseFloat(score) }))
    : artifacts.filter(a => a.includes('_model')).map(artifact => {
        const name = artifact.replace('_model.pkl', '').replace(/^\w/, c => c.toUpperCase());
        return { name, score: '---', scoreValue: 0 };
      });

  // Sort by score (descending)
  const sortedModels = [...models].sort((a, b) => b.scoreValue - a.scoreValue);

  return (
    <div className="bg-[#0d1117] border border-gray-800 rounded-lg p-8">
      <h2 className="text-[10px] font-bold uppercase tracking-[0.2em] text-blue-500 mb-8 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <BarChart3 size={14} className="text-blue-500" />
          AI Committee Review
        </div>
        <span className="text-[8px] bg-blue-500/10 text-blue-500 px-1.5 py-0.5 rounded border border-blue-500/20">{artifacts.length} Linked Models</span>
      </h2>
      
      <div className="space-y-6">
        {sortedModels.length > 0 ? (
          sortedModels.map((model, i) => (
            <div key={i} className="space-y-2 group">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <CheckCircle2 size={12} className={prediction ? (model.scoreValue > 50 ? "text-red-500" : "text-green-500") : "text-gray-600"} />
                  <span className="text-[10px] font-black uppercase text-gray-300 tracking-widest">{model.name}</span>
                </div>
                <span className="text-[10px] font-black text-white font-mono">
                  {model.score === '---' ? '---' : parseFloat(model.score).toFixed(2)}
                </span>
              </div>
              <div className="h-1.5 bg-black rounded-full overflow-hidden border border-gray-800 shadow-inner">
                <div 
                  className={cn(
                    "h-full transition-all duration-1000 ease-out", 
                    model.scoreValue > 50 
                      ? "bg-red-500 shadow-[0_0_10px_rgba(239,68,68,0.4)]" 
                      : "bg-green-500 shadow-[0_0_10px_rgba(34,197,94,0.4)]"
                  )} 
                  style={{ width: model.score === '---' ? '0%' : model.score }} 
                />
              </div>
            </div>
          ))
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
