import { Activity, BrainCircuit } from 'lucide-react'

const ForensicInput = ({ inputs, onInputChange, onPredict, loading, disabled }) => {
  return (
    <div className="bg-[#0d1117] border border-gray-800 rounded-lg p-5">
      <h2 className="text-[10px] font-bold uppercase tracking-[0.2em] text-white mb-6 flex items-center gap-2">
        <Activity size={12} className="text-blue-500" />
        Forensic Parameters
      </h2>
      <div className="space-y-4">
        {Object.keys(inputs).map(key => (
          <div key={key} className="flex flex-col gap-1.5">
            <label className="text-[9px] font-bold text-gray-500 uppercase tracking-widest">{key.replace(/_/g, ' ')}</label>
            <input 
              type="number"
              value={inputs[key]}
              onChange={(e) => onInputChange(key, parseFloat(e.target.value))}
              className="bg-black border border-gray-800 rounded-md p-3 text-xs focus:outline-none focus:border-blue-500 transition-all font-mono text-white"
            />
          </div>
        ))}
        <button 
          onClick={onPredict}
          disabled={loading || disabled}
          className="w-fit px-8 py-2.5 mt-2 bg-blue-600 text-white text-[9px] font-black uppercase tracking-[0.2em] rounded-md hover:bg-blue-700 transition-all disabled:opacity-20 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          {loading ? "Calculating..." : "Execute Audit"}
          <BrainCircuit size={14} />
        </button>
      </div>
    </div>
  )
}

export default ForensicInput
