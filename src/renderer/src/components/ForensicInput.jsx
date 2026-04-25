import { Activity, Zap } from 'lucide-react'
import { cn } from '../lib/utils'

const InputGroup = ({ label, value, onChange, unit, icon: Icon }) => (
  <div className="flex flex-col gap-1.5 group flex-1">
    <label className="text-[8px] font-black text-gray-600 uppercase tracking-widest flex items-center gap-1 group-hover:text-blue-500 transition-colors truncate">
      <Icon size={10} />
      {label}
    </label>
    <div className="relative">
      <input 
        type="number"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
        className="w-full bg-black border border-gray-800 rounded p-2 text-[10px] focus:outline-none focus:border-blue-500/50 transition-all font-mono text-white pr-10 shadow-inner"
      />
      <span className="absolute right-2 top-1/2 -translate-y-1/2 text-[7px] font-black text-gray-700 uppercase">{unit}</span>
    </div>
  </div>
);

const ForensicInput = ({ inputs, onInputChange, onPredict, loading, disabled }) => {
  return (
    <div className="bg-[#0d1117] border border-gray-800 rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-[9px] font-bold uppercase tracking-[0.2em] text-gray-500 flex items-center gap-2">
          <Activity size={12} className="text-blue-500" />
          Forensic Parameters
        </h2>
        
        <button 
          onClick={onPredict}
          disabled={loading || disabled}
          className={cn(
            "px-4 py-2 rounded text-[8px] font-black uppercase tracking-[0.2em] transition-all flex items-center gap-2",
            loading || disabled 
              ? "bg-gray-800 text-gray-600 cursor-not-allowed" 
              : "bg-blue-600 text-white hover:bg-blue-500 shadow-[0_0_15px_rgba(37,99,235,0.2)]"
          )}
        >
          {loading ? (
            <div className="w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin" />
          ) : (
            <>
              <Zap size={10} />
              Audit
            </>
          )}
        </button>
      </div>

      <div className="flex gap-4">
        <InputGroup 
          label="AFP" 
          value={inputs.AFP_pg_per_ml} 
          onChange={(v) => onInputChange('AFP_pg_per_ml', v)}
          unit="pg/ml"
          icon={Activity}
        />
        <InputGroup 
          label="CA125" 
          value={inputs.CA125_U_per_ml} 
          onChange={(v) => onInputChange('CA125_U_per_ml', v)}
          unit="U/ml"
          icon={Activity}
        />
        <InputGroup 
          label="PSA" 
          value={inputs.PSA_pg_per_ml} 
          onChange={(v) => onInputChange('PSA_pg_per_ml', v)}
          unit="pg/ml"
          icon={Zap}
        />
      </div>
    </div>
  )
}

export default ForensicInput
