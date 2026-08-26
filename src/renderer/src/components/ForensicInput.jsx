import { Activity, Zap } from 'lucide-react'
import { cn } from '../lib/utils'

const ForensicInput = ({ onPredict, loading, disabled }) => {
  return (
    <div className="fixed bottom-8 right-8 z-50 w-[450px] bg-[#0d1117]/90 backdrop-blur-md border border-gray-800 rounded-lg p-4 flex items-center justify-between shadow-2xl">
      <div className="flex flex-col gap-1">
        <h2 className="text-[9px] font-bold tracking-[0.2em] text-gray-500 flex items-center gap-2">
          <Activity size={12} className="text-blue-500" />
          Forensic Parameters
        </h2>
        <p className="text-[8px] text-gray-600 font-mono tracking-widest">Execute full clinical audit on current patient profile</p>
      </div>
      
      <button 
        onClick={onPredict}
        disabled={loading || disabled}
        className={cn(
          "px-8 py-3 rounded text-[10px] font-black tracking-[0.2em] transition-all flex items-center gap-2",
          loading || disabled 
            ? "bg-gray-800 text-gray-600 cursor-not-allowed" 
            : "bg-blue-600 text-white hover:bg-blue-500 shadow-[0_0_15px_rgba(37,99,235,0.3)]"
        )}
      >
        {loading ? (
          <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
        ) : (
          <>
            <Zap size={14} />
            EXECUTE AUDIT
          </>
        )}
      </button>
    </div>
  )
}

export default ForensicInput
