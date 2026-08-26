import { useEffect, useState } from 'react'
import { User, CloudDownload, RefreshCw, Power, CheckCircle, AlertTriangle } from 'lucide-react'
import { cn } from '../lib/utils'

const Header = () => {
  const [status, setStatus] = useState({ type: 'idle' })

  useEffect(() => {
    let unsub = () => {}
    if (window.api?.update) {
      unsub = window.api.update.onStatus((payload) => setStatus(payload))
    }
    return unsub
  }, [])

  const handleClick = async () => {
    if (status.type === 'downloaded') {
      window.api.update.restart()
      return
    }
    if (status.type === 'available' || status.type === 'not-available' || status.type === 'idle' || status.type === 'error') {
      setStatus({ type: 'checking' })
      await window.api.update.check()
      return
    }
    if (status.type === 'available') {
      setStatus({ type: 'downloading', percent: 0 })
      await window.api.update.download()
    }
  }

  const handleDownload = async () => {
    setStatus((prev) => ({ ...prev, type: 'downloading', percent: prev.percent ?? 0 }))
    await window.api.update.download()
  }

  const busy = status.type === 'checking' || status.type === 'downloading'

  return (
    <header className="h-[50px] min-h-[50px] border-b border-gray-800 bg-[#0e1117] flex items-center justify-end px-6 sticky top-0 z-10">
      <div className="flex items-center gap-3">
        {/* Update button */}
        <button
          onClick={handleClick}
          disabled={busy}
          className={cn(
            'flex items-center gap-2 px-3 py-1.5 rounded text-[8px] font-black tracking-[0.2em] transition-all border',
            status.type === 'available'
              ? 'bg-blue-600 text-white border-blue-500/50 hover:bg-blue-500 shadow-[0_0_15px_rgba(37,99,235,0.3)]'
              : status.type === 'downloading'
                ? 'bg-blue-500/10 text-blue-400 border-blue-500/30'
                : status.type === 'downloaded'
                  ? 'bg-emerald-600 text-white border-emerald-500/50 hover:bg-emerald-500 shadow-[0_0_15px_rgba(16,185,129,0.3)]'
                  : status.type === 'error'
                    ? 'bg-red-500/10 text-red-400 border-red-500/30'
                    : 'bg-white/5 text-gray-400 border-gray-800 hover:bg-white/10 hover:text-white'
          )}
        >
          {status.type === 'downloading' ? (
            <>
              <RefreshCw size={10} className="animate-spin" />
              Downloading {status.percent ?? 0}%
            </>
          ) : status.type === 'downloaded' ? (
            <>
              <Power size={10} />
              Restart to Update
            </>
          ) : status.type === 'available' ? (
            <>
              <CloudDownload size={10} />
              Update Available
            </>
          ) : status.type === 'checking' ? (
            <>
              <RefreshCw size={10} className="animate-spin" />
              Checking...
            </>
          ) : status.type === 'error' ? (
            <>
              <AlertTriangle size={10} />
              Update Failed
            </>
          ) : status.type === 'not-available' ? (
            <>
              <CheckCircle size={10} />
              Up to Date
            </>
          ) : (
            <>
              <CloudDownload size={10} />
              Check Update
            </>
          )}
        </button>

        {status.type === 'available' && (
          <button
            onClick={handleDownload}
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded text-[8px] font-black tracking-[0.2em] bg-blue-500/10 text-blue-400 border border-blue-500/30 hover:bg-blue-500/20 transition-all"
          >
            <CloudDownload size={9} />
            Download
          </button>
        )}

        {status.type === 'downloading' && (
          <div className="w-28 h-1 bg-gray-800 rounded overflow-hidden">
            <div
              className="h-full bg-blue-500 transition-all duration-300"
              style={{ width: `${status.percent ?? 0}%` }}
            />
          </div>
        )}

        <div className="flex items-center gap-3 group cursor-pointer hover:bg-white/5 p-1.5 rounded transition-all">
          <div className="flex flex-col items-end">
            <span className="text-[8px] text-gray-500 font-bold tracking-widest">Operator</span>
            <span className="text-[10px] text-white font-black tracking-tight">ADMIN_CORE</span>
          </div>
          <div className="w-7 h-7 rounded-full bg-gradient-to-br from-blue-600 to-indigo-700 flex items-center justify-center text-[10px] font-bold text-white border border-blue-500/30">
            <User size={14} />
          </div>
        </div>
      </div>
    </header>
  )
}

export default Header
