import { useEffect, useState } from 'react'
import { User, CloudDownload, RefreshCw, Power, CheckCircle, AlertTriangle, Bot } from 'lucide-react'
import { cn } from '../lib/utils'

const Header = ({ onOpenChat }) => {
  const [status, setStatus] = useState({ type: 'idle' })

  useEffect(() => {
    let unsub = () => {}
    if (window.api?.update) {
      unsub = window.api.update.onStatus((payload) => setStatus(payload))
    }
    return unsub
  }, [])

  const handleClick = async () => {
    switch (status.type) {
      case 'downloaded':
        window.api.update.restart()
        return
      case 'downloading':
        return // already in progress, ignore clicks
      case 'available':
        setStatus((prev) => ({ ...prev, type: 'downloading', percent: 0 }))
        await window.api.update.download()
        return
      default:
        // idle, checking, not-available, error → check for updates
        setStatus({ type: 'checking' })
        await window.api.update.check()
    }
  }

  const busy = status.type === 'checking' || status.type === 'downloading'
  const progress = status.type === 'downloading' ? status.percent ?? 0 : 0

  let icon = <CloudDownload size={10} />
  let label = 'Check Update'
  if (status.type === 'checking') {
    icon = <RefreshCw size={10} className="animate-spin" />
    label = 'Checking...'
  } else if (status.type === 'available') {
    icon = <CloudDownload size={10} />
    label = 'Update Available'
  } else if (status.type === 'downloading') {
    icon = <RefreshCw size={10} className="animate-spin" />
    label = `Downloading ${progress}%`
  } else if (status.type === 'downloaded') {
    icon = <Power size={10} />
    label = 'Restart to Update'
  } else if (status.type === 'not-available') {
    icon = <CheckCircle size={10} />
    label = 'Up to Date'
  } else if (status.type === 'error') {
    icon = <AlertTriangle size={10} />
    label = 'Update Failed'
  }

  return (
    <header className="h-[50px] min-h-[50px] border-b border-gray-800 bg-[#0e1117] flex items-center justify-end px-6 sticky top-0 z-10">
      <div className="flex items-center gap-3">
        <button
          onClick={onOpenChat}
          className="relative flex items-center gap-2 px-3 py-1.5 rounded text-[8px] font-black tracking-[0.2em] transition-all border overflow-hidden bg-blue-600 text-white border-blue-500/50 hover:bg-blue-500"
        >
          <span className="relative flex items-center gap-2">
            <Bot size={10} />
            AI COPILOT
          </span>
        </button>
        <button
          onClick={handleClick}
          disabled={busy}
          className={cn(
            'relative flex items-center gap-2 px-3 py-1.5 rounded text-[8px] font-black tracking-[0.2em] transition-all border overflow-hidden',
            status.type === 'available'
              ? 'bg-blue-600 text-white border-blue-500/50 hover:bg-blue-500'
              : status.type === 'downloading'
                ? 'bg-blue-500/10 text-blue-400 border-blue-500/30 cursor-wait'
                : status.type === 'downloaded'
                  ? 'bg-emerald-600 text-white border-emerald-500/50 hover:bg-emerald-500'
                  : status.type === 'error'
                    ? 'bg-red-500/10 text-red-400 border-red-500/30 hover:bg-red-500/20'
                    : 'bg-white/5 text-gray-400 border-gray-800 hover:bg-white/10 hover:text-white'
          )}
        >
          {status.type === 'downloading' && (
            <span
              className="absolute inset-y-0 left-0 bg-blue-500/25 transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          )}
          <span className="relative flex items-center gap-2">
            {icon}
            {label}
          </span>
        </button>

        <div className="flex items-center gap-3 group cursor-pointer hover:bg-white/5 p-1.5 rounded transition-all">
          <div className="w-7 h-7 rounded-full bg-gradient-to-br from-blue-600 to-indigo-700 flex items-center justify-center text-[10px] font-bold text-white border border-blue-500/30">
            <User size={14} />
          </div>
        </div>
      </div>
    </header>
  )
}

export default Header
