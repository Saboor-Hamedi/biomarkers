import { useState } from 'react'
import { Upload, File, CheckCircle, XCircle, AlertCircle, RefreshCw } from 'lucide-react'
import { cn } from '../lib/utils'

const formatSize = (bytes) => {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]
}

const ArtifactPicker = ({ onSync, files = [], setFiles, syncing, setSyncing, onPurge }) => {
  const [isDragging, setIsDragging] = useState(false)

  const onDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const onDragLeave = () => {
    setIsDragging(false)
  }

  const handleFiles = (incomingFiles) => {
    const pklFiles = Array.from(incomingFiles).filter((f) => f.name.endsWith('.pkl'))

    //  Issue #1: Prevent duplicate .pkl files
    const uniqueIncoming = pklFiles.filter(
      (f) => !files.some((existing) => existing.name === f.name)
    )

    if (uniqueIncoming.length === 0 && pklFiles.length > 0) return

    const newFiles = uniqueIncoming.map((f) => {
      let path = ''
      try {
        path = window.api ? window.api.getPathForFile(f) : f.path || ''
      } catch (e) {
        path = f.path || ''
      }
      return {
        name: f.name,
        path,
        size: f.size,
        synced: false
      }
    })
    setFiles((prev) => [...prev, ...newFiles])
  }

  const onDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    handleFiles(e.dataTransfer.files)
  }

  const handleFileChange = (e) => {
    handleFiles(e.target.files)
    e.target.value = ''
  }

  const handleSync = async () => {
    if (files.length === 0 || syncing) return
    const unsyncedFiles = files.filter((f) => !f.synced)
    if (unsyncedFiles.length === 0) return

    setSyncing(true)
    const filePaths = unsyncedFiles.map((file) => file.path).filter(Boolean)

    try {
      const results = await window.electron.ipcRenderer.invoke('sync-artifacts', filePaths)
      setFiles((prev) =>
        prev.map((file) => {
          const result = results.find((r) => r.name === file.name)
          return result ? { ...file, synced: result.status === 'success' } : file
        })
      )
      if (onSync) await onSync()
    } catch (err) {
      console.error('Sync Failed', err)
    } finally {
      setSyncing(false)
    }
  }

  // 🛡️ Issue #2: Purge should reset syncing state immediately
  const handlePurge = () => {
    if (onPurge) {
      onPurge()
    } else {
      setFiles([])
      if (setSyncing) setSyncing(false)
    }
  }

  const removeFile = (index) => {
    setFiles((prev) => prev.filter((_, i) => i !== index))
  }

  const hasUnsynced = files.some((f) => !f.synced)

  return (
    <div className="bg-[#0d1117] border border-gray-800 rounded-lg p-4 flex flex-col h-auto overflow-hidden">
      <div className="mb-4 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2">
          <h2 className="text-[9px] font-black text-white tracking-[0.2em]">
            Calibration
          </h2>
          <div className="w-1 h-1 rounded-full bg-blue-500 animate-pulse" />
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleSync}
            disabled={syncing || !hasUnsynced}
            className={cn(
              'px-3 py-1.5 rounded text-[8px] font-black tracking-[0.2em] transition-all flex items-center gap-2',
              syncing || !hasUnsynced
                ? 'bg-gray-800 text-gray-600 cursor-not-allowed'
                : 'bg-blue-600 text-white hover:bg-blue-500 shadow-[0_0_15px_rgba(37,99,235,0.2)]'
            )}
          >
            {syncing ? (
              <div className="w-2.5 h-2.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            ) : (
              <>
                <RefreshCw size={10} />
                Sync
              </>
            )}
          </button>

          {files.length > 0 && (
            <button
              onClick={handlePurge}
              className="text-[9px] font-bold text-red-500/50 hover:text-red-500 transition-colors"
            >
              Purge All
            </button>
          )}
        </div>
      </div>

      <div className="flex-1 flex flex-col min-h-0 gap-4">
        <div
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
          className={cn(
            'relative rounded-lg border border-dashed border-slate-700 bg-slate-900/20 p-8 text-center transition-all hover:border-sky-500/50 hover:bg-sky-500/5 cursor-pointer group flex flex-col items-center justify-center shrink-0 min-h-[120px]',
            isDragging && 'border-sky-500 bg-sky-500/5'
          )}
        >
          <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-slate-800 text-slate-400 transition-colors group-hover:bg-sky-500/20 group-hover:text-sky-400">
            <Upload className="h-5 w-5" />
          </div>
          <p className="text-sm font-medium text-slate-300 pointer-events-none">Drop Model Files Here</p>
          <p className="mt-1 text-xs text-slate-500 pointer-events-none">Supports .pkl formats</p>
          <input
            type="file"
            multiple
            accept=".pkl"
            onChange={handleFileChange}
            className="absolute inset-0 opacity-0 cursor-pointer"
          />
        </div>

        <div className="flex-1 min-h-0 overflow-y-auto custom-scrollbar pr-2 max-h-[150px]">
          {files.length > 0 ? (
            <div className="grid grid-cols-2 gap-1.5">
              {files.map((file, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between p-1.5 bg-black/40 border border-gray-800 rounded group transition-all"
                >
                  <div className="flex flex-col min-w-0">
                    <div className="flex items-center gap-1">
                      <File
                        size={8}
                        className={cn('shrink-0', file.synced ? 'text-green-500' : 'text-blue-500')}
                      />
                      <span className="text-[7px] font-bold text-gray-300 truncate max-w-[70px]">
                        {file.name}
                      </span>
                    </div>
                    <span className="text-[8px] text-gray-600 font-mono ml-3">
                      {formatSize(file.size || 0)}
                    </span>
                  </div>
                  <div className="flex items-center gap-1 shrink-0">
                    {file.synced ? (
                      <CheckCircle size={10} className="text-green-500" />
                    ) : (
                      <div className="w-2 h-2 rounded-full border border-gray-700" />
                    )}
                    <button
                      onClick={() => removeFile(i)}
                      className="p-0.5 hover:bg-red-500/10 rounded transition-colors"
                    >
                      <XCircle
                        size={10}
                        className="text-red-500 opacity-0 group-hover:opacity-100"
                      />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="h-full flex items-center justify-center border border-gray-800/10 border-dashed rounded py-4">
              <p className="text-[7px] text-gray-800 font-bold tracking-widest opacity-20">
                No Models Calibrated
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default ArtifactPicker
