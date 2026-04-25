import { useState, useCallback } from 'react'
import { Upload, File, CheckCircle, XCircle, AlertCircle } from 'lucide-react'
import { cn } from '../lib/utils'

const ArtifactPicker = ({ onSync }) => {
  const [files, setFiles] = useState([])
  const [isDragging, setIsDragging] = useState(false)
  const [syncing, setSyncing] = useState(false)

  const onDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const onDragLeave = () => {
    setIsDragging(false)
  }

  const handleFiles = (incomingFiles) => {
    const pklFiles = Array.from(incomingFiles).filter(f => f.name.endsWith('.pkl'))
    const newFiles = pklFiles.map(f => {
      let path = ''
      try {
        path = window.api ? window.api.getPathForFile(f) : (f.path || '')
      } catch (e) {
        path = f.path || ''
      }
      return { name: f.name, path, synced: false }
    })
    setFiles(prev => [...prev, ...newFiles])
  }

  const onDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    handleFiles(e.dataTransfer.files)
  }

  const handleFileChange = (e) => {
    handleFiles(e.target.files)
    e.target.value = '' // Clear input so same file can be selected again
  }

  const handleSync = async () => {
    if (files.length === 0 || syncing) return
    const unsyncedFiles = files.filter(f => !f.synced)
    if (unsyncedFiles.length === 0) return

    setSyncing(true)
    const filePaths = unsyncedFiles.map(file => file.path).filter(Boolean)
    
    try {
      const results = await window.electron.ipcRenderer.invoke('sync-artifacts', filePaths)
      setFiles(prev => prev.map(file => {
        const result = results.find(r => r.name === file.name)
        return result ? { ...file, synced: result.status === 'success' } : file
      }))
      if (onSync) await onSync()
    } catch (err) {
      alert("Sync Failed: " + err.message)
    } finally {
      setSyncing(false)
    }
  }

  const removeFile = (index) => {
    setFiles(prev => prev.filter((_, i) => i !== index))
  }

  return (
    <div className="bg-[#0d1117] border border-gray-800 rounded-lg p-8 flex flex-col h-[480px]">
      <div className="mb-8 flex items-center justify-between shrink-0">
        <div>
          <h2 className="text-sm font-black tracking-tighter text-white uppercase italic">Artifact Calibration</h2>
          <p className="text-[8px] text-gray-500 uppercase tracking-widest font-bold">Inject .pkl models into forensic engine</p>
        </div>
        {files.length > 0 && (
          <button onClick={() => setFiles([])} className="text-[8px] font-bold text-red-500 uppercase tracking-widest hover:underline">
            Purge Queue
          </button>
        )}
      </div>

      <div className="flex-1 flex flex-col min-h-0 gap-6">
        <div
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
          className={cn(
            "relative border border-dashed rounded-lg p-12 transition-all duration-200 flex flex-col items-center justify-center gap-4 shrink-0",
            isDragging 
              ? "border-blue-500 bg-blue-500/5 scale-[0.99]" 
              : "border-gray-800 bg-black/40 hover:border-gray-700"
          )}
        >
          <div className="w-12 h-12 rounded-full bg-[#161b22] flex items-center justify-center text-blue-500 border border-gray-800 shadow-inner">
            <Upload size={20} />
          </div>
          <div className="text-center">
            <p className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">
              Drop <span className="text-blue-500">.pkl</span> Artifacts
            </p>
            <p className="text-[8px] text-gray-600 mt-1 uppercase tracking-tighter">Support for GNN, XGB, RF, SVM, LR</p>
          </div>
          <input type="file" multiple accept=".pkl" onChange={handleFileChange} className="absolute inset-0 opacity-0 cursor-pointer" />
        </div>

        <div className="flex-1 min-h-0 flex flex-col gap-4">
          {files.length > 0 ? (
            <>
              <div className="flex-1 overflow-y-auto custom-scrollbar pr-2 space-y-2">
                {files.map((file, i) => (
                  <div key={i} className="flex items-center justify-between p-3 bg-black/40 border border-gray-800 rounded-md group hover:border-gray-600 transition-all">
                    <div className="flex items-center gap-3">
                      <File size={14} className={cn("transition-colors", file.synced ? "text-green-500" : "text-blue-500")} />
                      <span className="text-[10px] font-bold text-gray-300 truncate">{file.name}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      {file.synced ? (
                        <CheckCircle size={14} className="text-green-500 animate-in zoom-in duration-300" />
                      ) : (
                        <div className="w-3.5 h-3.5 rounded-full border border-gray-700" />
                      )}
                      <button onClick={() => removeFile(i)} className="p-1 hover:bg-red-500/10 rounded transition-colors">
                        <XCircle size={14} className="text-red-500 opacity-0 group-hover:opacity-100 transition-opacity" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
              {files.some(f => !f.synced) ? (
                <button 
                  onClick={handleSync}
                  disabled={syncing}
                  className="w-fit px-8 py-2.5 bg-blue-600 hover:bg-blue-700 text-white text-[9px] font-black uppercase tracking-[0.2em] rounded-md transition-all shadow-lg active:scale-95 shrink-0 flex items-center gap-2 disabled:opacity-50"
                >
                  {syncing ? "Synchronizing..." : "Synchronize Engine"}
                </button>
              ) : (
                <div className="w-fit px-8 py-2.5 bg-green-600/20 border border-green-500/30 text-green-500 text-[9px] font-black uppercase tracking-[0.2em] rounded-md shrink-0 flex items-center gap-2">
                  <CheckCircle size={14} />
                  Engine Synced
                </div>
              )}
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center border border-gray-800/50 rounded-lg bg-black/20 border-dashed">
              <div className="flex items-center gap-3 opacity-30">
                <AlertCircle size={14} className="text-yellow-500" />
                <p className="text-[9px] text-yellow-500 font-bold uppercase tracking-widest">
                  Waiting for model weights...
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default ArtifactPicker
