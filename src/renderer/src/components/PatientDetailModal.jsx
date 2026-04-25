import { X, FileText, Activity, ShieldAlert, Fingerprint, Calendar } from 'lucide-react'
import { cn } from '../lib/utils'

const PatientDetailModal = ({ patient, onClose }) => {
  if (!patient) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-300">
      <div className="bg-[#0d1117] border border-gray-800 w-full max-w-2xl rounded-lg shadow-2xl overflow-hidden animate-in zoom-in-95 duration-300">
        {/* Header */}
        <div className="bg-black/50 p-6 border-b border-gray-800 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className={cn(
              "p-3 rounded-lg border",
              patient.score > 0.5 ? "bg-red-500/10 border-red-500/30 text-red-500" : "bg-green-500/10 border-green-500/30 text-green-500"
            )}>
              <Fingerprint size={24} />
            </div>
            <div>
              <h3 className="text-xl font-black uppercase italic tracking-tight text-white">{patient.id}</h3>
              <p className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">Forensic Clinical Profile</p>
            </div>
          </div>
          <button onClick={onClose} className="p-2 hover:bg-white/5 rounded-full transition-colors text-gray-500 hover:text-white">
            <X size={20} />
          </button>
        </div>

        {/* Body */}
        <div className="p-8 space-y-8 max-h-[70vh] overflow-y-auto custom-scrollbar">
          {/* Summary Cards */}
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-black/40 border border-gray-800 p-4 rounded-lg">
              <span className="text-[8px] font-black uppercase text-gray-600 block mb-1">Neural Risk Score</span>
              <span className={cn("text-2xl font-mono font-black", patient.score > 0.5 ? "text-red-500" : "text-green-500")}>
                {patient.score}
              </span>
            </div>
            <div className="bg-black/40 border border-gray-800 p-4 rounded-lg">
              <span className="text-[8px] font-black uppercase text-gray-600 block mb-1">Clinical Status</span>
              <span className={cn(
                "text-lg font-black uppercase italic",
                patient.status === 'Urgent' ? "text-red-500" : 
                patient.status === 'Critical' ? "text-orange-500" :
                "text-green-500"
              )}>
                {patient.status}
              </span>
            </div>
          </div>

          {/* Biomarkers */}
          <div className="space-y-4">
            <h4 className="text-[10px] font-black uppercase text-blue-500 border-b border-gray-800 pb-2 flex items-center gap-2">
              <Activity size={12} />
              Biometric Analysis
            </h4>
            <div className="grid grid-cols-1 gap-3">
              {patient.details && Object.entries(patient.details).map(([key, val]) => (
                <div key={key} className="flex items-center justify-between p-3 bg-white/5 rounded border border-gray-800/50">
                  <span className="text-[10px] font-bold text-gray-400 uppercase tracking-wider">{key}</span>
                  <span className="text-[10px] font-mono font-bold text-white">{val}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Forensic Notes */}
          <div className="space-y-4">
            <h4 className="text-[10px] font-black uppercase text-blue-500 border-b border-gray-800 pb-2 flex items-center gap-2">
              <FileText size={12} />
              Forensic Intelligence
            </h4>
            <div className="p-4 bg-blue-500/5 border border-blue-500/20 rounded-lg">
              <p className="text-[11px] text-gray-300 leading-relaxed italic">
                Subject {patient.id} exhibits a {patient.status.toLowerCase()} risk profile. 
                The neural ensemble identifies specific markers in the {patient.details['Forensic Cluster']} cluster 
                contributing to a certainty level of {patient.details['Neural Certainty']}. 
                Follow-up diagnostic imaging is {patient.score > 0.6 ? "strongly advised" : "recommended per standard protocol"}.
              </p>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-6 bg-black/50 border-t border-gray-800 flex justify-end gap-3">
          <button 
            onClick={onClose}
            className="px-6 py-2 text-[10px] font-black uppercase tracking-widest text-gray-400 hover:text-white transition-colors"
          >
            Dismiss Report
          </button>
          <button 
            className="px-6 py-2 bg-blue-600 hover:bg-blue-500 text-white text-[10px] font-black uppercase tracking-widest rounded transition-all shadow-[0_0_15px_rgba(37,99,235,0.3)]"
          >
            Export Clinical File
          </button>
        </div>
      </div>
    </div>
  )
}

export default PatientDetailModal
