import React, { useState, useEffect } from 'react';
import { Settings as SettingsIcon, X, Key, ShieldCheck, Save } from 'lucide-react';
import { getSetting, saveSetting } from '../lib/settings';
import { cn } from '../lib/utils';

const SettingsModal = ({ isOpen, onClose }) => {
  const [apiKey, setApiKey] = useState('');
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    if (isOpen) {
      setApiKey(getSetting('DEEPSEEK_API_KEY', ''));
      setSaved(false);
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handleSave = (e) => {
    e.preventDefault();
    saveSetting('DEEPSEEK_API_KEY', apiKey);
    setSaved(true);
    setTimeout(() => {
      setSaved(false);
      onClose();
    }, 1500);
  };

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center animate-in fade-in duration-200">
      <div className="bg-[#0d1117] border border-gray-800 rounded-lg shadow-2xl w-full max-w-md animate-in zoom-in-95 duration-200">
        
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-800">
          <div className="flex items-center gap-2">
            <SettingsIcon size={18} className="text-gray-400" />
            <h2 className="text-sm font-black uppercase tracking-widest text-white">System Settings</h2>
          </div>
          <button onClick={onClose} className="text-gray-500 hover:text-white transition-colors">
            <X size={18} />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <Key size={16} className="text-blue-500" />
              <h3 className="text-xs font-bold text-gray-300 uppercase tracking-wider">DeepSeek API Configuration</h3>
            </div>
            
            <p className="text-[10px] text-gray-500 leading-relaxed uppercase font-bold">
              Securely bind your DeepSeek AI access token. This key is encrypted and strictly stored locally on your machine. It is required to power the Forensic Co-Pilot.
            </p>

            <form onSubmit={handleSave} className="space-y-4">
              <div>
                <label className="block text-[10px] font-bold text-gray-400 uppercase tracking-wider mb-2">
                  Access Token
                </label>
                <input
                  type="password"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="sk-..."
                  className="w-full bg-black border border-gray-800 rounded text-xs text-white px-3 py-2 focus:outline-none focus:border-blue-500 transition-colors placeholder:text-gray-700 font-mono"
                />
              </div>

              <button 
                type="submit"
                className={cn(
                  "w-full flex items-center justify-center gap-2 py-2.5 rounded text-xs font-black uppercase tracking-widest transition-all duration-300",
                  saved ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/50" : "bg-blue-600 hover:bg-blue-500 text-white shadow-[0_0_15px_rgba(37,99,235,0.4)]"
                )}
              >
                {saved ? (
                  <>
                    <ShieldCheck size={14} />
                    Secured
                  </>
                ) : (
                  <>
                    <Save size={14} />
                    Save Configuration
                  </>
                )}
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SettingsModal;
