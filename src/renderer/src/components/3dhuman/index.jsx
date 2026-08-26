import React, { Suspense, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import AnatomyScene from './AnatomyScene';
import * as THREE from 'three';
import { useAnatomyStore } from './useAnatomyStore';
import { Eye, EyeOff, Activity, Layers, ChevronDown, ChevronRight, Box } from 'lucide-react';

export default function HumanAnatomyWorkspace({ onPredict, loading }) {
  const { 
    layers, toggleLayer, viewMode, setViewMode: switchMode, 
    showTumor, currentRiskScore, primaryDriver 
  } = useAnatomyStore();

  const [modelsOpen, setModelsOpen] = useState(true);
  const [layersOpen, setLayersOpen] = useState(true);

  return (
    <div className="w-full h-full flex gap-0 min-h-0 bg-[#020408]">
      
      {/* CENTER: MASTER VIEWPORT (Full Width/Height, NO OVERLAYS) */}
      <div className="flex-1 h-full overflow-hidden bg-[#05080f] min-w-0 relative">
        {/* HUD: Top-Left Score */}
        <div className="absolute top-6 left-6 z-10 pointer-events-none">
          <div className="flex flex-col gap-1">
            <span className="text-[10px] text-gray-500 font-bold tracking-widest uppercase">Global Risk Assessment</span>
            <span className="text-4xl font-black text-white drop-shadow-md">
              {currentRiskScore ? (currentRiskScore * 100).toFixed(1) : "0.0"}%
            </span>
            <span className="text-xs font-mono text-red-400 font-bold tracking-wider uppercase">
              Primary Biomarker: {primaryDriver || "None"}
            </span>
          </div>
        </div>
        
        <Canvas shadows dpr={[1, 2]} gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping }}>
          <Suspense fallback={null}>
            <AnatomyScene previewMode={viewMode} isMain={true} />
          </Suspense>
        </Canvas>
      </div>

      {/* RIGHT: PROPERTIES PANEL */}
      <div className="w-72 flex flex-col gap-4 shrink-0 overflow-y-auto pr-2 custom-scrollbar h-full text-slate-200">
        
        {/* Patient Card */}
        <div className="bg-slate-900/50 border border-slate-800 p-4 rounded-xl backdrop-blur-sm shrink-0">
          <div className="flex justify-between items-start mb-4">
            <div className="w-full">
              <div className="text-[10px] text-sky-500 font-mono mb-2">TARGET SUBJECT ID</div>
              <form 
                onSubmit={(e) => {
                  e.preventDefault();
                  const rawVal = e.target.elements.sampleId.value.trim();
                  if (rawVal) {
                    // Extract only numbers from the input
                    const matches = rawVal.match(/\d+/g);
                    if (matches) {
                      const numStr = matches.join('');
                      onPredict(`Sample_${numStr.padStart(4, '0')}`);
                    } else {
                      // If they typed something entirely non-numeric, fallback to random
                      onPredict();
                    }
                  } else {
                    // Fall back to random if empty
                    onPredict();
                  }
                }}
                className="flex gap-2 w-full"
              >
                <div className="relative w-full">
                  <span className="absolute left-2 top-1/2 -translate-y-1/2 text-slate-500 font-mono text-sm pointer-events-none">Sample_</span>
                  <input 
                    type="text" 
                    name="sampleId"
                    placeholder="0001"
                    pattern="[0-9]*"
                    className="bg-slate-950 border border-slate-800 rounded pl-16 pr-2 py-1.5 text-sm text-white w-full focus:outline-none focus:border-sky-500 font-mono"
                  />
                </div>
                <button 
                  type="submit"
                  disabled={loading}
                  className="bg-blue-600 hover:bg-blue-500 text-white rounded px-3 flex items-center justify-center transition-colors shrink-0 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? (
                    <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  ) : (
                    <Activity size={14} />
                  )}
                </button>
              </form>
            </div>
          </div>
          <div className="space-y-2">
            <div className="flex justify-between text-xs border-b border-slate-800 pb-2">
              <span className="text-slate-400">Target:</span>
              <span className="text-white font-medium">Pelvis</span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-slate-400">Risk:</span>
              <span className={`${currentRiskScore > 0.5 ? 'text-red-400' : 'text-green-400'} font-medium`}>
                {currentRiskScore ? (currentRiskScore * 100).toFixed(1) + '%' : '---'}
              </span>
            </div>
          </div>
        </div>

        {/* SECTION: 3D MODELS */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-xl shrink-0 overflow-hidden">
          <button 
            onClick={() => setModelsOpen(!modelsOpen)}
            className="w-full p-4 flex items-center justify-between hover:bg-slate-800/50 transition-colors"
          >
            <div className="flex items-center gap-2 text-sky-400 font-bold text-xs uppercase tracking-wider">
              <Box size={14} /> 3D Models
            </div>
            {modelsOpen ? <ChevronDown size={16} className="text-slate-400" /> : <ChevronRight size={16} className="text-slate-400" />}
          </button>
          
          {modelsOpen && (
            <div className="p-4 pt-2 space-y-4">
              {/* Full Anatomy Thumbnail */}
              <div 
                onClick={() => switchMode('overview')}
                className={`w-full h-32 rounded-lg overflow-hidden cursor-pointer border transition-all ${viewMode === 'overview' ? 'border-sky-500/70 shadow-[0_0_10px_rgba(14,165,233,0.2)]' : 'border-slate-800/60 hover:border-slate-600/80'}`}
              >
                <div className="bg-black/60 w-full h-full relative group">
                  <div className="absolute top-1.5 left-1.5 z-10 bg-slate-900/60 px-1.5 py-0.5 rounded text-[9px] font-bold text-slate-400">Anatomy</div>
                  <Canvas dpr={[1, 1]} gl={{ antialias: false }}>
                    <Suspense fallback={null}>
                      <AnatomyScene previewMode="overview" />
                    </Suspense>
                  </Canvas>
                </div>
              </div>

              {/* Heart Thumbnail */}
              <div 
                onClick={() => switchMode('heart')}
                className={`w-full h-32 rounded-lg overflow-hidden cursor-pointer border transition-all ${viewMode === 'heart' ? 'border-sky-500/70 shadow-[0_0_10px_rgba(14,165,233,0.2)]' : 'border-slate-800/60 hover:border-slate-600/80'}`}
              >
                <div className="bg-black/60 w-full h-full relative group">
                  <div className="absolute top-1.5 left-1.5 z-10 bg-slate-900/60 px-1.5 py-0.5 rounded text-[9px] font-bold text-slate-400">Heart</div>
                  <Canvas dpr={[1, 1]} gl={{ antialias: false }}>
                    <Suspense fallback={null}>
                      <AnatomyScene previewMode="heart" />
                    </Suspense>
                  </Canvas>
                </div>
              </div>

              {/* Prostate Thumbnail */}
              <div 
                onClick={() => switchMode('focus')}
                className={`w-full h-32 rounded-lg overflow-hidden cursor-pointer border transition-all ${viewMode === 'focus' ? 'border-sky-500/70 shadow-[0_0_10px_rgba(14,165,233,0.2)]' : 'border-slate-800/60 hover:border-slate-600/80'}`}
              >
                <div className="bg-black/60 w-full h-full relative group">
                  <div className="absolute top-1.5 left-1.5 z-10 bg-slate-900/60 px-1.5 py-0.5 rounded text-[9px] font-bold text-slate-400">Prostate</div>
                  <Canvas dpr={[1, 1]} gl={{ antialias: false }}>
                    <Suspense fallback={null}>
                      <AnatomyScene previewMode="focus" />
                    </Suspense>
                  </Canvas>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* SECTION: ANATOMY LAYERS */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-xl shrink-0 overflow-hidden">
          <button 
            onClick={() => setLayersOpen(!layersOpen)}
            className="w-full p-4 flex items-center justify-between hover:bg-slate-800/50 transition-colors"
          >
            <div className="flex items-center gap-2 text-sky-400 font-bold text-xs uppercase tracking-wider">
              <Layers size={14} /> Anatomy Layers
            </div>
            {layersOpen ? <ChevronDown size={16} className="text-slate-400" /> : <ChevronRight size={16} className="text-slate-400" />}
          </button>
          
          {layersOpen && (
            <div className="p-4 pt-2 space-y-2">
              {Object.entries(layers).map(([key, active]) => (
                <button key={key} onClick={() => toggleLayer(key)} className={`w-full flex items-center justify-between p-2.5 rounded-lg border transition-all ${active ? 'bg-sky-950/30 border-sky-500/30 text-sky-100' : 'bg-slate-950/30 border-transparent text-slate-500 hover:bg-slate-900'}`}>
                  <span className="text-xs font-medium capitalize">{key}</span>
                  {active ? <Eye size={14} className="text-sky-400" /> : <EyeOff size={14} />}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Tumor Overlay */}
        <div className="bg-red-950/10 border border-red-900/30 p-4 rounded-xl shrink-0 mb-4">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2 text-red-400 font-bold text-xs uppercase">
              <Activity size={14} /> Tumor Overlay
            </div>
            <input type="checkbox" checked={showTumor} onChange={(e) => useAnatomyStore.setState({ showTumor: e.target.checked })} className="accent-red-500" />
          </div>
        </div>

      </div>
    </div>
  );
}
