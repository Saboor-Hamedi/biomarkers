import { create } from 'zustand';

export const useAnatomyStore = create((set) => ({
  // View Modes: 'overview' | 'focus'
  viewMode: 'overview',
  setViewMode: (mode) => set({ viewMode: mode }),

  // Layer Visibility
  layers: {
    integumentary: true,
    skeletal: true,
    vascular: false,
    organs: true,
    prostate: true,
  },
  
  // Selected 3D Model
  activeModelPath: './human_anatomy.glb',
  setActiveModelPath: (path) => set({ activeModelPath: path }),

  toggleLayer: (layer) => set((state) => ({
    layers: { ...state.layers, [layer]: !state.layers[layer] }
  })),

  // Advanced Features
  limbs: ['left_arm', 'right_arm'], // Procedural limbs
  addLimb: (limbId) => set((state) => ({ 
    limbs: [...state.limbs, limbId] 
  })),
  
  // Pathology
  showTumor: false,
  clipPlaneY: 2.5,

  // ML Integration
  currentRiskScore: 0,
  primaryDriver: '',
}));
