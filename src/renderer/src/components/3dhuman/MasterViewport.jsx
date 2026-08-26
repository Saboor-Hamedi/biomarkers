import React, { useRef, Suspense } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Environment, ContactShadows, useGLTF } from '@react-three/drei';
import { EffectComposer, Bloom, Vignette } from '@react-three/postprocessing';
import * as THREE from 'three';
import { useAnatomyStore } from './useAnatomyStore';

function RealisticModel() {
  const { viewMode, activeModelPath } = useAnatomyStore();
  const groupRef = useRef();

  // Load the selected 3D model
  // Vite usually serves files in renderer/public at the root level, but we can also import them.
  // Assuming the .glb files are in the public folder, useGLTF handles loading them.
  const { scene } = useGLTF(activeModelPath);

  // Animation for smooth zoom (optional, adjust based on model scale)
  useFrame((state, delta) => {
    if (groupRef.current) {
      const targetScale = viewMode === 'focus' ? 1.5 : 1;
      const targetY = viewMode === 'focus' ? -0.5 : 0;
      groupRef.current.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), delta * 2);
      groupRef.current.position.y = THREE.MathUtils.lerp(groupRef.current.position.y, targetY, delta * 2);
    }
  });

  return (
    <group ref={groupRef}>
      <primitive object={scene} />
    </group>
  );
}

export default function MasterViewport() {
  return (
    <div className="w-full h-full bg-gradient-to-b from-[#05080f] to-[#020408]">
      <Canvas shadows dpr={[1, 2]} gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping }}>
        <PerspectiveCamera makeDefault position={[0, 0, 6]} fov={45} />
        
        {/* Lighting setup specifically for realistic models */}
        <ambientLight intensity={1.5} color="#ffffff" />
        <spotLight position={[10, 10, 10]} angle={0.5} penumbra={1} intensity={2} castShadow />
        <spotLight position={[-10, 10, -10]} angle={0.5} penumbra={1} intensity={1} color="#38bdf8" />
        <pointLight position={[0, -5, 5]} intensity={1} color="#3b82f6" />
        
        <Suspense fallback={null}>
          <RealisticModel />
        </Suspense>
        
        <ContactShadows resolution={1024} scale={10} blur={2} opacity={0.5} far={10} color="#000000" />
        <Environment preset="city" background={false} />

        <EffectComposer disableNormalPass>
          <Bloom luminanceThreshold={1.2} mipmapBlur intensity={1.0} radius={0.5} />
          <Vignette eskil={false} offset={0.1} darkness={1.1} />
        </EffectComposer>

        <OrbitControls minDistance={1} maxDistance={20} dampingFactor={0.05} enableDamping />
      </Canvas>
      
      {/* Overlay Text */}
      <div className="absolute top-6 left-6 pointer-events-none">
        <h2 className="text-sky-400 font-bold tracking-widest text-sm uppercase">Master Viewport</h2>
        <p className="text-slate-500 text-xs mt-1">Realistic GLTF Render</p>
      </div>
    </div>
  );
}
