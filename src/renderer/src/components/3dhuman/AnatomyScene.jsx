import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { useGLTF, Html, Environment, ContactShadows, OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { useAnatomyStore } from './useAnatomyStore';

// --- MATERIALS as React Components ---
const BoneMaterial = () => <meshStandardMaterial color="#e2e8f0" roughness={0.4} metalness={0.1} />;
const OrganMaterial = () => <meshStandardMaterial color="#be123c" roughness={0.3} metalness={0.1} />;
const TumorMaterial = ({ intensity = 2 }) => <meshStandardMaterial color="#ef4444" emissive="#ff0000" emissiveIntensity={intensity} />;
const SkinMaterial = () => (
  <meshPhysicalMaterial 
    color="#38bdf8" transparent opacity={0.15} 
    roughness={0.1} metalness={0.8} clearcoat={1} 
  />
);

export default function AnatomyScene({ previewMode, isMain = false }) {
  const { layers, showTumor, currentRiskScore, primaryDriver } = useAnatomyStore();
  const groupRef = useRef();
  
  // Calculate dynamic properties based on ML score
  const tumorRadius = Math.max(0.01, currentRiskScore * 0.05);
  const tumorGlow = Math.max(0.5, currentRiskScore * 10);
  
  // Load Models using relative paths for Electron production (file:// protocol)
  const skeleton = useGLTF('./human_anatomy.glb', true); 
  const prostateModel = useGLTF('./prostate_cancer.glb', true);
  const heartModel = useGLTF('./realistic_human_heart.glb', true);

  // Keep it perfectly centered, no weird lerping off-screen
  useFrame(() => {
    if (!groupRef.current) return;
    
    // Smoothly adjust scale based on model, but keep position 0,0,0
    let targetScale = 1;
    if (previewMode === 'overview') targetScale = 1.0; // Let's scale up the human directly below
    if (previewMode === 'focus') targetScale = 0.4; // Prostate was too close (too big), reduce scale significantly
    if (previewMode === 'heart') targetScale = 1.2;

    groupRef.current.scale.set(targetScale, targetScale, targetScale);
  });

  return (
    <>
      {/* NO position offset on the group, keep it at 0,0,0 so OrbitControls zooms cleanly into it */}
      <group ref={groupRef} position={[0, 0, 0]}>
        
        {/* --- FULL ANATOMY MODE --- */}
        {(previewMode === 'overview') && (
          <group visible={layers.skeletal}>
             {skeleton.scene ? (
               <primitive object={skeleton.scene.clone()} scale={isMain ? 0.05 : 0.03} position={[0, 0, 0]} /> 
             ) : (
               <mesh position={[0,0,0]}>
                  <capsuleGeometry args={[0.5, 3, 4, 16]} />
                  <BoneMaterial />
               </mesh>
             )}
          </group>
        )}

        {/* --- FOCAL PATHOLOGY MODE (Prostate) --- */}
        {previewMode === 'focus' && (
          <group position={[0, 0, 0]}>
            {prostateModel.scene ? (
              <primitive object={prostateModel.scene.clone()} scale={1}>
                <OrganMaterial />
              </primitive>
            ) : (
              <mesh>
                <sphereGeometry args={[0.5, 32, 32]} />
                <OrganMaterial />
              </mesh>
            )}

            {/* Tumor Overlay */}
            {showTumor && (
              <mesh position={[0.6, 0.3, 0.4]}>
                <sphereGeometry args={[tumorRadius * 3, 24, 24]} />
                <TumorMaterial intensity={tumorGlow} />
                {isMain && (
                  <Html position={[0.2, 0.2, 0]} center distanceFactor={4} zIndexRange={[100, 0]}>
                    <div className="bg-red-950/90 border border-red-500/50 px-3 py-2 rounded text-xs whitespace-nowrap shadow-[0_0_15px_rgba(239,68,68,0.3)] animate-pulse flex flex-col items-center">
                      <span className="font-black tracking-widest text-red-400 text-[10px]">DETECTED NEOPLASM</span>
                      <div className="w-full h-px bg-red-500/30 my-1.5" />
                      <span className="text-[9px] text-red-300 font-mono">Malignancy Prob: {(currentRiskScore * 100).toFixed(1)}%</span>
                      <span className="text-[9px] text-red-400/80 font-mono">Correlated Biomarker: {primaryDriver}</span>
                    </div>
                  </Html>
                )}
              </mesh>
            )}
            
            {/* Holographic Shell for Context */}
            {layers.integumentary && (
               <mesh scale={[1.0, 1.0, 1.0]}>
                 <sphereGeometry args={[0.6, 32, 32]} />
                 <SkinMaterial />
               </mesh>
            )}
          </group>
        )}

        {/* --- REALISTIC HEART MODE --- */}
        {previewMode === 'heart' && (
          <group position={[0, 0, 0]}>
            {heartModel.scene ? (
              <primitive object={heartModel.scene.clone()} scale={0.8} />
            ) : (
              <mesh>
                <sphereGeometry args={[0.5, 32, 32]} />
                <OrganMaterial />
              </mesh>
            )}
            
            {isMain && (
              <Html position={[0.3, 0.3, 0]} center distanceFactor={2}>
                <div className="bg-sky-950/90 border border-sky-500 text-sky-400 px-3 py-1 rounded text-xs font-bold whitespace-nowrap shadow-lg">
                  Left Ventricle
                </div>
              </Html>
            )}
          </group>
        )}
      </group>

      {/* --- LIGHTING & ENVIRONMENT --- */}
      <ambientLight intensity={0.5} color="#cbd5e1" />
      <spotLight position={[5, 10, 5]} angle={0.5} penumbra={1} intensity={isMain ? 2 : 1} castShadow={isMain} color="#38bdf8" />
      <pointLight position={[-5, 2, -5]} intensity={1} color="#3b82f6" />
      
      {isMain && <ContactShadows resolution={1024} scale={10} blur={2} opacity={0.5} far={10} color="#000000" />}
      <Environment preset="city" background={false} />

      <OrbitControls 
        makeDefault={isMain}
        minDistance={0.1} 
        maxDistance={100} 
        dampingFactor={0.05} 
        enableDamping 
      />
    </>
  );
}
