import React from 'react';
import { Float } from '@react-three/drei';
import { BoneMaterial, IntegumentaryMaterial } from './AdvancedMaterials';
import { useAnatomyStore } from './useAnatomyStore';

export default function ProceduralLimb({ position, rotation, type = 'arm' }) {
  const { layers } = useAnatomyStore();

  return (
    <group position={position} rotation={rotation}>
      {/* Skeletal Structure */}
      {layers.skeletal && (
        <group>
          {/* Humerus */}
          <mesh position={[0, -0.6, 0]}>
            <cylinderGeometry args={[0.08, 0.07, 1.2, 16]} />
            <BoneMaterial />
          </mesh>
          {/* Elbow Joint */}
          <mesh position={[0, -1.2, 0]}>
            <sphereGeometry args={[0.09, 16, 16]} />
            <BoneMaterial />
          </mesh>
          {/* Radius/Ulna */}
          <mesh position={[0, -1.9, 0]}>
            <cylinderGeometry args={[0.06, 0.05, 1.2, 16]} />
            <BoneMaterial />
          </mesh>
        </group>
      )}

      {/* Translucent Skin Shell */}
      {layers.integumentary && (
        <Float speed={2} rotationIntensity={0.2} floatIntensity={0.5}>
          <mesh scale={[1.6, 1.6, 1.6]}>
            <capsuleGeometry args={[0.45, 2.8, 16, 32]} />
            <IntegumentaryMaterial />
          </mesh>
        </Float>
      )}
    </group>
  );
}
