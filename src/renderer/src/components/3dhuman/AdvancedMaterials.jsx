import React, { useMemo } from 'react';
import * as THREE from 'three';

// 1. Fleshy Organ Material (Subsurface Scattering simulation)
export const OrganMaterial = ({ color, emissive = '#000' }) => {
  return (
    <meshStandardMaterial
      color={color}
      roughness={0.3}
      metalness={0.1}
      emissive={emissive}
      emissiveIntensity={0.2}
      // Simulates light penetrating soft tissue
      transmission={0.1} 
      thickness={1.5}
    />
  );
};

// 2. High-Tech Holographic Skin
export const IntegumentaryMaterial = () => {
  return (
    <meshPhysicalMaterial
      color="#38bdf8"
      transparent
      opacity={0.15}
      roughness={0.1}
      metalness={0.8}
      clearcoat={1}
      clearcoatRoughness={0.1}
      side={THREE.DoubleSide}
    />
  );
};

// 3. Glowing Tumor / Pathology
export const TumorMaterial = () => {
  return (
    <meshStandardMaterial
      color="#ef4444"
      emissive="#ff0000"
      emissiveIntensity={2.5} // High intensity for Bloom effect
      roughness={0.2}
      metalness={0.5}
    />
  );
};

// 4. Clinical Bone Material
export const BoneMaterial = () => {
  return (
    <meshStandardMaterial
      color="#e2e8f0"
      roughness={0.6}
      metalness={0.1}
    />
  );
};
