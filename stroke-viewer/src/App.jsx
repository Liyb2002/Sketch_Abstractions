// stroke-viewer/src/app.jsx
import { Suspense, useEffect, useMemo, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Line, GizmoHelper, GizmoViewport } from "@react-three/drei";
import * as THREE from "three";

async function loadFromAPI(url = "/api/strokes") {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load ${url}: ${res.status}`);
  return await res.json();
}

function PolylineGroup({ polylines, color = "#111827", lineWidth = 1.6 }) {
  if (!polylines?.length) return null;
  return (
    <group>
      {polylines.map((pl, i) =>
        pl.points?.length >= 2 ? (
          <Line key={i} points={pl.points} color={color} lineWidth={lineWidth} />
        ) : null
      )}
    </group>
  );
}

export default function App() {
  const [data, setData] = useState({
    perturbed_feature_lines: [],
    perturbed_construction_lines: [],
    feature_lines: [],
  });

  useEffect(() => {
    loadFromAPI().then(setData).catch(console.error);
  }, []);

  // Z-up: rotate world (Three is Y-up)
  const zUpRotation = useMemo(() => new THREE.Euler(-Math.PI / 2, 0, 0), []);

  // Simple camera fit after data loads
  useEffect(() => {
    const allPts = [
      ...data.perturbed_feature_lines.flatMap(pl => pl.points || []),
      ...data.perturbed_construction_lines.flatMap(pl => pl.points || []),
      ...data.feature_lines.flatMap(pl => pl.points || []),
    ];
    if (!allPts.length) return;
    const cam = window.__r3f?.store.getState().camera;
    if (!cam) return;
    const box = new THREE.Box3();
    allPts.forEach(p => box.expandByPoint(new THREE.Vector3(p[0], p[1], p[2])));
    const size = new THREE.Vector3(), center = new THREE.Vector3();
    box.getSize(size); box.getCenter(center);
    const maxSize = Math.max(size.x, size.y, size.z) || 1;
    const dist = maxSize * 1.8 / Math.tan((cam.fov * Math.PI) / 360);
    cam.position.copy(center.clone().add(new THREE.Vector3(1,1,1).normalize().multiplyScalar(dist)));
    cam.near = Math.max(0.001, maxSize / 1000);
    cam.far = Math.max(1000, maxSize * 1000);
    cam.lookAt(center);
    cam.updateProjectionMatrix();
  }, [data]);

  return (
    <div style={{ width: "100vw", height: "100vh", background: "#f8fafc" }}>
      <Canvas camera={{ fov: 45 }}>
        <color attach="background" args={["#f8fafc"]} />
        <Suspense fallback={null}>
          <ambientLight intensity={0.7} />
          <directionalLight position={[5, 5, 10]} intensity={0.7} />
          <directionalLight position={[-5, -3, -8]} intensity={0.2} />
          <group rotation={zUpRotation}>
            <axesHelper args={[2]} />
            {/* focus on perturbed_feature_lines for now */}
            <PolylineGroup polylines={data.perturbed_feature_lines} color="#111827" lineWidth={1.6} />
          </group>
          <GizmoHelper alignment="bottom-right" margin={[80, 80]}>
            <GizmoViewport axisColors={["#EF4444", "#10B981", "#3B82F6"]} labelColor="black" />
          </GizmoHelper>
        </Suspense>
        <OrbitControls makeDefault enablePan enableRotate enableZoom />
      </Canvas>
    </div>
  );
}
