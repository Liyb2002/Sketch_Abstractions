// stroke-viewer/src/app.jsx
import { Suspense, useEffect, useMemo, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Line, GizmoHelper, GizmoViewport } from "@react-three/drei";
import * as THREE from "three";

// --------- existing strokes fetch ----------
async function loadFromAPI(url = "/api/strokes") {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load ${url}: ${res.status}`);
  return await res.json();
}

// --------- simple HUD to execute the program ----------
function ProgramLoader({ onCuboids }) {
  // This is the line you asked about — it's here:
  const runDefault = async () => {
    const res = await fetch(`/api/execute-default?use_offsets=true&use_scales=false`, {
      method: "POST",
    });
    if (!res.ok) {
      console.error("execute-default failed", res.status);
      return;
    }
    const json = await res.json(); // { cuboids: [...] }
    onCuboids(json.cuboids || []);
  };

  return (
    <div style={{
      position: "absolute", zIndex: 10, left: 12, top: 12,
      background: "#fff", border: "1px solid #e5e7eb",
      borderRadius: 8, padding: 8
    }}>
      <div style={{ fontSize: 12, marginBottom: 6 }}>Program</div>
      <button onClick={runDefault} style={{ fontSize: 12, padding: "4px 8px" }}>
        Run default (offsets on)
      </button>
    </div>
  );
}

// --------- strokes renderer (unchanged) ----------
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

// --------- cuboids renderer (center + size, axis-aligned) ----------
function CuboidGroup({ cuboids }) {
  if (!cuboids?.length) return null;
  return (
    <group>
      {cuboids.map((c) => {
        const rot = c.rotationEuler || [0, 0, 0]; // None from server for now
        return (
          <group key={c.id} position={c.center} rotation={rot}>
            <mesh>
              <boxGeometry args={c.size} />
              <meshBasicMaterial wireframe />
            </mesh>
          </group>
        );
      })}
    </group>
  );
}

// --------- helpers to compute a side-by-side offset ----------
function bboxOfPolylines(polys) {
  const box = new THREE.Box3();
  for (const pl of polys) for (const p of pl.points ?? []) {
    box.expandByPoint(new THREE.Vector3(p[0], p[1], p[2]));
  }
  return box;
}
function bboxOfCuboids(cuboids) {
  const box = new THREE.Box3();
  const tmp = new THREE.Box3();
  for (const c of cuboids) {
    const [L, W, H] = c.size;
    const cx = c.center[0], cy = c.center[1], cz = c.center[2];
    tmp.min.set(cx - L/2, cy - W/2, cz - H/2);
    tmp.max.set(cx + L/2, cy + W/2, cz + H/2);
    box.union(tmp);
  }
  return box;
}
function separationOffset(strokePolys, cuboids, marginFrac = 0.15) {
  const bs = bboxOfPolylines(strokePolys);
  const bc = bboxOfCuboids(cuboids);
  if (bs.isEmpty() || bc.isEmpty()) return new THREE.Vector3(0, 0, 0);

  const sSize = new THREE.Vector3(), sCenter = new THREE.Vector3();
  const cSize = new THREE.Vector3(), cCenter = new THREE.Vector3();
  bs.getSize(sSize); bs.getCenter(sCenter);
  bc.getSize(cSize); bc.getCenter(cCenter);

  const diag = Math.max(sSize.length(), cSize.length());
  const gap = (sSize.x / 2) + (cSize.x / 2) + diag * marginFrac;

  // move cuboids to +X of strokes by 'gap'
  return new THREE.Vector3((sCenter.x - cCenter.x) + gap, (sCenter.y - cCenter.y), (sCenter.z - cCenter.z));
}

export default function App() {
  const [data, setData] = useState({
    perturbed_feature_lines: [],
    perturbed_construction_lines: [],
    feature_lines: [],
  });
  const [cuboids, setCuboids] = useState([]);

  useEffect(() => {
    loadFromAPI().then(setData).catch(console.error);
  }, []);

  // Z-up: rotate world (Three is Y-up)
  const zUpRotation = useMemo(() => new THREE.Euler(-Math.PI / 2, 0, 0), []);

  // Compute offset so cuboids render beside strokes
  const offsetVec = useMemo(
    () => separationOffset(data.perturbed_feature_lines, cuboids),
    [data.perturbed_feature_lines, cuboids]
  );

  // Camera fit over both strokes and (offset) cuboids
  useEffect(() => {
    const cam = window.__r3f?.store.getState().camera;
    if (!cam) return;

    const box = new THREE.Box3();

    // strokes contribution
    for (const pl of data.perturbed_feature_lines) {
      for (const p of pl.points ?? []) box.expandByPoint(new THREE.Vector3(p[0], p[1], p[2]));
    }

    // cuboids contribution (with offset)
    const tmp = new THREE.Box3();
    for (const c of cuboids) {
      const [L, W, H] = c.size;
      const cx = c.center[0] + offsetVec.x;
      const cy = c.center[1] + offsetVec.y;
      const cz = c.center[2] + offsetVec.z;
      tmp.min.set(cx - L/2, cy - W/2, cz - H/2);
      tmp.max.set(cx + L/2, cy + W/2, cz + H/2);
      box.union(tmp);
    }

    if (box.isEmpty()) return;
    const size = new THREE.Vector3(), center = new THREE.Vector3();
    box.getSize(size); box.getCenter(center);
    const maxSize = Math.max(size.x, size.y, size.z) || 1;
    const dist = maxSize * 1.8 / Math.tan((cam.fov * Math.PI) / 360);
    cam.position.copy(center.clone().add(new THREE.Vector3(1, 1, 1).normalize().multiplyScalar(dist)));
    cam.near = Math.max(0.001, maxSize / 1000);
    cam.far = Math.max(1000, maxSize * 1000);
    cam.lookAt(center);
    cam.updateProjectionMatrix();
  }, [data.perturbed_feature_lines, cuboids, offsetVec]);

  return (
    <div style={{ width: "100vw", height: "100vh", background: "#f8fafc" }}>
      {/* Button to call /api/execute-default?use_offsets=true&use_scales=false */}
      <ProgramLoader onCuboids={setCuboids} />

      <Canvas camera={{ fov: 45 }}>
        <color attach="background" args={["#f8fafc"]} />
        <Suspense fallback={null}>
          <ambientLight intensity={0.7} />
          <directionalLight position={[5, 5, 10]} intensity={0.7} />
          <directionalLight position={[-5, -3, -8]} intensity={0.2} />

          {/* Z-up world */}
          <group rotation={zUpRotation}>
            {/* strokes (left) */}
            <PolylineGroup polylines={data.perturbed_feature_lines} color="#111827" lineWidth={1.6} />

            {/* cuboids (shifted to the right by offsetVec) */}
            <group position={[offsetVec.x, offsetVec.y, offsetVec.z]}>
              <CuboidGroup cuboids={cuboids} />
            </group>
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
