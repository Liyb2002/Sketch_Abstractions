// stroke-viewer/src/app.jsx
import { Suspense, useEffect, useMemo, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Line, GizmoHelper, GizmoViewport } from "@react-three/drei";
import * as THREE from "three";

// ---------- API helpers ----------
async function fetchJSON(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(`${opts?.method || "GET"} ${url} -> ${res.status}`);
  return await res.json();
}
const loadStrokes = () => fetchJSON("/api/strokes");
const loadCuboidsDefault = async ({ use_offsets = true, use_scales = false } = {}) => {
  const qs = `?use_offsets=${String(use_offsets)}&use_scales=${String(use_scales)}`;
  return fetchJSON(`/api/execute-default${qs}`, { method: "POST" });
};

// ---------- strokes renderer ----------
function PolylineGroup({ polylines, color = "#111827", lineWidth = 2.0 }) {
  if (!polylines?.length) return null;
  return (
    <group>
      {polylines.map((pl, i) =>
        pl.points?.length >= 2 ? (
          <Line key={i} points={pl.points} color={color} lineWidth={lineWidth} transparent={false} />
        ) : null
      )}
    </group>
  );
}

// ---------- cuboids as edge-only thick lines (no meshes) ----------
const EDGE_IDX = [
  [0,1],[1,2],[2,3],[3,0], // bottom
  [4,5],[5,6],[6,7],[7,4], // top
  [0,4],[1,5],[2,6],[3,7], // verticals
];

function cuboidCorners(center, size, rotationEuler) {
  const [cx, cy, cz] = center;
  const [L, W, H] = size;
  const hx = L / 2, hy = W / 2, hz = H / 2;

  const corners = [
    [-hx, -hy, -hz], [ hx, -hy, -hz],
    [ hx,  hy, -hz], [-hx,  hy, -hz],
    [-hx, -hy,  hz], [ hx, -hy,  hz],
    [ hx,  hy,  hz], [-hx,  hy,  hz],
  ].map(([x, y, z]) => new THREE.Vector3(x, y, z));

  if (rotationEuler?.length === 3) {
    const e = new THREE.Euler(rotationEuler[0], rotationEuler[1], rotationEuler[2]);
    const m = new THREE.Matrix4().makeRotationFromEuler(e);
    corners.forEach(v => v.applyMatrix4(m));
  }
  corners.forEach(v => v.add(new THREE.Vector3(cx, cy, cz)));
  return corners;
}

function CuboidEdges({ cuboids, color = "#111827", lineWidth = 3.0 }) {
  if (!cuboids?.length) return null;
  return (
    <group>
      {cuboids.flatMap((c) => {
        const corners = cuboidCorners(c.center, c.size, c.rotationEuler);
        return EDGE_IDX.map(([a, b], i) => (
          <Line
            key={`${c.id}-${i}`}
            points={[
              [corners[a].x, corners[a].y, corners[a].z],
              [corners[b].x, corners[b].y, corners[b].z],
            ]}
            color={color}
            lineWidth={lineWidth}
            transparent={false}
            dashed={false}
            depthTest={true}
          />
        ));
      })}
    </group>
  );
}

// ---------- layout helpers (place cuboids beside strokes) ----------
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
  const [strokes, setStrokes] = useState({
    perturbed_feature_lines: [],
    perturbed_construction_lines: [],
    feature_lines: [],
  });
  const [cuboids, setCuboids] = useState([]);

  // Auto-load both on mount
  useEffect(() => {
    (async () => {
      try {
        const [s, c] = await Promise.all([
          loadStrokes(),
          loadCuboidsDefault({ use_offsets: true, use_scales: false }),
        ]);
        setStrokes(s);
        setCuboids(c.cuboids || []);
      } catch (e) {
        console.error(e);
      }
    })();
  }, []);

  // Z-up (Three is Y-up by default)
  const zUpRotation = useMemo(() => new THREE.Euler(-Math.PI / 2, 0, 0), []);

  // Non-overlap offset
  const offsetVec = useMemo(
    () => separationOffset(strokes.perturbed_feature_lines, cuboids),
    [strokes.perturbed_feature_lines, cuboids]
  );

  // Camera fit over strokes + (offset) cuboids
  useEffect(() => {
    const cam = window.__r3f?.store.getState().camera;
    if (!cam) return;

    const box = new THREE.Box3();

    // strokes
    for (const pl of strokes.perturbed_feature_lines) {
      for (const p of pl.points ?? []) {
        box.expandByPoint(new THREE.Vector3(p[0], p[1], p[2]));
      }
    }

    // cuboids (apply offset)
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
  }, [strokes.perturbed_feature_lines, cuboids, offsetVec]);

  return (
    <div style={{ width: "100vw", height: "100vh", background: "#ffffff" }}>
      <Canvas camera={{ fov: 45 }}>
        <color attach="background" args={["#ffffff"]} />
        <Suspense fallback={null}>
          <ambientLight intensity={0.7} />
          <directionalLight position={[5, 5, 10]} intensity={0.7} />
          <directionalLight position={[-5, -3, -8]} intensity={0.2} />

          {/* Z-up world */}
          <group rotation={zUpRotation}>
            {/* Strokes (left) */}
            <PolylineGroup polylines={strokes.perturbed_feature_lines} color="#111827" lineWidth={2.4} />

            {/* Cuboids (edge-only, shifted to the right) */}
            <group position={[offsetVec.x, offsetVec.y, offsetVec.z]}>
              <CuboidEdges cuboids={cuboids} color="#111827" lineWidth={3.0} />
            </group>

            {/* Remove this line too if you want ZERO helpers */}
            <axesHelper args={[2]} />
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
