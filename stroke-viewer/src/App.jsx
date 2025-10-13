// stroke-viewer/src/app.jsx

// --- Build tag to confirm bundle freshness ---
console.log("[BUILD TAG]", new Date().toISOString());

import { Suspense, useEffect, useMemo, useState, Fragment } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Line, GizmoHelper, GizmoViewport } from "@react-three/drei";
import * as THREE from "three";

/* ================= API helpers ================= */
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

/* ================= colors & keys ================= */
const PALETTE = [
  "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
  "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
  "#6366f1","#10b981","#f59e0b","#ef4444","#14b8a6",
];
function hashString(s) { let h=2166136261>>>0; for (let i=0;i<s.length;i++){h^=s.charCodeAt(i); h=Math.imul(h,16777619);} return h>>>0; }
function colorForName(nameOrId) { const idx = hashString(String(nameOrId ?? "")) % PALETTE.length; return PALETTE[idx]; }
function normalizeKey(s) {
  return String(s ?? "").trim().toLowerCase().replace(/\s+/g, " ");
}

/* ================= small debug helpers ================= */
function Ping({ label }) {
  console.log("Ping render:", label);
  return null;
}
function DebugOverlay({ cuboids, anchors, colorMap }) {
  return (
    <div style={{
      position: "absolute", left: 12, top: 12, zIndex: 9999,
      background: "rgba(255,255,255,0.95)", border: "1px solid #e5e7eb",
      borderRadius: 8, padding: "6px 8px", fontSize: 12, color: "#111827"
    }}>
      <div>cuboids: {cuboids.length}</div>
      <div>anchors: {anchors.length}</div>
      <div>colorMap keys: {colorMap.size}</div>
    </div>
  );
}

/* ================= strokes renderer ================= */
function PolylineGroup({ polylines, color = "#e5e7eb", lineWidth = 1.4 }) {
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

/* ========== anchored strokes overlay (INDEX-BASED, with logs) ========== */
function AnchoredStrokesByIndex({ anchors, strokesPF, colorMap, lineWidth = 4.0 }) {
  // Immediate log on every render
  console.log(
    "[AnchoredStrokesByIndex] anchors:", anchors?.length ?? 0,
    "strokesPF:", strokesPF?.length ?? 0
  );

  useEffect(() => {
    console.group("🔎 Anchor → Component → Color mapping");
    const totalStrokes = strokesPF?.length ?? 0;
    console.log("total anchors:", anchors?.length ?? 0, "total strokes:", totalStrokes);
    (anchors ?? []).forEach((a, i) => {
      const rawKey = a.cuboidName ?? a.cuboidId;
      const key = normalizeKey(rawKey);
      const hasColor = colorMap?.has?.(key);
      const color = hasColor ? colorMap.get(key) : undefined;
      const sidx = a.strokeIndex;
      const validStroke = Number.isInteger(sidx) && sidx >= 0 && sidx < totalStrokes;
      const strokeLen = validStroke ? (strokesPF[sidx]?.points?.length ?? 0) : 0;
      console.log(`[${i}]`, { rawKey, key, hasColor, color, strokeIndex: sidx, strokeLen });
      if (!hasColor) console.warn(`⚠️ No color for key="${key}" (from "${rawKey}")`);
      if (!validStroke) console.warn(`⚠️ Bad strokeIndex=${sidx} (total=${totalStrokes}) for key="${key}"`);
    });
    console.groupEnd();
  }, [anchors, colorMap, strokesPF]);

  if (!anchors?.length || !strokesPF?.length) return null;

  return (
    <group>
      {anchors.map((a, i) => {
        const sidx = a.strokeIndex;
        const poly = strokesPF[sidx];
        if (!poly || !poly.points || poly.points.length < 2) return null;

        const key = normalizeKey(a.cuboidName ?? a.cuboidId);
        const clr = colorMap.get(key) || "#111827";

        return (
          <Line
            key={`${a.cuboidId}-${i}`}
            points={poly.points}
            color={clr}
            lineWidth={lineWidth}
            transparent={false}
            depthTest={false}   // draw on top
          />
        );
      })}
    </group>
  );
}

/* ================= cuboids as bounding boxes (edge-only) ================= */
const EDGE_IDX = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
function cuboidCorners(center, size, rotationEuler) {
  const [cx, cy, cz] = center, [L, W, H] = size;
  const hx=L/2, hy=W/2, hz=H/2;
  const corners = [
    [-hx,-hy,-hz],[ hx,-hy,-hz],[ hx, hy,-hz],[-hx, hy,-hz],
    [-hx,-hy, hz],[ hx,-hy, hz],[ hx, hy, hz],[-hx, hy, hz],
  ].map(([x,y,z]) => new THREE.Vector3(x,y,z));
  if (rotationEuler?.length === 3) {
    const e = new THREE.Euler(rotationEuler[0], rotationEuler[1], rotationEuler[2]);
    const m = new THREE.Matrix4().makeRotationFromEuler(e);
    corners.forEach(v => v.applyMatrix4(m));
  }
  corners.forEach(v => v.add(new THREE.Vector3(cx,cy,cz)));
  return corners;
}
function CuboidEdges({ cuboids, colorMap, lineWidth = 3.0 }) {
  if (!cuboids?.length) return null;
  return (
    <group>
      {cuboids.flatMap((c) => {
        const key = normalizeKey(c.name ?? c.id);
        const clr = colorMap.get(key) || "#111827";
        const corners = cuboidCorners(c.center, c.size, c.rotationEuler);
        return EDGE_IDX.map(([a, b], i) => (
          <Line
            key={`${c.id}-${i}`}
            points={[
              [corners[a].x, corners[a].y, corners[a].z],
              [corners[b].x, corners[b].y, corners[b].z],
            ]}
            color={clr}
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

/* ================= layout helpers ================= */
function bboxOfPolylines(polys) {
  const box = new THREE.Box3();
  for (const pl of polys) for (const p of pl.points ?? []) {
    box.expandByPoint(new THREE.Vector3(p[0], p[1], p[2]));
  }
  return box;
}
function bboxOfCuboids(cuboids) {
  const box = new THREE.Box3(), tmp = new THREE.Box3();
  for (const c of cuboids) {
    const [L, W, H] = c.size; const [cx, cy, cz] = c.center;
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
  const gap = (sSize.x/2) + (cSize.x/2) + diag * marginFrac;
  return new THREE.Vector3((sCenter.x - cCenter.x) + gap, (sCenter.y - cCenter.y), (sCenter.z - cCenter.z));
}

/* ================= legend (top-right) ================= */
function Legend({ names, colorMap }) {
  if (!names.length) return null;
  return (
    <div style={{
      position: "absolute", right: 12, top: 12, zIndex: 20,
      background: "rgba(255,255,255,0.95)", border: "1px solid #e5e7eb",
      borderRadius: 8, padding: "8px 10px", maxWidth: 320
    }}>
      <div style={{ fontSize: 12, marginBottom: 6, color: "#374151" }}>Components</div>
      <div style={{ display: "grid", gridTemplateColumns: "auto 1fr", gap: "6px 10px" }}>
        {names.map((label) => (
          <Fragment key={label}>
            <span
              style={{
                width: 14, height: 14, borderRadius: 3,
                background: colorMap.get(normalizeKey(label)) || "#111827",
                display: "inline-block", marginTop: 2
              }}
            />
            <span style={{ fontSize: 12, color: "#111827", whiteSpace: "nowrap" }}>{label}</span>
          </Fragment>
        ))}
      </div>
    </div>
  );
}

/* ================= main ================= */
export default function App() {
  const [strokes, setStrokes] = useState({
    perturbed_feature_lines: [],
    perturbed_construction_lines: [],
    feature_lines: [],
  });
  const [cuboids, setCuboids] = useState([]);
  const [anchors, setAnchors] = useState([]);

  // Auto-load both (and log payload)
  useEffect(() => {
    (async () => {
      try {
        const [s, c] = await Promise.all([
          loadStrokes(),
          loadCuboidsDefault({ use_offsets: true, use_scales: false }),
        ]);

        console.log("[UI] execute-default payload:", c, "anchors length:", (c?.anchors || []).length);

        setStrokes(s);
        setCuboids(c.cuboids || []);
        setAnchors(c.anchors || []);
      } catch (e) {
        console.error(e);
      }
    })();
  }, []);

  // color map: union of cuboid names and anchor names (normalized)
  const colorMap = useMemo(() => {
    const m = new Map();
    for (const c of cuboids) {
      const key = normalizeKey(c.name ?? c.id);
      if (!m.has(key)) m.set(key, colorForName(key));
    }
    for (const a of anchors) {
      const key = normalizeKey(a.cuboidName ?? a.cuboidId);
      if (!m.has(key)) m.set(key, colorForName(key));
    }
    return m;
  }, [cuboids, anchors]);

  const zUpRotation = useMemo(() => new THREE.Euler(-Math.PI / 2, 0, 0), []);
  const offsetVec = useMemo(
    () => separationOffset(strokes.perturbed_feature_lines, cuboids),
    [strokes.perturbed_feature_lines, cuboids]
  );

  // Fit camera to strokes + offset cuboids
  useEffect(() => {
    const cam = window.__r3f?.store.getState().camera;
    if (!cam) return;
    const box = new THREE.Box3();

    // base strokes
    for (const pl of strokes.perturbed_feature_lines) {
      for (const p of pl.points ?? []) box.expandByPoint(new THREE.Vector3(p[0], p[1], p[2]));
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

  // Legend labels from union (original labels preserved)
  const legendNames = useMemo(() => {
    const map = new Map();
    for (const c of cuboids) {
      const label = String(c.name ?? c.id ?? "");
      const key = normalizeKey(label);
      if (!map.has(key)) map.set(key, label);
    }
    for (const a of anchors) {
      const label = String(a.cuboidName ?? a.cuboidId ?? "");
      const key = normalizeKey(label);
      if (!map.has(key)) map.set(key, label);
    }
    return Array.from(map.values());
  }, [cuboids, anchors]);

  return (
    <div style={{ width: "100vw", height: "100vh", background: "#ffffff" }}>
      <DebugOverlay cuboids={cuboids} anchors={anchors} colorMap={colorMap} />
      <Legend names={legendNames} colorMap={colorMap} />
      <Ping label="outside-canvas" />

      <Canvas camera={{ fov: 45 }}>
        <color attach="background" args={["#ffffff"]} />
        <Suspense fallback={null}>
          <ambientLight intensity={0.6} />
          <directionalLight position={[6, 8, 10]} intensity={1.0} />
          <directionalLight position={[-6, -4, -8]} intensity={0.3} />

          {/* Z-up world */}
          <group rotation={zUpRotation}>
            {/* Base strokes in light grey */}
            <PolylineGroup polylines={strokes.perturbed_feature_lines} color="#e5e7eb" lineWidth={1.4} />

            {/* Force that this subtree is mounted */}
            <Ping label="inside-canvas-before-anchors" />

            {/* Anchored strokes highlighted with component colors */}
            <AnchoredStrokesByIndex
              anchors={anchors}
              strokesPF={strokes.perturbed_feature_lines}
              colorMap={colorMap}
              lineWidth={4.0}
            />

            {/* Cuboids (edge-only), shifted right to avoid overlap */}
            <group position={[offsetVec.x, offsetVec.y, offsetVec.z]}>
              <CuboidEdges cuboids={cuboids} colorMap={colorMap} lineWidth={3.0} />
            </group>

            {/* Remove axesHelper if you want zero helpers */}
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
