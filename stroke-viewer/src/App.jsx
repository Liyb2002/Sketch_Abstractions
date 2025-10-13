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
const saveAnchors = async (anchorsArray) => {
  return fetchJSON("/api/save-anchors", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ anchors: anchorsArray }),
  });
};

/* ================= colors & keys ================= */
const PALETTE = [
  "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
  "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
  "#6366f1","#10b981","#f59e0b","#ef4444","#14b8a6",
];
function hashString(s) { let h=2166136261>>>0; for (let i=0;i<s.length;i++){h^=s.charCodeAt(i); h=Math.imul(h,16777619);} return h>>>0; }
function colorForName(nameOrId) { const idx = hashString(String(nameOrId ?? "")) % PALETTE.length; return PALETTE[idx]; }
function normalizeKey(s) { return String(s ?? "").trim().toLowerCase().replace(/\s+/g, " "); }

/* ================= UI bits ================= */
function DebugOverlay({ cuboids, anchorsMap, colorMap, selectedComp }) {
  return (
    <div style={{
      position: "absolute", left: 12, top: 12, zIndex: 9999,
      background: "rgba(255,255,255,0.95)", border: "1px solid #e5e7eb",
      borderRadius: 8, padding: "6px 8px", fontSize: 12, color: "#111827"
    }}>
      <div>cuboids: {cuboids.length}</div>
      <div>anchors: {anchorsMap.size}</div>
      <div>colorMap keys: {colorMap.size}</div>
      <div>selected: {selectedComp || "-"}</div>
    </div>
  );
}

function Instructions({ selected, onReset, onExport, onImport, onSave, saveDisabled }) {
  return (
    <div style={{
      position: "absolute", left: 12, bottom: 12, zIndex: 9999,
      background: "rgba(255,255,255,0.97)", border: "1px solid #e5e7eb",
      borderRadius: 8, padding: "8px 10px", fontSize: 13, color: "#111827",
      display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap"
    }}>
      <span style={{ fontWeight: 600 }}>
        {selected ? `Selected component: ${selected}. Now click a stroke to anchor.` : "Click a component, then click a stroke to anchor."}
      </span>
      <button onClick={onReset} style={btnStyle}>Clear selection</button>
      <button onClick={onSave} style={{ ...btnStyle, opacity: saveDisabled ? 0.5 : 1 }} disabled={saveDisabled}>
        Save anchor strokes
      </button>
      <button onClick={onExport} style={btnStyle}>Export anchors</button>
      <label style={{ ...btnStyle, cursor: "pointer" }}>
        Import anchors
        <input type="file" accept="application/json" onChange={onImport} style={{ display: "none" }} />
      </label>
    </div>
  );
}
const btnStyle = {
  padding: "4px 8px", fontSize: 12, borderRadius: 6, border: "1px solid #d1d5db", background: "#fff"
};

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

/* ========== clickable strokes (for selection) ========== */
function ClickableStroke({
  index, points, onPick, visibleLineWidth = 1.4, pickLineWidth = 8.0,
  baseColor = "#e5e7eb", highlightColor = null
}) {
  if (!points || points.length < 2) return null;

  const visColor = highlightColor || baseColor;
  return (
    <group>
      <Line points={points} color={visColor} lineWidth={visibleLineWidth} transparent={false} />
      <Line
        points={points}
        color={highlightColor || baseColor}
        lineWidth={pickLineWidth}
        transparent
        opacity={0.0}
        onPointerDown={(e) => { e.stopPropagation(); onPick?.(index); }}
      />
    </group>
  );
}

function ClickableStrokes({
  strokesPF,
  anchorsMap,            // Map(compKey -> strokeIndex)
  colorMap,               // Map(compKey -> color)
  selectedCompKey,        // normalized key of selected component
  onPickStrokeIndex,      // (strokeIndex) => void
}) {
  if (!strokesPF?.length) return null;

  // strokeIndex -> compKey (so we can color anchored strokes)
  const strokeToComp = new Map();
  anchorsMap.forEach((sidx, compKey) => {
    if (Number.isInteger(sidx)) strokeToComp.set(sidx, compKey);
  });

  return (
    <group>
      {strokesPF.map((pl, i) => {
        const compKey = strokeToComp.get(i); // which component (if any) anchors to this stroke
        const highlightColor = compKey ? (colorMap.get(compKey) || "#111827") : null;

        return (
          <ClickableStroke
            key={i}
            index={i}
            points={pl.points}
            onPick={(sidx) => onPickStrokeIndex?.(sidx)}
            visibleLineWidth={highlightColor ? 3.0 : 1.4}
            pickLineWidth={8.0}
            baseColor="#d1d5db"
            highlightColor={highlightColor}
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

function ClickableCuboidEdges({ cuboid, color, lineWidth = 3.0, isSelected = false, onPick }) {
  const corners = cuboidCorners(cuboid.center, cuboid.size, cuboid.rotationEuler);
  const thick = isSelected ? lineWidth + 1.5 : lineWidth;

  return (
    <group
      onPointerDown={(e) => { e.stopPropagation(); onPick?.(cuboid); }}
      onPointerOver={(e) => { e.stopPropagation(); document.body.style.cursor = "pointer"; }}
      onPointerOut={() => { document.body.style.cursor = "default"; }}
    >
      {EDGE_IDX.map(([a, b], i) => (
        <Line
          key={`${cuboid.id}-${i}`}
          points={[
            [corners[a].x, corners[a].y, corners[a].z],
            [corners[b].x, corners[b].y, corners[b].z],
          ]}
          color={color}
          lineWidth={thick}
          transparent={false}
          dashed={false}
          depthTest={true}
        />
      ))}
      {EDGE_IDX.map(([a, b], i) => (
        <Line
          key={`${cuboid.id}-pick-${i}`}
          points={[
            [corners[a].x, corners[a].y, corners[a].z],
            [corners[b].x, corners[b].y, corners[b].z],
          ]}
          color={color}
          lineWidth={thick + 6}
          transparent
          opacity={0.0}
          onPointerDown={(e) => { e.stopPropagation(); onPick?.(cuboid); }}
        />
      ))}
    </group>
  );
}

function CuboidEdgesInteractive({ cuboids, colorMap, selectedCompKey, onPickCuboid }) {
  if (!cuboids?.length) return null;
  return (
    <group>
      {cuboids.map((c) => {
        const key = normalizeKey(c.name ?? c.id);
        const clr = colorMap.get(key) || "#111827";
        const isSelected = key === selectedCompKey;
        return (
          <ClickableCuboidEdges
            key={c.id}
            cuboid={c}
            color={isSelected ? "#111827" : clr}
            lineWidth={3.0}
            isSelected={isSelected}
            onPick={(cub) => onPickCuboid?.(cub)}
          />
        );
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
  const [anchorsMap, setAnchorsMap] = useState(new Map()); // Map(compKey -> strokeIndex)
  const [selectedCompKey, setSelectedCompKey] = useState(null);
  const [saving, setSaving] = useState(false);

  // Auto-load strokes & default execution
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

        // init anchors map
        const init = new Map();
        (c.anchors || []).forEach(a => {
          const compKey = normalizeKey(a.cuboidName ?? a.cuboidId);
          if (Number.isInteger(a.strokeIndex)) init.set(compKey, a.strokeIndex);
        });
        setAnchorsMap(init);
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
    anchorsMap.forEach((_sidx, compKey) => {
      if (!m.has(compKey)) m.set(compKey, colorForName(compKey));
    });
    return m;
  }, [cuboids, anchorsMap]);

  // derived values
  const zUpRotation = useMemo(() => new THREE.Euler(-Math.PI / 2, 0, 0), []);
  const offsetVec = useMemo(
    () => separationOffset(strokes.perturbed_feature_lines, cuboids),
    [strokes.perturbed_feature_lines, cuboids]
  );
  const legendNames = useMemo(() => {
    const map = new Map();
    for (const c of cuboids) {
      const label = String(c.name ?? c.id ?? "");
      const key = normalizeKey(label);
      if (!map.has(key)) map.set(key, label);
    }
    anchorsMap.forEach((_sidx, compKey) => {
      if (!map.has(compKey)) map.set(compKey, compKey);
    });
    return Array.from(map.values());
  }, [cuboids, anchorsMap]);

  // Fit camera to strokes + offset cuboids
  useEffect(() => {
    const cam = window.__r3f?.store.getState().camera;
    if (!cam) return;
    const box = new THREE.Box3();

    for (const pl of strokes.perturbed_feature_lines) {
      for (const p of pl.points ?? []) box.expandByPoint(new THREE.Vector3(p[0], p[1], p[2]));
    }
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

  // Handlers
  const handlePickCuboid = (cub) => {
    const compKey = normalizeKey(cub.name ?? cub.id);
    setSelectedCompKey(compKey);
  };
  const handlePickStrokeIndex = (sidx) => {
    if (!Number.isInteger(sidx)) return;
    if (!selectedCompKey) return;
    setAnchorsMap((prev) => {
      const next = new Map(prev);
      next.set(selectedCompKey, sidx);
      return next;
    });
  };

  // Save anchors → POST /api/save-anchors
  const handleSave = async () => {
    try {
      setSaving(true);
      // Build array with ORIGINAL component labels (prefer c.name over id)
      const anchorsArray = [];
      anchorsMap.forEach((sidx, compKey) => {
        const found = cuboids.find(c => normalizeKey(c.name ?? c.id) === compKey);
        const label = found ? (found.name ?? found.id) : compKey; // fallback to key
        if (Number.isInteger(sidx)) {
          anchorsArray.push({ cuboidId: String(label), strokeIndex: sidx });
        }
      });
      const resp = await saveAnchors(anchorsArray);
      console.log("Saved anchors:", resp);
      alert(`Saved ${resp.saved} anchors to:\n${resp.path}`);
    } catch (e) {
      console.error(e);
      alert("Failed to save anchors. Check console for details.");
    } finally {
      setSaving(false);
    }
  };

  // Export/Import (front-end convenience)
  const handleExport = () => {
    const obj = {};
    anchorsMap.forEach((sidx, compKey) => { obj[compKey] = sidx; });
    const blob = new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "anchors_map.json"; a.click();
    URL.revokeObjectURL(url);
  };
  const handleImport = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      const obj = JSON.parse(text);
      const next = new Map();
      Object.entries(obj).forEach(([compKey, sidx]) => {
        const k = normalizeKey(compKey);
        if (Number.isInteger(sidx)) next.set(k, sidx);
      });
      setAnchorsMap(next);
    } catch (err) {
      console.error("Failed to import anchors_map.json", err);
    } finally {
      e.target.value = "";
    }
  };

  return (
    <div style={{ width: "100vw", height: "100vh", background: "#ffffff" }}>
      <DebugOverlay cuboids={cuboids} anchorsMap={anchorsMap} colorMap={colorMap} selectedComp={selectedCompKey} />
      <Legend names={legendNames} colorMap={colorMap} />
      <Instructions
        selected={selectedCompKey}
        onReset={() => setSelectedCompKey(null)}
        onExport={handleExport}
        onImport={handleImport}
        onSave={handleSave}
        saveDisabled={saving || anchorsMap.size === 0}
      />

      <Canvas camera={{ fov: 45 }}>
        <color attach="background" args={["#ffffff"]} />
        <Suspense fallback={null}>
          <ambientLight intensity={0.6} />
          <directionalLight position={[6, 8, 10]} intensity={1.0} />
          <directionalLight position={[-6, -4, -8]} intensity={0.3} />

          {/* Z-up world */}
          <group rotation={new THREE.Euler(-Math.PI / 2, 0, 0)}>
            {/* Clickable strokes (left), colored where anchored */}
            <ClickableStrokes
              strokesPF={strokes.perturbed_feature_lines}
              anchorsMap={anchorsMap}
              colorMap={colorMap}
              selectedCompKey={selectedCompKey}
              onPickStrokeIndex={handlePickStrokeIndex}
            />

            {/* Clickable cuboids (right) */}
            <group position={[offsetVec.x, offsetVec.y, offsetVec.z]}>
              <CuboidEdgesInteractive
                cuboids={cuboids}
                colorMap={colorMap}
                selectedCompKey={selectedCompKey}
                onPickCuboid={handlePickCuboid}
              />
            </group>

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
