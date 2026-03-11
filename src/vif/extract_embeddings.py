"""Extract VIF model embeddings and generate interactive 3D visualization.

Loads a trained ordinal critic checkpoint, extracts hidden-layer activations
and SBERT embeddings for all data points, reduces to 3D via PCA and t-SNE,
and generates a self-contained HTML file with a Three.js visualization.

Usage:
    python -m src.vif.extract_embeddings \
        --checkpoint logs/experiments/artifacts/.../BalancedSoftmax/selected_checkpoint.pt
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.vif.critic_ordinal import OrdinalCriticBase
from src.vif.dataset import VIFDataset, load_all_data, split_by_persona
from src.vif.encoders import create_encoder
from src.vif.state_encoder import StateEncoder


# ─── Data extraction ─────────────────────────────────────────────────────────


def _load_model(checkpoint_path: str | Path) -> tuple[OrdinalCriticBase, dict]:
    """Load a trained ordinal critic from checkpoint."""
    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = OrdinalCriticBase.from_config(cp["model_config"])
    model.load_state_dict(cp["model_state_dict"])
    model.eval()
    return model, cp


def _build_datasets(
    training_config: dict,
) -> tuple[VIFDataset, VIFDataset, VIFDataset, StateEncoder]:
    """Recreate train/val/test datasets matching the checkpoint's split."""
    encoder = create_encoder(
        {
            "type": "sbert",
            "model_name": training_config["encoder_model"],
            "trust_remote_code": training_config.get("trust_remote_code", True),
            "truncate_dim": training_config["truncate_dim"],
            "text_prefix": training_config.get("text_prefix", "classification: "),
        }
    )
    state_encoder = StateEncoder(
        encoder, window_size=training_config.get("window_size", 1)
    )

    labels_df, entries_df = load_all_data()

    # Reproduce the exact split from training
    holdout_path = training_config.get("fixed_holdout_manifest_path")
    fixed_val, fixed_test = None, None
    if holdout_path:
        import yaml

        with open(holdout_path) as f:
            manifest = yaml.safe_load(f)
        fixed_val = set(manifest.get("val_persona_ids", []))
        fixed_test = set(manifest.get("test_persona_ids", []))

    train_df, val_df, test_df = split_by_persona(
        labels_df,
        entries_df,
        train_ratio=training_config.get("train_ratio", 0.70),
        val_ratio=training_config.get("val_ratio", 0.15),
        seed=training_config.get("split_seed", 42),
        fixed_val_persona_ids=fixed_val,
        fixed_test_persona_ids=fixed_test,
    )

    train_ds = VIFDataset(train_df, state_encoder, cache_embeddings=True)
    val_ds = VIFDataset(val_df, state_encoder, cache_embeddings=True)
    test_ds = VIFDataset(test_df, state_encoder, cache_embeddings=True)

    return train_ds, val_ds, test_ds, state_encoder


def _extract_from_dataset(
    model: OrdinalCriticBase,
    dataset: VIFDataset,
    split_name: str,
) -> list[dict]:
    """Run forward pass, capture hidden activations, and collect metadata."""
    activation_cache: dict[str, torch.Tensor] = {}

    def hook_fn(module, inp, out):
        activation_cache["hidden"] = out.detach()

    # Hook after layer-2 LayerNorm (the final hidden representation)
    hook_handle = model.ln2.register_forward_hook(hook_fn)

    points = []
    try:
        for idx in range(len(dataset)):
            state, target = dataset[idx]
            state_batch = state.unsqueeze(0)  # (1, state_dim)

            with torch.no_grad():
                probs = model.predict_probabilities(state_batch)  # (1, 10, 3)
                preds = model.predict(state_batch)  # (1, 10)

            hidden = activation_cache["hidden"].squeeze(0).numpy()  # (hidden_dim,)
            probs_np = probs.squeeze(0).numpy()  # (10, 3)
            preds_np = preds.squeeze(0).numpy()  # (10,)
            target_np = target.numpy()  # (10,)

            # Extract SBERT embedding from the state vector (first 256 dims)
            sbert_emb = state.numpy()[:256]  # (256,)

            # Metadata
            meta = dataset.get_sample_metadata(idx)
            persona_id = meta["persona_id"]
            t_index = meta["t_index"]

            # Get text snippet from the dataset's entry_lookup
            row = dataset.entry_lookup[(persona_id, t_index)]
            text = row.get("initial_entry", "")

            # Compute entropy as uncertainty measure
            entropy = -np.sum(probs_np * np.log(probs_np + 1e-10), axis=1)  # (10,)

            points.append(
                {
                    "persona_id": persona_id,
                    "t_index": int(t_index),
                    "date": meta.get("date", ""),
                    "split": split_name,
                    "text": text,
                    "ground_truth": target_np.tolist(),
                    "predictions": preds_np.tolist(),
                    "probabilities": probs_np.tolist(),
                    "uncertainty": entropy.tolist(),
                    "hidden": hidden.tolist(),
                    "sbert": sbert_emb.tolist(),
                }
            )
    finally:
        hook_handle.remove()

    return points


# ─── Dimensionality reduction ────────────────────────────────────────────────


def _reduce_to_3d(
    points: list[dict],
    key: str,
    method: str = "pca",
    perplexity: float = 30.0,
    random_state: int = 42,
) -> np.ndarray:
    """Reduce high-dim vectors to 3D coordinates."""
    matrix = np.array([p[key] for p in points], dtype=np.float32)

    if method == "pca":
        reducer = PCA(n_components=3, random_state=random_state)
    elif method == "tsne":
        reducer = TSNE(
            n_components=3,
            perplexity=min(perplexity, len(points) - 1),
            random_state=random_state,
            init="pca",
            learning_rate="auto",
            max_iter=1000,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    coords = reducer.fit_transform(matrix)

    # Normalize to [-10, 10] for the Three.js scene
    max_abs = np.abs(coords).max()
    if max_abs > 0:
        coords = coords * (10.0 / max_abs)

    return coords


# ─── HTML generation ─────────────────────────────────────────────────────────


def _generate_html(data: dict, output_path: Path) -> None:
    """Generate self-contained Three.js visualization HTML."""
    data_json = json.dumps(data, separators=(",", ":"))
    html = _HTML_TEMPLATE.replace("/*__EMBEDDING_DATA__*/null", data_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    print(f"Visualization written to {output_path}")


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Twinkl — Embedding Explorer</title>
<style>
  *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    overflow: hidden;
    background: #050510;
    font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    color: #e0e0f0;
  }
  canvas { display: block; }

  /* ── Glass panels ────────────────────────── */
  .panel {
    position: fixed;
    background: rgba(8, 8, 25, 0.75);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(120, 130, 255, 0.12);
    border-radius: 14px;
    padding: 18px 22px;
    pointer-events: auto;
    transition: opacity 0.3s ease;
  }

  /* Controls panel — top-left */
  #controls {
    top: 24px; left: 24px;
    min-width: 220px;
  }
  #controls h1 {
    font-size: 16px;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 14px;
    background: linear-gradient(135deg, #8ec5fc, #e0c3fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  #controls label {
    display: block;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #8888aa;
    margin: 10px 0 4px;
  }
  #controls select, #controls input[type="range"] {
    width: 100%;
    padding: 6px 8px;
    font-size: 13px;
    background: rgba(20, 20, 50, 0.8);
    border: 1px solid rgba(120, 130, 255, 0.2);
    border-radius: 8px;
    color: #d0d0e8;
    outline: none;
    cursor: pointer;
    -webkit-appearance: none;
  }
  #controls select option { background: #1a1a3a; }
  #controls .stats {
    margin-top: 14px;
    font-size: 11px;
    color: #6666aa;
    line-height: 1.6;
  }

  /* Legend — below controls on the left */
  #legend {
    top: auto; left: 24px;
    min-width: 220px;
    font-size: 12px;
  }
  #legend h2 {
    font-size: 12px;
    font-weight: 500;
    color: #8888aa;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
  }
  .legend-item {
    display: flex;
    align-items: center;
    margin: 5px 0;
  }
  .legend-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    margin-right: 8px;
    box-shadow: 0 0 6px currentColor;
  }

  /* Detail panel — middle-right, shifted toward center */
  #detail {
    top: 50%; right: 80px;
    transform: translateY(-50%);
    width: 440px;
    max-height: 75vh;
    overflow-y: auto;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.25s ease;
  }
  #detail.visible { opacity: 1; pointer-events: auto; }
  #detail h2 {
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 8px;
    color: #b0b0d8;
  }
  #detail .field {
    margin: 6px 0;
    font-size: 12px;
    line-height: 1.5;
  }
  #detail .field-label {
    color: #7777aa;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
  }
  #detail .text-snippet {
    background: rgba(30, 30, 60, 0.6);
    border-radius: 8px;
    padding: 10px 12px;
    font-size: 12px;
    line-height: 1.6;
    color: #c8c8e8;
    margin-top: 4px;
    max-height: 300px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-wrap: break-word;
  }
  #detail .predictions-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 3px 12px;
    font-size: 11px;
    margin-top: 6px;
  }
  .pred-item {
    display: grid;
    grid-template-columns: 1fr 32px 36px;
    align-items: center;
    padding: 2px 0;
  }
  .pred-val { font-weight: 600; text-align: right; }
  .pred-val.neg { color: #ff6b6b; }
  .pred-val.neu { color: #8888bb; }
  .pred-val.pos { color: #69f0ae; }
  .gt-val { font-weight: 400; color: #777799; text-align: right; }

  /* Tooltip */
  #tooltip {
    position: fixed;
    background: rgba(10, 10, 30, 0.9);
    border: 1px solid rgba(120, 130, 255, 0.2);
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 12px;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.15s ease;
    max-width: 250px;
    z-index: 100;
  }
  #tooltip.visible { opacity: 1; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: rgba(120, 130, 255, 0.3); border-radius: 2px; }
</style>
</head>
<body>

<!-- Controls -->
<div id="controls" class="panel">
  <h1>Embedding Explorer</h1>

  <label>Space</label>
  <select id="spaceSelect">
    <option value="hidden_pca">Hidden Layer — PCA</option>
    <option value="hidden_tsne">Hidden Layer — t-SNE</option>
    <option value="sbert_pca">SBERT Embedding — PCA</option>
    <option value="sbert_tsne">SBERT Embedding — t-SNE</option>
  </select>

  <label>Color by</label>
  <select id="colorSelect">
    <option value="split">Data Split</option>
    <option value="prediction">Dominant Prediction</option>
    <option value="ground_truth">Ground Truth</option>
    <option value="persona">Persona</option>
    <option value="uncertainty">Uncertainty</option>
  </select>

  <label>Dimension</label>
  <select id="dimSelect" disabled></select>

  <label>Bloom Intensity</label>
  <input type="range" id="bloomSlider" min="0" max="3" step="0.1" value="0.3">

  <label>Persona Lines</label>
  <select id="lineSelect">
    <option value="off">Hide Trajectories</option>
    <option value="on">Show Trajectories</option>
  </select>

  <div class="stats" id="stats"></div>
</div>

<!-- Legend -->
<div id="legend" class="panel">
  <h2>Legend</h2>
  <div id="legendItems"></div>
</div>

<!-- Detail -->
<div id="detail" class="panel">
  <h2>Entry Details</h2>
  <div id="detailContent"></div>
</div>

<!-- Tooltip -->
<div id="tooltip"></div>

<script type="importmap">
{
  "imports": {
    "three": "https://unpkg.com/three@0.162.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.162.0/examples/jsm/"
  }
}
</script>

<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';

// ─── Data ────────────────────────────────────────────────────────────────────
const DATA = /*__EMBEDDING_DATA__*/null;
if (!DATA) { document.body.innerHTML = '<p style="color:#f88;padding:40px">No data loaded.</p>'; throw new Error('No data'); }

const DIMS = DATA.dimensions;
const POINTS = DATA.points;
const N = POINTS.length;

// ─── Color palettes ──────────────────────────────────────────────────────────
const SPLIT_COLORS = { train: [0.31, 0.76, 0.97], val: [0.88, 0.25, 0.98], test: [1.0, 0.84, 0.25] };
const PRED_COLORS = { '-1': [1.0, 0.42, 0.42], '0': [0.53, 0.53, 0.73], '1': [0.41, 0.94, 0.68] };
const DIM_COLORS = [
  [1.0, 0.42, 0.42], [1.0, 0.56, 0.33], [0.99, 0.78, 0.60],
  [1.0, 0.85, 0.24], [0.42, 0.80, 0.47], [0.30, 0.59, 1.0],
  [0.42, 0.36, 0.91], [0.65, 0.37, 0.92], [1.0, 0.42, 0.51], [0.04, 0.74, 0.89]
];

// Generate persona colors via golden-ratio hue spacing
const personaIds = [...new Set(POINTS.map(p => p.persona_id))];
const PERSONA_COLORS = {};
personaIds.forEach((pid, i) => {
  const hue = (i * 0.618033988749895) % 1.0;
  const c = new THREE.Color().setHSL(hue, 0.7, 0.6);
  PERSONA_COLORS[pid] = [c.r, c.g, c.b];
});

// ─── Scene setup ─────────────────────────────────────────────────────────────
const W = window.innerWidth, H = window.innerHeight;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x050510);
// No fog — let bloom handle depth perception

const camera = new THREE.PerspectiveCamera(55, W / H, 0.1, 500);
camera.position.set(18, 12, 18);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(W, H);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
document.body.prepend(renderer.domElement);

// Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.4;
controls.minDistance = 5;
controls.maxDistance = 80;

// ─── Post-processing ─────────────────────────────────────────────────────────
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloomPass = new UnrealBloomPass(new THREE.Vector2(W, H), 0.3, 0.3, 0.6);
composer.addPass(bloomPass);

// ─── Background star field ───────────────────────────────────────────────────
{
  const starCount = 3000;
  const starPos = new Float32Array(starCount * 3);
  const starSizes = new Float32Array(starCount);
  for (let i = 0; i < starCount; i++) {
    starPos[i * 3]     = (Math.random() - 0.5) * 300;
    starPos[i * 3 + 1] = (Math.random() - 0.5) * 300;
    starPos[i * 3 + 2] = (Math.random() - 0.5) * 300;
    starSizes[i] = Math.random() * 0.8 + 0.2;
  }
  const starGeo = new THREE.BufferGeometry();
  starGeo.setAttribute('position', new THREE.Float32BufferAttribute(starPos, 3));
  starGeo.setAttribute('size', new THREE.Float32BufferAttribute(starSizes, 1));
  const starMat = new THREE.PointsMaterial({
    color: 0x4444aa,
    size: 0.15,
    transparent: true,
    opacity: 0.35,
    sizeAttenuation: true,
    depthWrite: false,
  });
  scene.add(new THREE.Points(starGeo, starMat));
}

// ─── Main particle system ────────────────────────────────────────────────────
const positions = new Float32Array(N * 3);
const targetPositions = new Float32Array(N * 3);
const colors = new Float32Array(N * 3);
const targetColors = new Float32Array(N * 3);
const sizes = new Float32Array(N);

// Initialize with hidden_pca coordinates
const currentSpace = { value: 'hidden_pca' };
POINTS.forEach((p, i) => {
  const c = p.hidden_pca;
  positions[i * 3] = c[0]; positions[i * 3 + 1] = c[1]; positions[i * 3 + 2] = c[2];
  targetPositions[i * 3] = c[0]; targetPositions[i * 3 + 1] = c[1]; targetPositions[i * 3 + 2] = c[2];
  sizes[i] = 6.0;
});

const geometry = new THREE.BufferGeometry();
geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

// Create soft circular sprite texture for particles
const spriteCanvas = document.createElement('canvas');
spriteCanvas.width = 64; spriteCanvas.height = 64;
const sCtx = spriteCanvas.getContext('2d');
const grad = sCtx.createRadialGradient(32, 32, 0, 32, 32, 32);
grad.addColorStop(0.0, 'rgba(255,255,255,1.0)');
grad.addColorStop(0.35, 'rgba(255,255,255,0.95)');
grad.addColorStop(0.6, 'rgba(255,255,255,0.5)');
grad.addColorStop(0.85, 'rgba(255,255,255,0.15)');
grad.addColorStop(1.0, 'rgba(255,255,255,0.0)');
sCtx.fillStyle = grad;
sCtx.fillRect(0, 0, 64, 64);
const spriteTexture = new THREE.CanvasTexture(spriteCanvas);

const COLOR_INTENSITY = 1.0;  // Full intensity — NormalBlending won't wash out

const particleMaterial = new THREE.PointsMaterial({
  size: 1.0,
  map: spriteTexture,
  vertexColors: true,
  sizeAttenuation: true,
  transparent: true,
  alphaTest: 0.5,
  depthWrite: true,
  blending: THREE.NormalBlending,
  opacity: 0.9,
});

const particles = new THREE.Points(geometry, particleMaterial);
scene.add(particles);

// ─── Persona trajectory lines ────────────────────────────────────────────────
const lineGroup = new THREE.Group();
scene.add(lineGroup);

function buildTrajectoryLines(spaceKey) {
  // Clear previous
  while (lineGroup.children.length) {
    const c = lineGroup.children[0];
    c.geometry.dispose();
    c.material.dispose();
    lineGroup.remove(c);
  }

  // Group points by persona
  const byPersona = {};
  POINTS.forEach((p, i) => {
    if (!byPersona[p.persona_id]) byPersona[p.persona_id] = [];
    byPersona[p.persona_id].push({ idx: i, t: p.t_index, coords: p[spaceKey] });
  });

  for (const pid of Object.keys(byPersona)) {
    const entries = byPersona[pid].sort((a, b) => a.t - b.t);
    if (entries.length < 2) continue;

    const linePoints = entries.map(e => new THREE.Vector3(e.coords[0], e.coords[1], e.coords[2]));
    const lineGeo = new THREE.BufferGeometry().setFromPoints(linePoints);
    const pc = PERSONA_COLORS[pid] || [0.5, 0.5, 0.5];
    const lineMat = new THREE.LineBasicMaterial({
      color: new THREE.Color(pc[0], pc[1], pc[2]),
      transparent: true,
      opacity: 0.12,
      blending: THREE.NormalBlending,
      depthWrite: false,
    });
    lineGroup.add(new THREE.Line(lineGeo, lineMat));
  }
}

buildTrajectoryLines('hidden_pca');
lineGroup.visible = false;  // Off by default

// ─── Coloring ────────────────────────────────────────────────────────────────
let currentColorMode = 'split';
let currentDimIndex = 0;

function applyColors(mode, dimIdx) {
  currentColorMode = mode;
  currentDimIndex = dimIdx;

  for (let i = 0; i < N; i++) {
    let c;
    if (mode === 'split') {
      c = SPLIT_COLORS[POINTS[i].split] || [0.5, 0.5, 0.5];
    } else if (mode === 'prediction') {
      // Color by prediction for the selected dimension
      const pred = POINTS[i].predictions[dimIdx];
      c = PRED_COLORS[String(pred)] || PRED_COLORS['0'];
    } else if (mode === 'ground_truth') {
      const gt = POINTS[i].ground_truth[dimIdx];
      c = PRED_COLORS[String(Math.sign(gt))] || PRED_COLORS['0'];
    } else if (mode === 'persona') {
      c = PERSONA_COLORS[POINTS[i].persona_id] || [0.5, 0.5, 0.5];
    } else if (mode === 'uncertainty') {
      // Mean uncertainty across dimensions → heat color
      const meanU = POINTS[i].uncertainty.reduce((a, b) => a + b, 0) / POINTS[i].uncertainty.length;
      const maxEntropy = Math.log(3); // Max entropy for 3 classes
      const t = Math.min(meanU / maxEntropy, 1.0);
      // Blue (certain) → Yellow (uncertain)
      c = [t * 1.0, t * 0.85 + (1 - t) * 0.3, (1 - t) * 1.0];
    } else if (mode === 'dimension') {
      // Color by specific dimension prediction
      const pred = POINTS[i].predictions[dimIdx];
      c = PRED_COLORS[String(pred)] || PRED_COLORS['0'];
    } else {
      c = [0.5, 0.5, 0.7];
    }
    targetColors[i * 3] = c[0] * COLOR_INTENSITY;
    targetColors[i * 3 + 1] = c[1] * COLOR_INTENSITY;
    targetColors[i * 3 + 2] = c[2] * COLOR_INTENSITY;
  }

  // Snap colors immediately for instant feedback
  for (let i = 0; i < N * 3; i++) { colors[i] = targetColors[i]; }
  geometry.attributes.color.needsUpdate = true;

  updateLegend(mode, dimIdx);
}

// ─── Legend ───────────────────────────────────────────────────────────────────
function updateLegend(mode, dimIdx) {
  const el = document.getElementById('legendItems');
  let html = '';

  const dot = (r, g, b, label) =>
    `<div class="legend-item"><div class="legend-dot" style="background:rgb(${r*255|0},${g*255|0},${b*255|0});color:rgb(${r*255|0},${g*255|0},${b*255|0})"></div>${label}</div>`;

  if (mode === 'split') {
    html += dot(...SPLIT_COLORS.train, `Train (${POINTS.filter(p=>p.split==='train').length})`);
    html += dot(...SPLIT_COLORS.val, `Val (${POINTS.filter(p=>p.split==='val').length})`);
    html += dot(...SPLIT_COLORS.test, `Test (${POINTS.filter(p=>p.split==='test').length})`);
  } else if (mode === 'prediction' || mode === 'ground_truth' || mode === 'dimension') {
    const dimName = DIMS[dimIdx] ? titleCase(DIMS[dimIdx]) : '';
    const label = mode + (dimName ? ' — ' + dimName : '');
    html += `<div style="font-size:10px;color:#7777aa;margin-bottom:6px">${label}</div>`;
    html += dot(...PRED_COLORS['-1'], 'Misaligned (-1)');
    html += dot(...PRED_COLORS['0'], 'Neutral (0)');
    html += dot(...PRED_COLORS['1'], 'Aligned (+1)');
  } else if (mode === 'persona') {
    html += `<div style="font-size:10px;color:#7777aa">${personaIds.length} personas</div>`;
    html += `<div style="font-size:10px;color:#5555aa;margin-top:4px">Golden-ratio hue spacing</div>`;
  } else if (mode === 'uncertainty') {
    html += dot(0.0, 0.3, 1.0, 'Low uncertainty');
    html += dot(0.5, 0.58, 0.5, 'Medium');
    html += dot(1.0, 0.85, 0.0, 'High uncertainty');
  }

  el.innerHTML = html;
}

// ─── Space switching ─────────────────────────────────────────────────────────
function setSpace(spaceKey) {
  currentSpace.value = spaceKey;
  for (let i = 0; i < N; i++) {
    const c = POINTS[i][spaceKey];
    targetPositions[i * 3] = c[0];
    targetPositions[i * 3 + 1] = c[1];
    targetPositions[i * 3 + 2] = c[2];
  }
  buildTrajectoryLines(spaceKey);
}

// ─── Raycasting ──────────────────────────────────────────────────────────────
const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 0.4;
const mouse = new THREE.Vector2();
let hoveredIdx = -1;
let pinnedIdx = -1;

function onMouseMove(e) {
  mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObject(particles);

  const tooltip = document.getElementById('tooltip');

  if (intersects.length > 0) {
    hoveredIdx = intersects[0].index;
    const p = POINTS[hoveredIdx];

    tooltip.innerHTML = `<strong>${p.persona_id}</strong> — Entry ${p.t_index}<br><span style="color:#8888aa">${p.date} · ${p.split}</span>`;
    tooltip.style.left = (e.clientX + 14) + 'px';
    tooltip.style.top = (e.clientY - 14) + 'px';
    tooltip.classList.add('visible');

    document.body.style.cursor = 'pointer';
    renderer.domElement.style.cursor = 'pointer';
  } else {
    hoveredIdx = -1;
    tooltip.classList.remove('visible');
    document.body.style.cursor = 'default';
    renderer.domElement.style.cursor = 'grab';
  }
}

function onClick() {
  if (hoveredIdx >= 0) {
    pinnedIdx = hoveredIdx;
    showDetail(pinnedIdx);
  } else {
    pinnedIdx = -1;
    document.getElementById('detail').classList.remove('visible');
  }
}

function showDetail(idx) {
  const p = POINTS[idx];
  const detail = document.getElementById('detail');
  const content = document.getElementById('detailContent');

  let predHtml = '<div class="predictions-grid">';
  DIMS.forEach((dim, di) => {
    const pred = p.predictions[di];
    const gt = p.ground_truth[di];
    const cls = pred < 0 ? 'neg' : pred > 0 ? 'pos' : 'neu';
    const gtCls = gt < 0 ? 'neg' : gt > 0 ? 'pos' : 'neu';
    const dimLabel = titleCase(dim);
    predHtml += `<div class="pred-item"><span>${dimLabel}</span><span class="pred-val ${cls}">${pred > 0 ? '+' : ''}${pred}</span><span class="gt-val ${gtCls}">(${gt > 0 ? '+' : ''}${gt})</span></div>`;
  });
  predHtml += '</div>';

  const meanU = (p.uncertainty.reduce((a, b) => a + b, 0) / p.uncertainty.length).toFixed(3);

  content.innerHTML = `
    <div class="field"><span class="field-label">Persona</span><br>${p.persona_id} — Entry ${p.t_index}</div>
    <div class="field"><span class="field-label">Date</span><br>${p.date} · <em>${p.split}</em></div>
    <div class="field"><span class="field-label">Journal Entry</span><div class="text-snippet">${p.text}</div></div>
    <div class="field"><span class="field-label">Predictions (ground truth)</span>${predHtml}</div>
    <div class="field"><span class="field-label">Mean Uncertainty</span><br>${meanU}</div>
  `;
  detail.classList.add('visible');
}

renderer.domElement.addEventListener('mousemove', onMouseMove);
renderer.domElement.addEventListener('click', onClick);

// ─── UI event handlers ──────────────────────────────────────────────────────
// Populate dimension selector
// Title-case helper: "self_direction" → "Self Direction"
function titleCase(s) {
  return s.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

const dimSelect = document.getElementById('dimSelect');
DIMS.forEach((dim, i) => {
  const opt = document.createElement('option');
  opt.value = i;
  opt.textContent = titleCase(dim);
  dimSelect.appendChild(opt);
});

document.getElementById('spaceSelect').addEventListener('change', e => setSpace(e.target.value));

document.getElementById('colorSelect').addEventListener('change', e => {
  const mode = e.target.value;
  // Enable dimension selector for per-dimension, prediction, and ground_truth modes
  const dimEnabled = (mode === 'dimension' || mode === 'prediction' || mode === 'ground_truth');
  dimSelect.disabled = !dimEnabled;

  if (mode === 'dimension') {
    applyColors('dimension', parseInt(dimSelect.value));
  } else if (mode === 'prediction') {
    applyColors('prediction', parseInt(dimSelect.value));
  } else if (mode === 'ground_truth') {
    applyColors('ground_truth', parseInt(dimSelect.value));
  } else {
    applyColors(mode, currentDimIndex);
  }
});

dimSelect.addEventListener('change', e => {
  const dimIdx = parseInt(e.target.value);
  if (currentColorMode === 'dimension' || currentColorMode === 'prediction' || currentColorMode === 'ground_truth') {
    applyColors(currentColorMode, dimIdx);
  }
});

document.getElementById('bloomSlider').addEventListener('input', e => {
  bloomPass.strength = parseFloat(e.target.value);
});

document.getElementById('lineSelect').addEventListener('change', e => {
  lineGroup.visible = e.target.value === 'on';
});

// Dimension selector starts disabled — enabled when prediction/ground_truth/dimension is chosen
dimSelect.disabled = true;

// Stats
document.getElementById('stats').innerHTML = `
  ${N} entries · ${personaIds.length} personas<br>
  Model: ${DATA.stats.model_variant}<br>
  Hidden dim: ${DATA.stats.hidden_dim}
`;

// ─── Animation loop ──────────────────────────────────────────────────────────
const LERP_SPEED = 0.06;

function animate() {
  requestAnimationFrame(animate);

  // Smooth position transitions
  let posChanged = false;
  for (let i = 0; i < N * 3; i++) {
    const diff = targetPositions[i] - positions[i];
    if (Math.abs(diff) > 0.001) {
      positions[i] += diff * LERP_SPEED;
      posChanged = true;
    }
  }
  if (posChanged) geometry.attributes.position.needsUpdate = true;

  // Smooth color transitions
  let colChanged = false;
  for (let i = 0; i < N * 3; i++) {
    const diff = targetColors[i] - colors[i];
    if (Math.abs(diff) > 0.001) {
      colors[i] += diff * LERP_SPEED;
      colChanged = true;
    }
  }
  if (colChanged) geometry.attributes.color.needsUpdate = true;

  // Hover highlight: update material size is not per-point with PointsMaterial
  // Per-point hover handled via raycaster visual feedback instead

  // Update trajectory line positions during transitions
  if (posChanged) {
    lineGroup.children.forEach(line => {
      // Lines use their own geometry; they animate via buildTrajectoryLines on space switch
    });
  }

  controls.update();
  composer.render();
}

// Position legend below controls
function positionLegend() {
  const ctrl = document.getElementById('controls');
  const legend = document.getElementById('legend');
  const ctrlRect = ctrl.getBoundingClientRect();
  legend.style.top = (ctrlRect.bottom + 12) + 'px';
}
positionLegend();
window.addEventListener('resize', positionLegend);

// Initialize colors
applyColors('split', 0);
animate();

// ─── Resize handling ─────────────────────────────────────────────────────────
window.addEventListener('resize', () => {
  const w = window.innerWidth, h = window.innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  composer.setSize(w, h);
  bloomPass.resolution.set(w, h);
});

// ─── Keyboard shortcuts ──────────────────────────────────────────────────────
window.addEventListener('keydown', e => {
  if (e.key === 'r' || e.key === 'R') {
    controls.autoRotate = !controls.autoRotate;
  }
  if (e.key === 'Escape') {
    pinnedIdx = -1;
    document.getElementById('detail').classList.remove('visible');
  }
});
</script>
</body>
</html>
"""


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings and generate 3D visualization")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="logs/experiments/artifacts/ordinal_v4_s2025_m33_20260310_230626/BalancedSoftmax/selected_checkpoint.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="viz/embedding_explorer.html",
        help="Output HTML file path",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser after generation",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity",
    )
    args = parser.parse_args()

    print("Loading model checkpoint...")
    model, cp = _load_model(args.checkpoint)
    training_config = cp["training_config"]

    print("Building datasets (encoding text embeddings)...")
    train_ds, val_ds, test_ds, state_encoder = _build_datasets(training_config)

    print(f"Extracting activations: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    points = []
    for ds, name in [(train_ds, "train"), (val_ds, "val"), (test_ds, "test")]:
        points.extend(_extract_from_dataset(model, ds, name))

    print(f"Total points: {len(points)}")

    # Dimensionality reduction
    print("Running PCA on hidden activations (64 → 3)...")
    hidden_pca = _reduce_to_3d(points, "hidden", method="pca")

    print("Running t-SNE on hidden activations (64 → 3)...")
    hidden_tsne = _reduce_to_3d(points, "hidden", method="tsne", perplexity=args.perplexity)

    print("Running PCA on SBERT embeddings (256 → 3)...")
    sbert_pca = _reduce_to_3d(points, "sbert", method="pca")

    print("Running t-SNE on SBERT embeddings (256 → 3)...")
    sbert_tsne = _reduce_to_3d(points, "sbert", method="tsne", perplexity=args.perplexity)

    # Attach 3D coordinates to points and drop raw vectors
    for i, p in enumerate(points):
        p["hidden_pca"] = hidden_pca[i].tolist()
        p["hidden_tsne"] = hidden_tsne[i].tolist()
        p["sbert_pca"] = sbert_pca[i].tolist()
        p["sbert_tsne"] = sbert_tsne[i].tolist()
        del p["hidden"]
        del p["sbert"]

    from src.models.judge import SCHWARTZ_VALUE_ORDER

    data = {
        "points": points,
        "dimensions": list(SCHWARTZ_VALUE_ORDER),
        "stats": {
            "n_points": len(points),
            "n_personas": len(set(p["persona_id"] for p in points)),
            "model_variant": cp["model_config"].get("variant", "unknown"),
            "hidden_dim": cp["model_config"].get("hidden_dim", "?"),
            "epoch": cp.get("epoch", "?"),
        },
    }

    output_path = Path(args.output)
    print("Generating HTML visualization...")
    _generate_html(data, output_path)

    if not args.no_browser:
        abs_path = output_path.resolve()
        print(f"Opening in browser: file://{abs_path}")
        webbrowser.open(f"file://{abs_path}")


if __name__ == "__main__":
    main()
