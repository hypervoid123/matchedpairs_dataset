"""
TAAM Local Validator — Apple M5 Pro Edition (fixed)
====================================================
Optimized for: 18-core CPU | 20-core GPU | 48GB unified RAM
Acceleration:  MPS (Metal) for saliency in main process
               multiprocessing (spawn) for batch TAAM refinement
Viewer:        self-contained HTML slider comparison page

SETUP (run once in your PyCharm terminal):
------------------------------------------
    pip install opencv-python numpy pillow tqdm
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install transformers accelerate

    # MPS (Metal GPU) is bundled with torch on Apple Silicon — no extra install.
    # To verify MPS works: python -c "import torch; print(torch.backends.mps.is_available())"

USAGE:
    python taam_m5_validator.py
    # Then open: ./taam_validation_output/viewer.html in Safari or Chrome
    # Keyboard: ← → to navigate samples | drag slider to compare masks
"""

from __future__ import annotations

import base64
import json
import logging
import math
import multiprocessing as mp
import random
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw

# ============================================================
# CONFIG — edit these before running
# ============================================================
DATA_DIR     = "./SynthScars"
SPLIT        = "train"
OUTPUT_DIR   = "./taam_validation_output"
SAMPLE_SIZE  = 100        # how many records to process locally (keep <=200)
RANDOM_SEED  = 42         # for reproducible SRS sampling
USE_SALIENCY = True       # set False to skip saliency (faster, rho_subject=0)
N_WORKERS    = 14         # CPU cores for batch refinement (leave 4 for system)
THUMB_SIZE   = (512, 512) # image size embedded in HTML viewer
# ============================================================

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TAAM Config
# ---------------------------------------------------------------------------

@dataclass
class TAAMConfig:
    # Dilation (Function I)
    r_max:  float = 15.0
    r_min:  float = 2.0
    rho_c:  float = 0.15
    tau_r:  float = 0.04   # overridden by compute_tau_r() at runtime
    beta:   float = 0.70

    # Closing (Function II)
    c_max:  float = 25.0
    c_min:  float = 3.0
    alpha:  float = 2.0
    n_ref:  int   = 30

    # Feathering (Function III)
    gamma:  float = 1.8

    # OpenCV kernel shapes
    dilation_shape: int = cv2.MORPH_ELLIPSE
    closing_shape:  int = cv2.MORPH_ELLIPSE
    max_kernel_px:  int = 51


def compute_tau_r(rho_values: list[float]) -> float:
    """
    Derive tau_r from the empirical coverage distribution (proof §7):
        tau_r = std(rho) / 1.833
    The 1.833 constant is the logistic distribution's 95% quantile in
    scale units — analogous to 1.96 for the normal distribution.
    """
    if len(rho_values) < 2:
        return 0.04
    sigma_rho = float(np.std(rho_values))
    tau_r = sigma_rho / 1.833
    logger.info(
        f"Derived  tau_r = {tau_r:.4f}  "
        f"from  sigma_rho = {sigma_rho:.4f}  "
        f"over {len(rho_values)} masks"
    )
    return tau_r


# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------

@dataclass
class MaskDescriptors:
    rho:         float
    S_bar:       float
    n:           int
    rho_subject: float


def compute_descriptors(
    mask: np.ndarray,
    saliency: Optional[np.ndarray] = None,
) -> MaskDescriptors:
    """
    Compute the four TAAM descriptors from a binary uint8 mask.
    Uses cv2.connectedComponentsWithStats — single call, no file splitting
    required even for multi-region annotations.
    """
    H, W = mask.shape
    binary = (mask > 127).astype(np.uint8)
    area = int(binary.sum())

    if area == 0:
        return MaskDescriptors(rho=0.0, S_bar=1.0, n=1, rho_subject=0.0)

    rho = area / (H * W)

    # Per-component solidity — distinguishes spiderweb (low S_bar, n=1)
    # from constellation (high S_bar, large n)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    n = n_labels - 1  # exclude background label

    solidities = []
    for lid in range(1, n_labels):
        comp_area = int(stats[lid, cv2.CC_STAT_AREA])
        if comp_area < 3:
            solidities.append(1.0)
            continue
        comp = (labels == lid).astype(np.uint8)
        contours, _ = cv2.findContours(
            comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            solidities.append(1.0)
            continue
        hull = cv2.convexHull(np.vstack(contours))
        hull_area = cv2.contourArea(hull)
        solidities.append(
            float(np.clip(comp_area / hull_area, 1e-6, 1.0))
            if hull_area >= 1.0 else 1.0
        )

    S_bar = float(np.mean(solidities)) if solidities else 1.0

    # Subject-weighted coverage — uses saliency map passed from main process
    if saliency is not None and saliency.shape == (H, W):
        rho_subject = float(
            np.clip(
                (binary.astype(np.float32) * saliency).sum() / area,
                0.0, 1.0
            )
        )
    else:
        rho_subject = 0.0

    return MaskDescriptors(rho=rho, S_bar=S_bar, n=n, rho_subject=rho_subject)


# ---------------------------------------------------------------------------
# TAAM parameter functions
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def compute_r(desc: MaskDescriptors, cfg: TAAMConfig) -> int:
    """
    r(rho, rho_subject) = r_min + (r_max - r_min) * sigmoid(-(rho - rho_c*) / tau_r)
    rho_c* = rho_c * (1 - beta * rho_subject)
    """
    rho_c_star = cfg.rho_c * (1.0 - cfg.beta * desc.rho_subject)
    val = _sigmoid(-(desc.rho - rho_c_star) / cfg.tau_r)
    r = cfg.r_min + (cfg.r_max - cfg.r_min) * val
    r_int = max(1, min(int(round(r)), cfg.max_kernel_px))
    return r_int if r_int % 2 == 1 else r_int + 1


def compute_c(desc: MaskDescriptors, cfg: TAAMConfig) -> int:
    """
    c(S_bar, n) = c_min + (c_max - c_min) * max(shape_term, count_term)
    shape_term  = (1 - S_bar)^alpha
    count_term  = clip(n / n_ref, 0, 1)
    max() selects the dominant fragmentation mode without double-counting.
    """
    shape_term = (1.0 - desc.S_bar) ** cfg.alpha
    count_term = float(np.clip(desc.n / cfg.n_ref, 0.0, 1.0))
    t = max(shape_term, count_term)
    c = cfg.c_min + (cfg.c_max - cfg.c_min) * t
    c_int = max(1, min(int(round(c)), cfg.max_kernel_px))
    return c_int if c_int % 2 == 1 else c_int + 1


def compute_sigma(r: int, rho: float, cfg: TAAMConfig) -> float:
    """
    sigma(r, rho) = (r / 3) * exp(-gamma * rho)
    3-sigma constraint (3*sigma <= r) is satisfied automatically since
    exp(-gamma*rho) <= 1 for all valid inputs.
    """
    return max(0.5, (r / 3.0) * math.exp(-cfg.gamma * rho))


# ---------------------------------------------------------------------------
# Full TAAM refinement
# ---------------------------------------------------------------------------

def refine_mask(
    mask: np.ndarray,
    cfg: TAAMConfig,
    saliency: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, dict, MaskDescriptors]:
    """
    Full TAAM pipeline: Close → Dilate → GaussianBlur
    Order is mathematically necessary — see proof §6.2.
    Returns float32 mask in [0, 1].
    """
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = mask.astype(np.uint8)

    desc  = compute_descriptors(mask, saliency)
    r     = compute_r(desc, cfg)
    c     = compute_c(desc, cfg)
    sigma = compute_sigma(r, desc.rho, cfg)

    ck = cv2.getStructuringElement(cfg.closing_shape,  (c, c))
    dk = cv2.getStructuringElement(cfg.dilation_shape, (r, r))

    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ck)
    m = cv2.dilate(m, dk)
    ksize = max(3, int(2 * math.ceil(3 * sigma) + 1))
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    m = cv2.GaussianBlur(m.astype(np.float32), (ksize, ksize), sigma)

    return (
        np.clip(m / 255.0, 0.0, 1.0),
        {
            "rho":         round(desc.rho, 4),
            "S_bar":       round(desc.S_bar, 4),
            "n":           desc.n,
            "rho_subject": round(desc.rho_subject, 4),
            "r_star":      r,
            "c_star":      c,
            "sigma_star":  round(sigma, 3),
        },
        desc,
    )


# ---------------------------------------------------------------------------
# Saliency detector — runs ONLY in main process
#
# FIX (Bug 1 + Bug 2): MPS/Metal cannot be used inside spawned subprocess
# on macOS. The model must stay in the main process. We run saliency
# sequentially here, then pass the resulting numpy arrays through the
# multiprocessing args tuple to workers. Workers only do CPU morphology.
# ---------------------------------------------------------------------------

class SaliencyDetector:
    """
    RMBG-1.4 salient object detector running on Apple MPS (Metal GPU).
    Produces float32 saliency maps in [0, 1] for subject-awareness.

    Lives in the main process only — never instantiated in workers.
    """

    def __init__(self):
        self._model = None
        self.device = None

        if not USE_SALIENCY:
            logger.info("USE_SALIENCY=False — rho_subject will be 0 for all samples.")
            return

        try:
            import torch
            from transformers import AutoModelForImageSegmentation

            # MPS = Metal Performance Shaders — Apple's GPU compute for ML
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logger.info("Saliency: using Metal GPU (MPS) — unified memory, no copy overhead.")
            else:
                self.device = torch.device("cpu")
                logger.warning("MPS not available — saliency running on CPU.")

            self._model = AutoModelForImageSegmentation.from_pretrained(
                "briaai/RMBG-1.4", trust_remote_code=True
            ).to(self.device)
            self._model.eval()
            logger.info("RMBG-1.4 loaded successfully.")

        except Exception as e:
            logger.warning(
                f"Saliency model failed to load: {e}\n"
                f"rho_subject will be 0 for all samples. "
                f"Check: pip install transformers accelerate"
            )

    def predict(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Returns float32 (H, W) saliency map in [0, 1], or None.
        Called sequentially in main process before worker dispatch.
        """
        if self._model is None:
            return None

        import torch
        import torch.nn.functional as F
        from torchvision.transforms.functional import normalize

        H, W = image_bgr.shape[:2]

        # Preprocess: BGR → RGB, resize to RMBG's native 1024x1024, normalize
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (1024, 1024))
        tensor = (
            torch.tensor(img_resized, dtype=torch.float32)
            .permute(2, 0, 1) / 255.0
        )
        tensor = normalize(tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds = self._model(tensor)
            # RMBG returns a list of outputs — primary mask is preds[0][0]
            mask = preds[0][0]
            mask = F.interpolate(
                mask.unsqueeze(0), size=(H, W),
                mode="bilinear", align_corners=False
            )
            mask = mask.squeeze().cpu().numpy()

        # Normalise to strict [0, 1]
        lo, hi = mask.min(), mask.max()
        return ((mask - lo) / (hi - lo + 1e-8)).astype(np.float32)


# ---------------------------------------------------------------------------
# SynthScars loading (identical logic to your notebook)
# ---------------------------------------------------------------------------

def load_records(
    data_dir: str, split: str, start_index: int, end_index: int
) -> list[dict]:
    ann_path = Path(data_dir) / split / "annotations" / f"{split}.json"
    with open(ann_path) as f:
        raw = json.load(f)

    records = []
    for entry in raw:
        for key, sample in entry.items():
            img_path = str(
                Path(data_dir) / split / "images" / sample["img_file_name"]
            )
            refs = sample.get("refs", [])
            segs, explanation = [], ""
            for ref in refs:
                if ref.get("explanation"):
                    explanation = ref["explanation"]
                for seg in ref.get("segmentation", []):
                    if seg:
                        segs.append(seg)
            records.append({
                "id": int(key),
                "img_path": img_path,
                "explanation": explanation,
                "segmentations": segs,
            })

    return (
        records[start_index:]
        if end_index == -1
        else records[start_index:end_index]
    )


def rasterize_mask(
    segmentations: list[list[float]], size: tuple[int, int]
) -> np.ndarray:
    pil_mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(pil_mask)
    for flat in segmentations:
        if len(flat) < 6:
            continue
        pairs = [(flat[i], flat[i + 1]) for i in range(0, len(flat) - 1, 2)]
        draw.polygon(pairs, fill=255)
    return np.array(pil_mask, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Per-sample worker
#
# FIX (Bug 1 + Bug 2): saliency is now a numpy array passed through the
# args tuple from the main process — not computed here. Workers do only
# CPU morphology (OpenCV), which is safe in spawned subprocesses.
# ---------------------------------------------------------------------------

def _process_one(args: tuple) -> Optional[dict]:
    """
    Worker function for multiprocessing.Pool.
    Receives a pre-computed saliency array from main process.
    Does TAAM refinement + base64 encoding for the HTML viewer.
    """
    rec, cfg_dict, thumb_size, saliency = args  # saliency: np.ndarray or None

    cfg = TAAMConfig(**cfg_dict)
    id_str = f"{rec['id']:05d}"

    try:
        img_path = Path(rec["img_path"])
        if not img_path.exists():
            return None

        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            return None

        H_img, W_img = image_bgr.shape[:2]
        mask_raw = rasterize_mask(rec["segmentations"], (W_img, H_img))

        if mask_raw.max() == 0:
            return None

        # TAAM refinement — saliency array passed in from main process
        refined_f32, params, _ = refine_mask(mask_raw, cfg, saliency=saliency)
        refined_u8 = (refined_f32 * 255).astype(np.uint8)

        def to_b64(arr_bgr: np.ndarray) -> str:
            pil = Image.fromarray(cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB))
            pil.thumbnail(thumb_size, Image.LANCZOS)
            buf = BytesIO()
            pil.save(buf, format="JPEG", quality=88)
            return base64.b64encode(buf.getvalue()).decode()

        def mask_to_b64(arr: np.ndarray) -> str:
            pil = Image.fromarray(arr).convert("RGB")
            pil.thumbnail(thumb_size, Image.LANCZOS)
            buf = BytesIO()
            pil.save(buf, format="JPEG", quality=88)
            return base64.b64encode(buf.getvalue()).decode()

        def refined_to_b64(arr: np.ndarray) -> str:
            heat = cv2.applyColorMap(arr, cv2.COLORMAP_INFERNO)
            pil = Image.fromarray(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB))
            pil.thumbnail(thumb_size, Image.LANCZOS)
            buf = BytesIO()
            pil.save(buf, format="JPEG", quality=88)
            return base64.b64encode(buf.getvalue()).decode()

        return {
            "id":            id_str,
            "explanation":   rec.get("explanation", "")[:120],
            "params":        params,
            "img_b64":       to_b64(image_bgr),
            "orig_mask_b64": mask_to_b64(mask_raw),
            "taam_mask_b64": refined_to_b64(refined_u8),
        }

    except Exception:
        return None


# ---------------------------------------------------------------------------
# HTML slider viewer
# ---------------------------------------------------------------------------

def generate_html_viewer(samples: list[dict], out_path: Path) -> None:
    samples_json = json.dumps(samples)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TAAM Mask Validator</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
    background: #0f0f0f; color: #e8e8e8;
    display: flex; flex-direction: column; align-items: center;
    padding: 24px 16px; min-height: 100vh;
  }}
  h1 {{ font-size: 20px; font-weight: 500; margin-bottom: 6px; color: #fff; }}
  .subtitle {{ font-size: 13px; color: #666; margin-bottom: 24px; }}
  .card {{
    background: #1a1a1a; border: 0.5px solid #2a2a2a;
    border-radius: 12px; padding: 20px; width: 100%; max-width: 900px;
    margin-bottom: 16px;
  }}
  .slider-wrap {{
    position: relative; width: 100%; aspect-ratio: 4/3;
    overflow: hidden; border-radius: 8px; cursor: col-resize;
    user-select: none; background: #000;
  }}
  .slider-wrap img {{
    position: absolute; top: 0; left: 0;
    width: 100%; height: 100%; object-fit: contain; pointer-events: none;
  }}
  .img-before {{ z-index: 1; }}
  .img-after  {{ z-index: 2; clip-path: inset(0 50% 0 0); }}
  .divider {{
    position: absolute; top: 0; bottom: 0; width: 2px;
    background: #fff; z-index: 3; left: 50%;
    transform: translateX(-50%); pointer-events: none;
  }}
  .handle {{
    position: absolute; top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    width: 36px; height: 36px; border-radius: 50%;
    background: #fff; z-index: 4;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.6); pointer-events: none;
  }}
  .handle svg {{ width: 18px; height: 18px; }}
  .labels {{
    display: flex; justify-content: space-between;
    font-size: 11px; color: #888; margin-top: 8px; padding: 0 2px;
  }}
  .label-left  {{ color: #f4a261; }}
  .label-right {{ color: #57cc99; }}
  .params {{
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;
    margin-top: 16px;
  }}
  .param {{
    background: #111; border-radius: 8px; padding: 10px 12px;
  }}
  .param .name  {{ font-size: 11px; color: #666; margin-bottom: 4px; }}
  .param .value {{ font-size: 18px; font-weight: 500; font-variant-numeric: tabular-nums; }}
  .param .value.r   {{ color: #57cc99; }}
  .param .value.c   {{ color: #f4a261; }}
  .param .value.rho {{ color: #74b9ff; }}
  .param .value.s   {{ color: #a29bfe; }}
  .source-wrap {{
    margin-top: 16px; border-radius: 8px; overflow: hidden;
    width: 100%; aspect-ratio: 16/9; background: #000;
  }}
  .source-wrap img {{ width: 100%; height: 100%; object-fit: contain; }}
  .src-label {{ font-size: 11px; color: #555; margin-top: 6px; }}
  .explanation {{
    font-size: 12px; color: #666; margin-top: 12px; line-height: 1.6;
    font-style: italic; border-left: 2px solid #2a2a2a; padding-left: 10px;
  }}
  .nav {{
    display: flex; align-items: center; gap: 12px;
    width: 100%; max-width: 900px; justify-content: center;
    margin-bottom: 16px;
  }}
  .nav button {{
    padding: 8px 20px; border-radius: 8px; border: 0.5px solid #333;
    background: #1a1a1a; color: #e8e8e8; cursor: pointer; font-size: 13px;
  }}
  .nav button:hover {{ background: #2a2a2a; }}
  .nav button:disabled {{ opacity: 0.3; cursor: default; }}
  .counter {{ font-size: 13px; color: #666; min-width: 80px; text-align: center; }}
</style>
</head>
<body>
<h1>TAAM Mask Validator</h1>
<p class="subtitle">Drag the slider to compare original annotation mask vs TAAM-refined mask</p>
<div class="nav">
  <button id="btn-prev" onclick="navigate(-1)">← Prev</button>
  <span class="counter" id="counter"></span>
  <button id="btn-next" onclick="navigate(1)">Next →</button>
</div>
<div class="card">
  <div class="slider-wrap" id="slider-wrap"
       onmousedown="startDrag(event)" ontouchstart="startTouch(event)">
    <img class="img-before" id="img-before" src="" alt="original mask">
    <img class="img-after"  id="img-after"  src="" alt="TAAM mask">
    <div class="divider" id="divider"></div>
    <div class="handle" id="handle">
      <svg viewBox="0 0 24 24" fill="none" stroke="#333" stroke-width="2.5">
        <path d="M8 5l-5 7 5 7M16 5l5 7-5 7"/>
      </svg>
    </div>
  </div>
  <div class="labels">
    <span class="label-left">◀ Original annotation mask</span>
    <span class="label-right">TAAM refined mask ▶</span>
  </div>
  <div class="params">
    <div class="param"><div class="name">Dilation r*</div><div class="value r"   id="p-r">—</div></div>
    <div class="param"><div class="name">Closing c*</div><div class="value c"   id="p-c">—</div></div>
    <div class="param"><div class="name">Coverage ρ</div><div class="value rho" id="p-rho">—</div></div>
    <div class="param"><div class="name">Solidity S̄</div><div class="value s"  id="p-s">—</div></div>
    <div class="param"><div class="name">Components n</div><div class="value"   id="p-n">—</div></div>
    <div class="param"><div class="name">σ* (feather)</div><div class="value"  id="p-sigma">—</div></div>
    <div class="param"><div class="name">ρ_subject</div><div class="value"     id="p-rsub">—</div></div>
    <div class="param"><div class="name">Sample ID</div><div class="value" style="font-size:14px" id="p-id">—</div></div>
  </div>
  <div class="source-wrap"><img id="img-source" src="" alt="source image"></div>
  <div class="src-label">Source image</div>
  <div class="explanation" id="explanation"></div>
</div>
<script>
const SAMPLES = {samples_json};
let idx = 0, pct = 50, dragging = false;

function load(i) {{
  const s = SAMPLES[i];
  document.getElementById('img-before').src = 'data:image/jpeg;base64,' + s.orig_mask_b64;
  document.getElementById('img-after').src  = 'data:image/jpeg;base64,' + s.taam_mask_b64;
  document.getElementById('img-source').src = 'data:image/jpeg;base64,' + s.img_b64;
  document.getElementById('explanation').textContent = s.explanation || '—';
  const p = s.params;
  document.getElementById('p-r').textContent     = p.r_star + 'px';
  document.getElementById('p-c').textContent     = p.c_star + 'px';
  document.getElementById('p-rho').textContent   = p.rho.toFixed(3);
  document.getElementById('p-s').textContent     = p.S_bar.toFixed(3);
  document.getElementById('p-n').textContent     = p.n;
  document.getElementById('p-sigma').textContent = p.sigma_star.toFixed(2) + 'px';
  document.getElementById('p-rsub').textContent  = p.rho_subject.toFixed(3);
  document.getElementById('p-id').textContent    = s.id;
  document.getElementById('counter').textContent = `${{i+1}} / ${{SAMPLES.length}}`;
  document.getElementById('btn-prev').disabled = i === 0;
  document.getElementById('btn-next').disabled = i === SAMPLES.length - 1;
  setSplit(pct);
}}

function navigate(dir) {{
  idx = Math.max(0, Math.min(SAMPLES.length - 1, idx + dir));
  load(idx);
}}

function setSplit(p) {{
  pct = Math.max(2, Math.min(98, p));
  document.getElementById('img-after').style.clipPath = `inset(0 ${{100 - pct}}% 0 0)`;
  document.getElementById('divider').style.left = pct + '%';
  document.getElementById('handle').style.left  = pct + '%';
}}

function startDrag(e) {{ dragging = true; e.preventDefault(); }}
function startTouch(e) {{ dragging = true; }}

document.addEventListener('mousemove', e => {{
  if (!dragging) return;
  const r = document.getElementById('slider-wrap').getBoundingClientRect();
  setSplit(((e.clientX - r.left) / r.width) * 100);
}});
document.addEventListener('mouseup', () => dragging = false);
document.addEventListener('touchmove', e => {{
  if (!dragging) return;
  const r = document.getElementById('slider-wrap').getBoundingClientRect();
  setSplit(((e.touches[0].clientX - r.left) / r.width) * 100);
}}, {{ passive: true }});
document.addEventListener('touchend', () => dragging = false);
document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowRight') navigate(1);
  if (e.key === 'ArrowLeft')  navigate(-1);
}});
load(0);
</script>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    logger.info(f"HTML viewer → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1 — Load and SRS sample
    # FIX (Bug 3): SAMPLE_SIZE config is now respected. main() no longer
    # hardcodes load_records(..., 0, -1) and ignores the config variable.
    logger.info("Loading SynthScars records...")
    all_records = load_records(DATA_DIR, SPLIT, 0, -1)
    random.seed(RANDOM_SEED)
    records = random.sample(all_records, min(SAMPLE_SIZE, len(all_records)))
    logger.info(f"  SRS: {len(records)} records sampled from {len(all_records)} total.")

    # Step 2 — Compute empirical tau_r
    logger.info("Computing empirical tau_r...")
    rhos = []
    for rec in records:
        try:
            img = Image.open(rec["img_path"])
            mask = rasterize_mask(rec["segmentations"], img.size)
            rho = float((mask > 127).sum() / (mask.shape[0] * mask.shape[1]))
            rhos.append(rho)
        except Exception:
            continue
    cfg = TAAMConfig()
    cfg.tau_r = compute_tau_r(rhos) if len(rhos) > 1 else 0.04

    # Step 3 — Run saliency in main process (MPS/Metal, sequential)
    # FIX (Bug 1 + Bug 2): saliency runs here, not in workers.
    # Workers receive numpy arrays — no Metal context crossing process boundary.
    logger.info("Running saliency detection in main process (Metal GPU)...")
    detector = SaliencyDetector()
    saliency_maps: dict[int, Optional[np.ndarray]] = {}

    for rec in records:
        try:
            img_bgr = cv2.imread(rec["img_path"])
            if img_bgr is not None:
                saliency_maps[rec["id"]] = detector.predict(img_bgr)
            else:
                saliency_maps[rec["id"]] = None
        except Exception as e:
            logger.warning(f"Saliency failed for id={rec['id']}: {e}")
            saliency_maps[rec["id"]] = None

    # Step 4 — Dispatch to worker pool (CPU morphology only)
    cfg_dict = {
        "r_max": cfg.r_max, "r_min": cfg.r_min, "rho_c": cfg.rho_c,
        "tau_r": cfg.tau_r, "beta":  cfg.beta,
        "c_max": cfg.c_max, "c_min": cfg.c_min, "alpha": cfg.alpha,
        "n_ref": cfg.n_ref, "gamma": cfg.gamma,
        "dilation_shape": cv2.MORPH_ELLIPSE,
        "closing_shape":  cv2.MORPH_ELLIPSE,
        "max_kernel_px":  51,
    }

    # Pack saliency into args — passed as numpy array, safe through pickle
    args = [
        (rec, cfg_dict, THUMB_SIZE, saliency_maps.get(rec["id"]))
        for rec in records
    ]

    logger.info(f"Dispatching {len(records)} samples to {N_WORKERS} CPU workers...")
    samples, skipped = [], 0

    ctx = mp.get_context("spawn")  # spawn = safe on macOS with OpenCV + ObjC
    with ctx.Pool(processes=N_WORKERS) as pool:
        for result in pool.imap_unordered(_process_one, args, chunksize=4):
            if result is not None:
                samples.append(result)
                p = result["params"]
                logger.info(
                    f"{result['id']} | rho={p['rho']:.3f} "
                    f"S̄={p['S_bar']:.3f} n={p['n']} "
                    f"ρ_subj={p['rho_subject']:.3f} → "
                    f"r*={p['r_star']}px c*={p['c_star']}px"
                )
            else:
                skipped += 1

    samples.sort(key=lambda s: s["id"])

    # Step 5 — Generate viewer and save log
    generate_html_viewer(samples, out_dir / "viewer.html")

    params_log = [
        {"id": s["id"], "explanation": s["explanation"], **s["params"]}
        for s in samples
    ]
    with open(out_dir / "taam_params_log.json", "w") as f:
        json.dump(params_log, f, indent=2)

    # Summary
    if params_log:
        r_vals   = [p["r_star"] for p in params_log]
        c_vals   = [p["c_star"] for p in params_log]
        rho_vals = [p["rho"]    for p in params_log]
        s_vals   = [p["S_bar"]  for p in params_log]

        print("\n" + "=" * 58)
        print("  TAAM M5 PRO VALIDATION SUMMARY")
        print("=" * 58)
        print(f"  Saliency    {'MPS (Metal)' if USE_SALIENCY else 'disabled'}")
        print(f"  Workers     {N_WORKERS} CPU cores (spawn)")
        print(f"  tau_r       {cfg.tau_r:.4f}  (data-derived)")
        print(f"  Processed   {len(params_log)}  |  Skipped {skipped}")
        print(f"  rho         mean={np.mean(rho_vals):.3f}  std={np.std(rho_vals):.3f}")
        print(f"  S_bar       mean={np.mean(s_vals):.3f}  std={np.std(s_vals):.3f}")
        print(f"  r*          mean={np.mean(r_vals):.1f}px  "
              f"min={min(r_vals)}px  max={max(r_vals)}px")
        print(f"  c*          mean={np.mean(c_vals):.1f}px  "
              f"min={min(c_vals)}px  max={max(c_vals)}px")
        print(f"\n  Open → {out_dir}/viewer.html")
        print(f"  (← → keys to navigate | drag slider to compare)")
        print("=" * 58)


if __name__ == "__main__":
    main()