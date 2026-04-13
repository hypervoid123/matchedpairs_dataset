#first cell: 


# INSTALL CELL
# ============================================================
!pip install --upgrade torch torchvision torchaudio "numpy<2" "Pillow>=10.0.0" "transformers>=4.40.0" "diffusers>=0.30.0" "accelerate>=0.30.0" "datasets>=2.19.0" "jinja2>=3.1.0" wandb tqdm opencv-python-headless
print("done pipping")



#second cell: 
from huggingface_hub import login
import wandb
wandb.login(key="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")

#third cell: 


# ============================================================
# MAIN SCRIPT
# ============================================================
import gc
import json
import logging
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import wandb
from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from tqdm import tqdm

# ============================================================
# CONFIG — edit these before running
# ============================================================
DATA_DIR       = "./SynthScars/SynthScars"
SPLIT          = "train"       # "train" or "test"
OUTPUT_DIR     = "./finalized_dataset" # <--- CHANGED TO NEW FOLDER
START_INDEX    = 0
END_INDEX      = -1            # -1 = process everything
SEED           = 42
MAX_RES        = 1024
MIN_RES        = 512
WANDB_API_KEY  = "redacted"
HF_TOKEN       = "redacted"
# ============================================================

# ---------------------------------------------------------------------------
# Prompt Translator
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an expert image prompt engineer. I will give you a description of an artifact or error in an image. "
    "Your job is to reverse-engineer that error and write a prompt describing the PERFECT, corrected state of that region. "
    "RULES:\n"
    "1. CONTEXT RETENTION: Extract EVERY physical object, person, or background element mentioned in the input. Do not drop any nouns.\n"
    "2. ORDER OF IMPORTANCE: Structure your sentence so the most important focal point is described first. Move secondary objects and background details to the end of the sentence.\n"
    "3. NO COMMANDS: NEVER use instruction words like 'Remove', 'Replace', 'Adjust', or 'Fix'.\n"
    "4. NO NEGATIVES: NEVER describe the error itself or use words like 'missing', 'deformed', or 'without'. Pretend the error never existed.\n"
    "Output ONLY the final, positive visual description."
)

_NEGATIVE_TERMS = frozenset([
    "missing", "without", "absent", "lack", "removed", "remove", "replace", "adjust", "lost",
    "distort", "broken", "wrong", "abnormal", "unnatural", "fused", "blend",
    "swollen", "misshapen", "deform", "truncat", "sunken", "asymmetri", "strange", "incomplete",
    "isn't", "aren't", "doesn't", "won't", "don't", "no gaps", "extra layer", "hanging over",
    "fix", "correct", "repair", "restore"
])

def _is_valid_prompt(text: str) -> bool:
    if not text or len(text.strip()) < 10:
        return False
    return not any(term in text.lower() for term in _NEGATIVE_TERMS)

class PromptTranslator:
    def __init__(
        self,
        output_dir: str,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        max_retries: int = 5,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.cache_path = self.output_dir / "prompt_cache.json"
        self.max_retries = max_retries
        self.cache: dict[str, str] = {}

        if self.cache_path.exists():
            with open(self.cache_path) as f:
                self.cache = json.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        self.model.eval()

    def _save_cache(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f)

    def _call_model(self, explanation: str, temperature: float = 0.3) -> str:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": explanation},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def translate(self, explanation: str) -> str:
        if explanation in self.cache:
            cached = self.cache[explanation]
            if _is_valid_prompt(cached):
                return cached

        last_result = ""
        for attempt in range(self.max_retries):
            temperature = 0.3 + attempt * 0.15
            last_result = self._call_model(explanation, temperature=temperature)
            if _is_valid_prompt(last_result):
                self.cache[explanation] = last_result
                self._save_cache()
                return last_result

        if last_result and len(last_result.strip()) >= 10:
            return last_result

        raise ValueError(f"Failed to produce a valid prompt after {self.max_retries} attempts. Last output: {last_result!r}")

# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------

def load_records(data_dir: str, split: str, start_index: int, end_index: int) -> list[dict[str, Any]]:
    ann_path = Path(data_dir) / split / "annotations" / f"{split}.json"
    with open(ann_path) as f:
        raw = json.load(f)

    records = []
    for entry in raw:
        for key, sample in entry.items():
            img_path = str(Path(data_dir) / split / "images" / sample["img_file_name"])
            refs = sample.get("refs", [])
            explanations = []
            segmentations = []
            for ref in refs:
                if ref.get("explanation"):
                    explanations.append(ref["explanation"])
                for seg in ref.get("segmentation", []):
                    if seg:
                        segmentations.append(seg)
            records.append({
                "id": int(key),
                "img_path": img_path,
                "explanation": " ".join(explanations),
                "segmentations": segmentations,
            })

    if end_index == -1:
        return records[start_index:]
    return records[start_index:end_index]

# ---------------------------------------------------------------------------
# Image / Mask Processing
# ---------------------------------------------------------------------------

def rasterize_mask(segmentations: list[list[float]], size: tuple[int, int]) -> Image.Image:
    pil_mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(pil_mask)
    for flat_coords in segmentations:
        if len(flat_coords) < 6:
            continue
        pairs = [(flat_coords[i], flat_coords[i + 1]) for i in range(0, len(flat_coords) - 1, 2)]
        draw.polygon(pairs, fill=255)

    arr = np.array(pil_mask, dtype=np.uint8)

    ref = max(size)
    close_r  = max(7,  ref // 100)
    dilate_r = max(5,  ref // 85)
    blur_k   = max(7,  ref // 50)
    if blur_k % 2 == 0:
        blur_k += 1

    ell = cv2.MORPH_ELLIPSE
    close_k  = cv2.getStructuringElement(ell, (close_r  * 2 + 1, close_r  * 2 + 1))
    dilate_k = cv2.getStructuringElement(ell, (dilate_r * 2 + 1, dilate_r * 2 + 1))

    arr = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, close_k)
    arr = cv2.dilate(arr, dilate_k)
    arr = cv2.GaussianBlur(arr, (blur_k, blur_k), 0)

    return Image.fromarray(arr, mode="L")

def _next_multiple_of_16(x: int) -> int:
    return ((x + 15) // 16) * 16

def preprocess(
    image: Image.Image,
    mask: Image.Image,
    max_res: int,
    min_res: int = 512,
) -> tuple[Image.Image, Image.Image, tuple[int, int], tuple[int, int]]:
    orig_dims = image.size
    W, H = orig_dims

    if max(W, H) < min_res:
        scale = min_res / max(W, H)
        new_W, new_H = int(W * scale), int(H * scale)
        image = image.resize((new_W, new_H), Image.LANCZOS)
        mask = mask.resize((new_W, new_H), Image.NEAREST)
        W, H = image.size

    if max(W, H) > max_res:
        scale = max_res / max(W, H)
        new_W, new_H = int(W * scale), int(H * scale)
        image = image.resize((new_W, new_H), Image.LANCZOS)
        mask = mask.resize((new_W, new_H), Image.NEAREST)
        W, H = image.size

    pad_W, pad_H = _next_multiple_of_16(W), _next_multiple_of_16(H)

    if pad_W != W or pad_H != H:
        offset_x, offset_y = (pad_W - W) // 2, (pad_H - H) // 2
        padded_image = Image.new("RGB", (pad_W, pad_H), (0, 0, 0))
        padded_image.paste(image, (offset_x, offset_y))
        padded_mask = Image.new("L", (pad_W, pad_H), 0)
        padded_mask.paste(mask, (offset_x, offset_y))
        image, mask = padded_image, padded_mask

    return image, mask, orig_dims, image.size

# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------

def load_flux():
    from diffusers import FluxFillPipeline
    pipe = FluxFillPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Fill-dev",
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    pipe.vae = pipe.vae.to(torch.bfloat16)
    pipe.set_progress_bar_config(disable=True)
    return pipe

def setup_dirs(output_dir: str) -> dict[str, Path]:
    root = Path(output_dir)
    dirs = {
        "root": root,
        "flawed": root / "flawed_images",
        "corrected": root / "corrected_images",
        "masks": root / "artifact_masks",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs

def setup_loggers(output_dir: str) -> tuple[logging.Logger, logging.Logger]:
    root = Path(output_dir)

    error_logger = logging.getLogger("error_logger")
    error_logger.setLevel(logging.ERROR)
    error_logger.addHandler(logging.FileHandler(root / "error.log", mode="a"))

    skip_logger = logging.getLogger("skip_logger")
    skip_logger.setLevel(logging.INFO)
    skip_logger.addHandler(logging.FileHandler(root / "skipped.log", mode="a"))

    return error_logger, skip_logger

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

dirs = setup_dirs(OUTPUT_DIR)
error_logger, skip_logger = setup_loggers(OUTPUT_DIR)

print("Initializing W&B...")
wandb.login(key=WANDB_API_KEY)
wandb.init(project="synthscars-flux-ground-truth", name="final-11k-run", config={ # <--- ADDED RUN NAME
    "data_dir": DATA_DIR, "split": SPLIT, "output_dir": OUTPUT_DIR,
    "start_index": START_INDEX, "end_index": END_INDEX,
    "seed": SEED, "max_res": MAX_RES, "min_res": MIN_RES,
})

print("Authenticating with Hugging Face...")
login(token=HF_TOKEN)

print("Loading translator (Qwen 7B)...")
translator = PromptTranslator(output_dir=OUTPUT_DIR)

print("Loading dataset...")
records = load_records(DATA_DIR, SPLIT, START_INDEX, END_INDEX)

print("Loading FLUX (this takes a minute)...")
pipe = load_flux()

processed_ids: set[str] = set()
meta_path = Path(OUTPUT_DIR) / "metadata.jsonl"
if meta_path.exists():
    with open(meta_path) as f:
        for line in f:
            try:
                processed_ids.add(json.loads(line)["id"])
            except Exception:
                pass
if processed_ids:
    print(f"Resuming: {len(processed_ids)} already processed, skipping those.")

assert len(records) > 0, "No records loaded. Check DATA_DIR and SPLIT settings."

meta_file = open(meta_path, "a")

print("\n" + "=" * 60)
print("PIPELINE FULLY INITIALIZED AND RUNNING!")
print("=" * 60 + "\n")

wandb_step = 0

try:
    for record in tqdm(records, desc="Processing"):
        id_str = f"{record['id']:05d}"

        if id_str in processed_ids:
            continue

        try:
            img_path = Path(record["img_path"])
            if not img_path.exists():
                skip_logger.info(f"{id_str} | Image not found: {img_path}")
                continue

            image = Image.open(img_path).convert("RGB")
            mask = rasterize_mask(record["segmentations"], image.size)

            if mask.getextrema()[1] == 0:
                skip_logger.info(f"{id_str} | Mask is entirely black")
                continue

            if not record["explanation"]:
                skip_logger.info(f"{id_str} | No explanation text")
                continue

            try:
                translated_prompt = translator.translate(record["explanation"])
            except Exception as e:
                skip_logger.info(f"{id_str} | Translation failed: {e}")
                continue

            proc_image, proc_mask, orig_dims, padded_dims = preprocess(image, mask, MAX_RES, MIN_RES)

            W, H = proc_image.size
            generator = torch.Generator(device="cuda").manual_seed(SEED)
            with torch.inference_mode():
                result = pipe(
                    prompt=translated_prompt,
                    image=proc_image,
                    mask_image=proc_mask,
                    height=H,
                    width=W,
                    guidance_scale=30.0,
                    num_inference_steps=50,
                    max_sequence_length=512,
                    generator=generator,
                ).images[0]

            flawed_rel   = f"flawed_images/{id_str}_flawed.png"
            corrected_rel = f"corrected_images/{id_str}_corrected.png"
            mask_rel     = f"artifact_masks/{id_str}_mask.png"

            proc_image.save(Path(OUTPUT_DIR) / flawed_rel)
            proc_mask.save(Path(OUTPUT_DIR) / mask_rel)
            result.save(Path(OUTPUT_DIR) / corrected_rel)

            tqdm.write(f"[{id_str}] {translated_prompt}")

            meta_file.write(json.dumps({
                "id": id_str,
                "original_explanation": record["explanation"],
                "translated_prompt": translated_prompt,
                "flawed_image_path": flawed_rel,
                "corrected_image_path": corrected_rel,
                "mask_path": mask_rel,
                "seed": SEED,
                "original_dimensions": list(orig_dims),
                "padded_dimensions": list(padded_dims),
            }) + "\n")
            meta_file.flush()

            log_payload: dict = {"sample_id": id_str, "translated_prompt": translated_prompt}
            if wandb_step % 10 == 0:
                log_payload["images"] = [
                    wandb.Image(proc_image, caption=f"Flawed: {record['explanation'][:80]}"),
                    wandb.Image(proc_mask, caption="Mask"),
                    wandb.Image(result, caption=f"Corrected: {translated_prompt}"),
                ]
            wandb.log(log_payload)
            wandb_step += 1

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                skip_logger.info(f"{id_str} | CUDA OOM")
            else:
                error_logger.error(f"Error on {id_str}", exc_info=True)
                skip_logger.info(f"{id_str} | RuntimeError: {e}")
            continue
        except Exception:
            error_logger.error(f"Error on {id_str}", exc_info=True)
            skip_logger.info(f"{id_str} | {traceback.format_exc().splitlines()[-1]}")
            continue
        finally:
            gc.collect()
            torch.cuda.empty_cache()

finally:
    meta_file.close()
    wandb.finish()