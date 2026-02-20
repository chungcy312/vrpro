#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    AutoTokenizer,
    SamModel,
    SamProcessor,
)


NOUN_STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "being",
    "been",
    "to",
    "for",
    "of",
    "on",
    "in",
    "at",
    "by",
    "with",
    "from",
    "and",
    "or",
    "as",
    "during",
    "while",
    "next",
    "into",
    "over",
    "under",
    "through",
    "his",
    "her",
    "their",
    "its",
    "this",
    "that",
    "these",
    "those",
    "team",
    "group",
    "people",
    "person",
}


@dataclass
class DetectionResult:
    label: str
    score: float
    box_xyxy: Tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Caption-guided per-frame subject detection + segmentation for motion-focused preprocessing"
    )
    parser.add_argument(
        "--video_dir",
        type=Path,
        default=Path("/home/hpc/ce505215/vrpro/dataset/motion_top100"),
        help="Directory that contains *.mp4 and optional paired *.txt captions",
    )
    parser.add_argument(
        "--caption_dir",
        type=Path,
        default=None,
        help="Optional caption directory; defaults to --video_dir",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/home/hpc/ce505215/vrpro/dataset/motion_top100_segmented"),
        help="Output directory for frame-level bbox/mask/overlay",
    )
    parser.add_argument("--video_glob", type=str, default="*.mp4")
    parser.add_argument(
        "--max_videos",
        type=int,
        default=0,
        help="0 means all videos",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Process one frame every N frames",
    )
    parser.add_argument(
        "--sample_interval_sec",
        type=float,
        default=0.0,
        help="If > 0, sample one frame every given seconds (overrides frame_stride)",
    )
    parser.add_argument(
        "--max_frames_per_video",
        type=int,
        default=0,
        help="0 means all sampled frames",
    )
    parser.add_argument(
        "--subject_mode",
        type=str,
        default="lm",
        choices=["lm", "heuristic"],
        help="lm: use a small language model to extract subjects; heuristic: regex fallback",
    )
    parser.add_argument(
        "--small_lm",
        type=str,
        default="google/flan-t5-small",
        help="Small LM for caption->subject extraction",
    )
    parser.add_argument(
        "--detector_model",
        type=str,
        default="IDEA-Research/grounding-dino-tiny",
    )
    parser.add_argument(
        "--sam_model",
        type=str,
        default="facebook/sam-vit-base",
    )
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.20)
    parser.add_argument("--context_topk", type=int, default=1)
    parser.add_argument(
        "--min_box_area_ratio",
        type=float,
        default=0.0004,
        help="Filter tiny detections by area/image_area",
    )
    parser.add_argument(
        "--max_box_area_ratio",
        type=float,
        default=0.45,
        help="Suppress over-large boxes (e.g., whole field/screen)",
    )
    parser.add_argument(
        "--subtitle_bottom_ratio",
        type=float,
        default=0.22,
        help="Bottom area ratio used to suppress subtitle-like boxes",
    )
    parser.add_argument(
        "--subtitle_aspect_ratio",
        type=float,
        default=4.5,
        help="Wide aspect ratio threshold for subtitle-like boxes",
    )
    parser.add_argument(
        "--motion_weight",
        type=float,
        default=0.8,
        help="Weight of per-box motion score when selecting primary",
    )
    parser.add_argument(
        "--temporal_iou_weight",
        type=float,
        default=0.7,
        help="Weight of IoU with previous primary box for temporal consistency",
    )
    parser.add_argument(
        "--min_motion_score",
        type=float,
        default=0.10,
        help="Low-motion boxes are penalized to avoid static background/text",
    )
    parser.add_argument(
        "--smooth_alpha",
        type=float,
        default=0.65,
        help="EMA smoothing factor for primary box coordinates",
    )
    parser.add_argument(
        "--max_center_jump_ratio",
        type=float,
        default=0.2,
        help="Maximum center jump ratio (of image diagonal) allowed in one step",
    )
    parser.add_argument(
        "--track_hold_frames",
        type=int,
        default=4,
        help="Keep last smoothed primary box this many frames when temporary miss happens",
    )
    parser.add_argument(
        "--save_mask_png",
        action="store_true",
        help="Save binary mask png for each processed frame",
    )
    parser.add_argument(
        "--save_overlay_video",
        action="store_true",
        help="Save visualization mp4 with bbox and alpha mask",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def normalize_phrase(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def singularize(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 3:
        return word[:-2]
    if word.endswith("s") and len(word) > 3 and not word.endswith("ss"):
        return word[:-1]
    return word


def split_compound_nouns(phrase: str) -> List[str]:
    words = [w for w in normalize_phrase(phrase).split() if w and w not in NOUN_STOPWORDS]
    if not words:
        return []
    candidates = []
    if len(words) >= 2:
        candidates.append(" ".join(words[-2:]))
    candidates.append(words[-1])
    candidates = [singularize(c) for c in candidates]
    dedup = []
    for c in candidates:
        if c and c not in dedup:
            dedup.append(c)
    return dedup


def heuristic_subjects(caption: str) -> Tuple[List[str], List[str]]:
    caption_n = normalize_phrase(caption)
    chunks = re.split(r"\b(?:with|on|in|at|near|next to|beside|during|while|and)\b", caption_n)
    primary = split_compound_nouns(chunks[0] if chunks else caption_n)
    context: List[str] = []
    for chunk in chunks[1:]:
        context.extend(split_compound_nouns(chunk))
    if not primary:
        tokens = [singularize(w) for w in caption_n.split() if w not in NOUN_STOPWORDS]
        if tokens:
            primary = [tokens[0]]
    context = [w for w in context if w not in primary]
    return primary[:2], context[:3]


def load_small_lm(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if device.startswith("cuda"):
        model = model.to(device)
    return tokenizer, model


def lm_extract_subjects(caption: str, tokenizer, model, device: str) -> Tuple[List[str], List[str]]:
    prompt = (
        "Extract visual entities from this caption for object localization. "
        "Return strict JSON with keys primary and context, each as list of short noun phrases. "
        "Use ONLY concrete, boxable objects (e.g., person, player, ball, motorcycle). "
        "Do NOT output actions/verbs/adjectives (e.g., running, scoring, giving instructions), "
        "nor scene-level regions (e.g., field, grassland, sky) unless directly contacted by primary. "
        "Do NOT output subtitles, text, caption, logo, watermark, UI, screen text. "
        "primary must contain main moving object nouns only; context optional for directly interacted object/surface. "
        f"Caption: {caption}"
    )
    inputs = tokenizer(text=prompt, return_tensors="pt", truncation=True)
    if device.startswith("cuda"):
        inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_new_tokens=96)
    text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()

    try:
        parsed = json.loads(text)
        primary = [normalize_phrase(x) for x in parsed.get("primary", []) if normalize_phrase(str(x))]
        context = [normalize_phrase(x) for x in parsed.get("context", []) if normalize_phrase(str(x))]
    except Exception:
        return heuristic_subjects(caption)

    if not primary:
        return heuristic_subjects(caption)

    primary = [singularize(x) for x in primary][:2]
    context = [singularize(x) for x in context if x not in primary][:3]
    return primary, context


def load_detection_model(model_name: str, device: str):
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
    if device.startswith("cuda"):
        model = model.to(device)
    model.eval()
    return processor, model


def load_sam(model_name: str, device: str):
    processor = SamProcessor.from_pretrained(model_name)
    model = SamModel.from_pretrained(model_name)
    if device.startswith("cuda"):
        model = model.to(device)
    model.eval()
    return processor, model


def build_prompt_labels(primary: Sequence[str], context: Sequence[str]) -> List[str]:
    labels = []
    for word in list(primary) + list(context):
        if word and word not in labels:
            labels.append(word)
    return labels


def detect_boxes(
    image: Image.Image,
    labels: Sequence[str],
    det_processor,
    det_model,
    box_threshold: float,
    text_threshold: float,
    min_box_area_ratio: float,
    device: str,
) -> List[DetectionResult]:
    if not labels:
        return []

    text = " ".join(f"{label}." for label in labels)
    inputs = det_processor(images=image, text=text, return_tensors="pt")
    if device.startswith("cuda"):
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = det_model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = det_processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=target_sizes,
    )[0]

    w, h = image.size
    image_area = float(w * h)
    detections: List[DetectionResult] = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        x1, y1, x2, y2 = [float(x) for x in box.tolist()]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area / image_area < min_box_area_ratio:
            continue
        detections.append(
            DetectionResult(
                label=normalize_phrase(str(label)),
                score=float(score.item()),
                box_xyxy=(x1, y1, x2, y2),
            )
        )
    return detections


def box_area_ratio(box_xyxy: Tuple[float, float, float, float], image_w: int, image_h: int) -> float:
    x1, y1, x2, y2 = box_xyxy
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return area / float(max(1, image_w * image_h))


def is_subtitle_like(
    box_xyxy: Tuple[float, float, float, float],
    image_w: int,
    image_h: int,
    subtitle_bottom_ratio: float,
    subtitle_aspect_ratio: float,
) -> bool:
    x1, y1, x2, y2 = box_xyxy
    width = max(1e-6, x2 - x1)
    height = max(1e-6, y2 - y1)
    cy = (y1 + y2) * 0.5
    aspect = width / height
    in_bottom = cy >= (1.0 - subtitle_bottom_ratio) * image_h
    return in_bottom and aspect >= subtitle_aspect_ratio and height <= 0.12 * image_h


def compute_motion_map(prev_gray: Optional[np.ndarray], curr_gray: np.ndarray) -> Optional[np.ndarray]:
    if prev_gray is None:
        return None
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=2,
        winsize=15,
        iterations=2,
        poly_n=5,
        poly_sigma=1.1,
        flags=0,
    )
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return mag


def motion_score_in_box(motion_map: Optional[np.ndarray], box_xyxy: Tuple[float, float, float, float]) -> float:
    if motion_map is None:
        return 0.0
    h, w = motion_map.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    ix1 = int(np.clip(np.floor(x1), 0, w - 1))
    iy1 = int(np.clip(np.floor(y1), 0, h - 1))
    ix2 = int(np.clip(np.ceil(x2), 1, w))
    iy2 = int(np.clip(np.ceil(y2), 1, h))
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    roi = motion_map[iy1:iy2, ix1:ix2]
    mean_mag = float(np.mean(roi))
    return float(np.clip(mean_mag / 6.0, 0.0, 1.0))


def box_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 1e-8:
        return 0.0
    return inter / denom


def smooth_primary_box(
    prev_box: Optional[Tuple[float, float, float, float]],
    curr_box: Tuple[float, float, float, float],
    alpha: float,
    max_center_jump_ratio: float,
    image_w: int,
    image_h: int,
) -> Tuple[float, float, float, float]:
    if prev_box is None:
        return curr_box

    px1, py1, px2, py2 = prev_box
    cx1, cy1, cx2, cy2 = curr_box

    prev_cx = 0.5 * (px1 + px2)
    prev_cy = 0.5 * (py1 + py2)
    curr_cx = 0.5 * (cx1 + cx2)
    curr_cy = 0.5 * (cy1 + cy2)

    max_jump = max_center_jump_ratio * np.sqrt(float(image_w * image_w + image_h * image_h))
    jump = np.sqrt((curr_cx - prev_cx) ** 2 + (curr_cy - prev_cy) ** 2)
    if jump > max_jump:
        return prev_box

    a = float(np.clip(alpha, 0.0, 1.0))
    sx1 = a * cx1 + (1 - a) * px1
    sy1 = a * cy1 + (1 - a) * py1
    sx2 = a * cx2 + (1 - a) * px2
    sy2 = a * cy2 + (1 - a) * py2
    return (sx1, sy1, sx2, sy2)


def interpolate_box(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
    t: float,
) -> Tuple[float, float, float, float]:
    return (
        (1.0 - t) * box_a[0] + t * box_b[0],
        (1.0 - t) * box_a[1] + t * box_b[1],
        (1.0 - t) * box_a[2] + t * box_b[2],
        (1.0 - t) * box_a[3] + t * box_b[3],
    )


def get_interpolated_primary_detection(
    frame_idx: int,
    keyframes: Sequence[int],
    det_by_frame: Dict[int, DetectionResult],
) -> Optional[DetectionResult]:
    if not keyframes:
        return None
    if frame_idx in det_by_frame:
        return det_by_frame[frame_idx]

    pos = bisect.bisect_left(keyframes, frame_idx)
    if pos <= 0:
        return det_by_frame[keyframes[0]]
    if pos >= len(keyframes):
        return det_by_frame[keyframes[-1]]

    prev_idx = keyframes[pos - 1]
    next_idx = keyframes[pos]
    prev_det = det_by_frame[prev_idx]
    next_det = det_by_frame[next_idx]
    denom = float(max(1, next_idx - prev_idx))
    t = float((frame_idx - prev_idx) / denom)
    box = interpolate_box(prev_det.box_xyxy, next_det.box_xyxy, t)
    score = (1.0 - t) * prev_det.score + t * next_det.score
    return DetectionResult(label=prev_det.label, score=float(score), box_xyxy=box)


def select_target_boxes(
    detections: Sequence[DetectionResult],
    primary_words: Sequence[str],
    context_words: Sequence[str],
    context_topk: int,
    image_w: int,
    image_h: int,
    motion_map: Optional[np.ndarray],
    prev_primary_box: Optional[Tuple[float, float, float, float]],
    max_box_area_ratio: float,
    subtitle_bottom_ratio: float,
    subtitle_aspect_ratio: float,
    motion_weight: float,
    temporal_iou_weight: float,
    min_motion_score: float,
) -> Tuple[List[DetectionResult], Optional[DetectionResult]]:
    prim = set(primary_words)
    ctx = set(context_words)

    primary_candidates = [d for d in detections if any(p in d.label for p in prim)]

    def primary_rank(det: DetectionResult) -> float:
        rank = det.score
        area_r = box_area_ratio(det.box_xyxy, image_w, image_h)
        if area_r > max_box_area_ratio:
            rank -= 1.2
        if is_subtitle_like(det.box_xyxy, image_w, image_h, subtitle_bottom_ratio, subtitle_aspect_ratio):
            rank -= 2.0

        motion_score = motion_score_in_box(motion_map, det.box_xyxy)
        rank += motion_weight * motion_score
        if motion_score < min_motion_score:
            rank -= 0.4

        if prev_primary_box is not None:
            rank += temporal_iou_weight * box_iou(prev_primary_box, det.box_xyxy)
        return rank

    selected_primary = max(primary_candidates, key=primary_rank) if primary_candidates else None

    selected: List[DetectionResult] = []
    if selected_primary is not None:
        selected.append(selected_primary)

    context_candidates = [
        d
        for d in sorted(detections, key=lambda x: x.score, reverse=True)
        if d is not selected_primary and any(c in d.label for c in ctx)
        and box_area_ratio(d.box_xyxy, image_w, image_h) <= max_box_area_ratio
        and not is_subtitle_like(d.box_xyxy, image_w, image_h, subtitle_bottom_ratio, subtitle_aspect_ratio)
    ]
    selected.extend(context_candidates[: max(0, context_topk)])

    return selected, selected_primary


def segment_with_boxes(
    image: Image.Image,
    boxes_xyxy: Sequence[Tuple[float, float, float, float]],
    sam_processor,
    sam_model,
    device: str,
) -> Optional[np.ndarray]:
    if not boxes_xyxy:
        return None

    input_boxes = [[list(box) for box in boxes_xyxy]]
    inputs = sam_processor(image, input_boxes=input_boxes, return_tensors="pt")
    if device.startswith("cuda"):
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = sam_model(**inputs, multimask_output=False)

    masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )

    if not masks:
        return None
    per_box = masks[0].squeeze(1).numpy() > 0.0
    union_mask = np.any(per_box, axis=0).astype(np.uint8) * 255
    return union_mask


def overlay_mask_and_boxes(
    frame_bgr: np.ndarray,
    mask: Optional[np.ndarray],
    detections: Sequence[DetectionResult],
    primary: Optional[DetectionResult],
) -> np.ndarray:
    vis = frame_bgr.copy()

    if mask is not None:
        color = np.zeros_like(vis)
        color[:, :, 1] = 220
        alpha = (mask.astype(np.float32) / 255.0) * 0.35
        vis = (vis.astype(np.float32) * (1 - alpha[..., None]) + color.astype(np.float32) * alpha[..., None]).astype(np.uint8)

    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det.box_xyxy]
        is_primary = primary is det
        box_color = (0, 255, 255) if is_primary else (255, 200, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(
            vis,
            f"{det.label}:{det.score:.2f}",
            (x1, max(0, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            box_color,
            1,
            cv2.LINE_AA,
        )
    return vis


def read_caption(video_path: Path, caption_dir: Path) -> str:
    txt_path = caption_dir / f"{video_path.stem}.txt"
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8").strip()
    return ""


def iter_sampled_frames(cap: cv2.VideoCapture, stride: int, max_frames: int):
    frame_idx = -1
    used = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % stride != 0:
            continue
        yield frame_idx, frame
        used += 1
        if max_frames > 0 and used >= max_frames:
            break


def resolve_stride(frame_stride: int, sample_interval_sec: float, fps: float) -> int:
    if sample_interval_sec is not None and sample_interval_sec > 0:
        if fps <= 0:
            return max(1, frame_stride)
        return max(1, int(round(sample_interval_sec * fps)))
    return max(1, frame_stride)


def process_video(
    video_path: Path,
    caption_dir: Path,
    output_dir: Path,
    subject_mode: str,
    lm_pack,
    det_pack,
    sam_pack,
    args,
) -> Dict:
    caption = read_caption(video_path, caption_dir)
    if not caption:
        caption = video_path.stem.replace("_", " ")

    if subject_mode == "lm":
        primary_words, context_words = lm_extract_subjects(caption, *lm_pack, args.device)
    else:
        primary_words, context_words = heuristic_subjects(caption)

    labels = build_prompt_labels(primary_words, context_words)

    det_processor, det_model = det_pack
    sam_processor, sam_model = sam_pack

    save_root = output_dir / video_path.stem
    frame_dir = save_root / "frames"
    mask_dir = save_root / "masks"
    frame_dir.mkdir(parents=True, exist_ok=True)
    if args.save_mask_png:
        mask_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    effective_stride = resolve_stride(args.frame_stride, args.sample_interval_sec, fps)

    frame_records = []
    processed = 0
    hit_primary = 0
    prev_gray: Optional[np.ndarray] = None
    prev_primary_box: Optional[Tuple[float, float, float, float]] = None
    miss_streak = 0
    sampled_primary_by_frame: Dict[int, DetectionResult] = {}

    for frame_idx, frame_bgr in iter_sampled_frames(cap, effective_stride, args.max_frames_per_video):
        curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        motion_map = compute_motion_map(prev_gray, curr_gray)
        image_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        detections = detect_boxes(
            image=image_pil,
            labels=labels,
            det_processor=det_processor,
            det_model=det_model,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            min_box_area_ratio=args.min_box_area_ratio,
            device=args.device,
        )

        selected, primary_det = select_target_boxes(
            detections=detections,
            primary_words=primary_words,
            context_words=context_words,
            context_topk=args.context_topk,
            image_w=width,
            image_h=height,
            motion_map=motion_map,
            prev_primary_box=prev_primary_box,
            max_box_area_ratio=args.max_box_area_ratio,
            subtitle_bottom_ratio=args.subtitle_bottom_ratio,
            subtitle_aspect_ratio=args.subtitle_aspect_ratio,
            motion_weight=args.motion_weight,
            temporal_iou_weight=args.temporal_iou_weight,
            min_motion_score=args.min_motion_score,
        )

        if primary_det is not None:
            smoothed_box = smooth_primary_box(
                prev_box=prev_primary_box,
                curr_box=primary_det.box_xyxy,
                alpha=args.smooth_alpha,
                max_center_jump_ratio=args.max_center_jump_ratio,
                image_w=width,
                image_h=height,
            )
            primary_det = DetectionResult(label=primary_det.label, score=primary_det.score, box_xyxy=smoothed_box)
            if selected:
                selected[0] = primary_det
            prev_primary_box = smoothed_box
            miss_streak = 0
        else:
            miss_streak += 1
            if prev_primary_box is not None and miss_streak <= args.track_hold_frames:
                held = DetectionResult(label="tracked_primary", score=0.0, box_xyxy=prev_primary_box)
                selected = [held] + selected
                primary_det = held

        mask = segment_with_boxes(
            image=image_pil,
            boxes_xyxy=[d.box_xyxy for d in selected],
            sam_processor=sam_processor,
            sam_model=sam_model,
            device=args.device,
        )

        if primary_det is not None:
            hit_primary += 1
            sampled_primary_by_frame[frame_idx] = primary_det

        vis = overlay_mask_and_boxes(frame_bgr, mask, selected, primary_det)
        frame_out = frame_dir / f"{frame_idx:06d}.jpg"
        cv2.imwrite(str(frame_out), vis)

        if args.save_mask_png and mask is not None:
            cv2.imwrite(str(mask_dir / f"{frame_idx:06d}.png"), mask)

        frame_records.append(
            {
                "frame_idx": frame_idx,
                "selected": [
                    {
                        "label": d.label,
                        "score": d.score,
                        "box_xyxy": [round(v, 2) for v in d.box_xyxy],
                    }
                    for d in selected
                ],
                "primary_found": primary_det is not None,
                "mask_saved": bool(mask is not None),
            }
        )
        processed += 1
        prev_gray = curr_gray

    cap.release()

    if args.save_overlay_video:
        overlay_path = save_root / "overlay.mp4"
        writer = cv2.VideoWriter(
            str(overlay_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(1e-3, float(fps)),
            (width, height),
        )

        cap_full = cv2.VideoCapture(str(video_path))
        keyframes = sorted(sampled_primary_by_frame.keys())
        full_idx = -1
        while True:
            ok, frame_full = cap_full.read()
            if not ok:
                break
            full_idx += 1
            interp_primary = get_interpolated_primary_detection(full_idx, keyframes, sampled_primary_by_frame)
            draw_dets = [interp_primary] if interp_primary is not None else []
            full_vis = overlay_mask_and_boxes(frame_full, None, draw_dets, interp_primary)
            writer.write(full_vis)

        cap_full.release()
        writer.release()

    summary = {
        "video": str(video_path),
        "caption": caption,
        "primary_words": primary_words,
        "context_words": context_words,
        "prompt_labels": labels,
        "processed_frames": processed,
        "primary_hit_frames": hit_primary,
        "primary_hit_ratio": (hit_primary / processed) if processed > 0 else 0.0,
    }

    (save_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (save_root / "frames.jsonl").open("w", encoding="utf-8") as f:
        for row in frame_records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return summary


def main() -> None:
    args = parse_args()

    caption_dir = args.caption_dir if args.caption_dir is not None else args.video_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(args.video_dir.glob(args.video_glob))
    if args.max_videos > 0:
        videos = videos[: args.max_videos]

    if not videos:
        raise RuntimeError(f"No video found under {args.video_dir} with glob {args.video_glob}")

    lm_pack = None
    if args.subject_mode == "lm":
        try:
            lm_pack = load_small_lm(args.small_lm, args.device)
        except Exception:
            args.subject_mode = "heuristic"

    if lm_pack is None:
        lm_pack = (None, None)

    det_pack = load_detection_model(args.detector_model, args.device)
    sam_pack = load_sam(args.sam_model, args.device)

    all_summaries = []
    for video_path in tqdm(videos, desc="Segment videos"):
        summary = process_video(
            video_path=video_path,
            caption_dir=caption_dir,
            output_dir=args.output_dir,
            subject_mode=args.subject_mode,
            lm_pack=lm_pack,
            det_pack=det_pack,
            sam_pack=sam_pack,
            args=args,
        )
        all_summaries.append(summary)

    stats = {
        "num_videos": len(all_summaries),
        "avg_primary_hit_ratio": float(np.mean([s["primary_hit_ratio"] for s in all_summaries])),
        "videos": all_summaries,
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"done": len(all_summaries), "avg_primary_hit_ratio": stats["avg_primary_hit_ratio"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
