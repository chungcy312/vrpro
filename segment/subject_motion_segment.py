#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
        "primary must contain the main moving subject; context contains interacted object/surface if explicit. "
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

    text = [[f"{label}." for label in labels]]
    inputs = det_processor(images=image, text=text, return_tensors="pt")
    if device.startswith("cuda"):
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = det_model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = det_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
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


def select_target_boxes(
    detections: Sequence[DetectionResult],
    primary_words: Sequence[str],
    context_words: Sequence[str],
    context_topk: int,
) -> Tuple[List[DetectionResult], Optional[DetectionResult]]:
    prim = set(primary_words)
    ctx = set(context_words)

    primary_candidates = [d for d in detections if any(p in d.label for p in prim)]
    selected_primary = max(primary_candidates, key=lambda x: x.score) if primary_candidates else None

    selected: List[DetectionResult] = []
    if selected_primary is not None:
        selected.append(selected_primary)

    context_candidates = [
        d
        for d in sorted(detections, key=lambda x: x.score, reverse=True)
        if d is not selected_primary and any(c in d.label for c in ctx)
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

    writer = None
    if args.save_overlay_video:
        save_root.mkdir(parents=True, exist_ok=True)
        overlay_path = save_root / "overlay.mp4"
        writer = cv2.VideoWriter(
            str(overlay_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps / max(1, args.frame_stride),
            (width, height),
        )

    frame_records = []
    processed = 0
    hit_primary = 0

    for frame_idx, frame_bgr in iter_sampled_frames(cap, args.frame_stride, args.max_frames_per_video):
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
        )

        mask = segment_with_boxes(
            image=image_pil,
            boxes_xyxy=[d.box_xyxy for d in selected],
            sam_processor=sam_processor,
            sam_model=sam_model,
            device=args.device,
        )

        if primary_det is not None:
            hit_primary += 1

        vis = overlay_mask_and_boxes(frame_bgr, mask, selected, primary_det)
        frame_out = frame_dir / f"{frame_idx:06d}.jpg"
        cv2.imwrite(str(frame_out), vis)

        if args.save_mask_png and mask is not None:
            cv2.imwrite(str(mask_dir / f"{frame_idx:06d}.png"), mask)

        if writer is not None:
            writer.write(vis)

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

    cap.release()
    if writer is not None:
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
