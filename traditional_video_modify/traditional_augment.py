#!/usr/bin/env python3
import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm


@dataclass
class AugParams:
    brightness: float
    contrast: float
    saturation: float
    hue_shift: float
    temperature: float
    red_gain: float
    green_gain: float
    blue_gain: float


def sample_params(rng: np.random.Generator, strength: str) -> AugParams:
    if strength == "mild":
        return AugParams(
            brightness=float(rng.uniform(-18.0, 18.0)),
            contrast=float(rng.uniform(0.92, 1.10)),
            saturation=float(rng.uniform(0.88, 1.18)),
            hue_shift=float(rng.uniform(-8.0, 8.0)),
            temperature=float(rng.uniform(-0.12, 0.12)),
            red_gain=float(rng.uniform(0.90, 1.10)),
            green_gain=float(rng.uniform(0.92, 1.08)),
            blue_gain=float(rng.uniform(0.90, 1.10)),
        )
    return AugParams(
        brightness=float(rng.uniform(-30.0, 30.0)),
        contrast=float(rng.uniform(0.84, 1.22)),
        saturation=float(rng.uniform(0.74, 1.34)),
        hue_shift=float(rng.uniform(-16.0, 16.0)),
        temperature=float(rng.uniform(-0.20, 0.20)),
        red_gain=float(rng.uniform(0.84, 1.20)),
        green_gain=float(rng.uniform(0.86, 1.16)),
        blue_gain=float(rng.uniform(0.84, 1.20)),
    )


def enforce_visible_change(params: AugParams, rng: np.random.Generator, min_level: float) -> AugParams:
    p = AugParams(**asdict(params))

    def with_sign(x: float, fallback_sign: float) -> float:
        if abs(x) > 1e-6:
            return x
        return fallback_sign

    if abs(p.brightness) < 20.0 * min_level:
        sign = np.sign(with_sign(p.brightness, rng.choice([-1.0, 1.0])))
        p.brightness = float(sign * 20.0 * min_level)

    if abs(p.contrast - 1.0) < 0.18 * min_level:
        sign = np.sign(with_sign(p.contrast - 1.0, rng.choice([-1.0, 1.0])))
        p.contrast = float(1.0 + sign * 0.18 * min_level)

    if abs(p.saturation - 1.0) < 0.25 * min_level:
        sign = np.sign(with_sign(p.saturation - 1.0, rng.choice([-1.0, 1.0])))
        p.saturation = float(1.0 + sign * 0.25 * min_level)

    if abs(p.hue_shift) < 10.0 * min_level:
        sign = np.sign(with_sign(p.hue_shift, rng.choice([-1.0, 1.0])))
        p.hue_shift = float(sign * 10.0 * min_level)

    if abs(p.temperature) < 0.14 * min_level:
        sign = np.sign(with_sign(p.temperature, rng.choice([-1.0, 1.0])))
        p.temperature = float(sign * 0.14 * min_level)

    if abs(p.red_gain - 1.0) < 0.14 * min_level:
        sign = np.sign(with_sign(p.red_gain - 1.0, rng.choice([-1.0, 1.0])))
        p.red_gain = float(1.0 + sign * 0.14 * min_level)

    if abs(p.green_gain - 1.0) < 0.12 * min_level:
        sign = np.sign(with_sign(p.green_gain - 1.0, rng.choice([-1.0, 1.0])))
        p.green_gain = float(1.0 + sign * 0.12 * min_level)

    if abs(p.blue_gain - 1.0) < 0.14 * min_level:
        sign = np.sign(with_sign(p.blue_gain - 1.0, rng.choice([-1.0, 1.0])))
        p.blue_gain = float(1.0 + sign * 0.14 * min_level)

    return p


def jitter(base: float, rng: np.random.Generator, ratio: float) -> float:
    return base * (1.0 + rng.uniform(-ratio, ratio))


def jitter_add(base: float, rng: np.random.Generator, amount: float) -> float:
    return base + rng.uniform(-amount, amount)


def apply_frame_aug(frame_bgr: np.ndarray, params: AugParams) -> np.ndarray:
    img = frame_bgr.astype(np.float32)

    hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + params.hue_shift) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1] * params.saturation, 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    img = (img - 127.5) * params.contrast + 127.5 + params.brightness
    img = np.clip(img, 0, 255)

    temp = params.temperature
    r_temp = 1.0 + 0.85 * temp
    g_temp = 1.0 + 0.18 * temp
    b_temp = 1.0 - 0.85 * temp

    b_gain = params.blue_gain * b_temp
    g_gain = params.green_gain * g_temp
    r_gain = params.red_gain * r_temp

    img[..., 0] = img[..., 0] * b_gain
    img[..., 1] = img[..., 1] * g_gain
    img[..., 2] = img[..., 2] * r_gain

    return np.clip(img, 0, 255).astype(np.uint8)


def smooth_params(prev: AugParams, cur: AugParams, momentum: float = 0.92) -> AugParams:
    return AugParams(
        brightness=momentum * prev.brightness + (1.0 - momentum) * cur.brightness,
        contrast=momentum * prev.contrast + (1.0 - momentum) * cur.contrast,
        saturation=momentum * prev.saturation + (1.0 - momentum) * cur.saturation,
        hue_shift=momentum * prev.hue_shift + (1.0 - momentum) * cur.hue_shift,
        temperature=momentum * prev.temperature + (1.0 - momentum) * cur.temperature,
        red_gain=momentum * prev.red_gain + (1.0 - momentum) * cur.red_gain,
        green_gain=momentum * prev.green_gain + (1.0 - momentum) * cur.green_gain,
        blue_gain=momentum * prev.blue_gain + (1.0 - momentum) * cur.blue_gain,
    )


def process_video(
    input_path: Path,
    output_path: Path,
    params_base: AugParams,
    seed: int,
    max_frames: int,
    temporal_jitter: float,
) -> Tuple[int, float, int, int]:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-3:
        fps = 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames > 0:
        n_frames = min(n_frames, max_frames)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    rng = np.random.default_rng(seed)
    cur = params_base
    wrote = 0

    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break

        target = AugParams(
            brightness=jitter_add(params_base.brightness, rng, 4.0 * temporal_jitter),
            contrast=jitter(params_base.contrast, rng, temporal_jitter),
            saturation=jitter(params_base.saturation, rng, temporal_jitter),
            hue_shift=jitter_add(params_base.hue_shift, rng, 2.5 * temporal_jitter),
            temperature=jitter_add(params_base.temperature, rng, 0.8 * temporal_jitter),
            red_gain=jitter(params_base.red_gain, rng, temporal_jitter),
            green_gain=jitter(params_base.green_gain, rng, temporal_jitter),
            blue_gain=jitter(params_base.blue_gain, rng, temporal_jitter),
        )

        cur = smooth_params(cur, target, momentum=0.92)
        out = apply_frame_aug(frame, cur)
        writer.write(out)
        wrote += 1

    cap.release()
    writer.release()
    return wrote, fps, width, height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Traditional random color jitter + temperature + RGB gain video augmentation.")
    parser.add_argument("--input-dir", required=True, help="Input directory with mp4 videos")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--glob", default="*.mp4", help="Input glob pattern")
    parser.add_argument("--strength", choices=["mild", "medium"], default="mild", help="Augmentation strength")
    parser.add_argument("--variants", type=int, default=2, help="How many style variants per source video")
    parser.add_argument("--max-videos", type=int, default=-1)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--temporal-jitter", type=float, default=0.04, help="Per-frame jitter amount (small keeps motion readable)")
    parser.add_argument("--force-visible-change", action="store_true", help="Force minimum color/style offset from original")
    parser.add_argument("--min-change-level", type=float, default=1.0, help="Minimum visible change level when --force-visible-change is enabled")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    videos = sorted([p for p in in_dir.glob(args.glob) if p.suffix.lower() == ".mp4"])
    if args.max_videos > 0:
        videos = videos[: args.max_videos]
    if not videos:
        raise RuntimeError("No mp4 files found.")

    out_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict] = []
    for vid_idx, video in enumerate(tqdm(videos, desc="Traditional augment")):
        key = video.stem
        for variant in range(args.variants):
            variant_name = f"v{variant+1:02d}"
            out_path = out_dir / variant_name / f"{key}_{variant_name}.mp4"
            if out_path.exists() and not args.overwrite:
                continue

            rng = np.random.default_rng(args.seed + vid_idx * 1000 + variant * 53)
            base = sample_params(rng, args.strength)
            if args.force_visible_change:
                base = enforce_visible_change(base, rng, min_level=args.min_change_level)

            frames, fps, w, h = process_video(
                input_path=video,
                output_path=out_path,
                params_base=base,
                seed=args.seed + vid_idx * 1000 + variant * 53 + 7,
                max_frames=args.max_frames,
                temporal_jitter=args.temporal_jitter,
            )

            records.append(
                {
                    "key": key,
                    "variant": variant_name,
                    "source": str(video),
                    "output": str(out_path),
                    "frames": frames,
                    "fps": fps,
                    "width": w,
                    "height": h,
                    **asdict(base),
                }
            )

    manifest_jsonl = out_dir / "manifest.jsonl"
    manifest_csv = out_dir / "manifest.csv"

    with open(manifest_jsonl, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if records:
        with open(manifest_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    print(f"Done. Generated {len(records)} videos at {out_dir}")
    print(f"Manifest: {manifest_jsonl}")


if __name__ == "__main__":
    main()
