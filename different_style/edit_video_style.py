#!/usr/bin/env python3
import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


COCO_CAR_CLASS_ID = 2
COCO_CLASS_NAME_TO_ID = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "train": 6,
    "truck": 7,
    "boat": 8,
    "bird": 14,
    "cat": 15,
    "dog": 16,
    "horse": 17,
}


@dataclass
class TrackerState:
    prev_mask: Optional[np.ndarray] = None
    prev_box: Optional[np.ndarray] = None
    missing_count: int = 0


class CarMaskTracker:
    def __init__(
        self,
        model_name: str,
        conf: float = 0.25,
        max_missing: int = 12,
        class_ids: Optional[list[int]] = None,
        min_mask_area_ratio: float = 0.0003,
        max_mask_area_ratio: float = 0.70,
    ):
        self.model = YOLO(model_name)
        self.conf = conf
        self.max_missing = max_missing
        self.class_ids = class_ids
        self.min_mask_area_ratio = min_mask_area_ratio
        self.max_mask_area_ratio = max_mask_area_ratio
        self.state = TrackerState()

    @staticmethod
    def _bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        xa1, ya1, xa2, ya2 = box_a
        xb1, yb1, xb2, yb2 = box_b
        inter_x1 = max(xa1, xb1)
        inter_y1 = max(ya1, yb1)
        inter_x2 = min(xa2, xb2)
        inter_y2 = min(ya2, yb2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
        area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
        denom = area_a + area_b - inter_area + 1e-6
        return float(inter_area / denom)

    def _select_candidate(self, boxes: np.ndarray) -> int:
        if boxes.shape[0] == 1:
            return 0
        if self.state.prev_box is None:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            return int(np.argmax(areas))
        ious = np.array([self._bbox_iou(self.state.prev_box, box) for box in boxes])
        return int(np.argmax(ious))

    def get_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            classes=self.class_ids,
            verbose=False,
            retina_masks=True,
        )

        if not results:
            return self._fallback_mask((h, w))

        result = results[0]
        if result.masks is None or result.boxes is None or len(result.boxes) == 0:
            return self._fallback_mask((h, w))

        boxes = result.boxes.xyxy.detach().cpu().numpy()
        idx = self._select_candidate(boxes)

        mask = result.masks.data[idx].detach().cpu().numpy()
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = (mask > 0.5).astype(np.uint8)

        ratio = float(mask.mean())
        if ratio < self.min_mask_area_ratio or ratio > self.max_mask_area_ratio:
            return self._fallback_mask((h, w))

        self.state.prev_box = boxes[idx]
        self.state.prev_mask = mask
        self.state.missing_count = 0
        return mask

    def _fallback_mask(self, shape_hw: Tuple[int, int]) -> np.ndarray:
        h, w = shape_hw
        if self.state.prev_mask is None:
            return np.zeros((h, w), dtype=np.uint8)
        self.state.missing_count += 1
        if self.state.missing_count > self.max_missing:
            self.state.prev_mask = None
            self.state.prev_box = None
            return np.zeros((h, w), dtype=np.uint8)
        return self.state.prev_mask.copy()


class BackgroundSource:
    def __init__(self, bg_image: Optional[str], bg_video: Optional[str], frame_size: Tuple[int, int]):
        self.width, self.height = frame_size
        self.bg_image = None
        self.bg_cap = None

        if bg_image:
            img = cv2.imread(bg_image)
            if img is None:
                raise ValueError(f"Cannot read background image: {bg_image}")
            self.bg_image = img
        elif bg_video:
            cap = cv2.VideoCapture(bg_video)
            if not cap.isOpened():
                raise ValueError(f"Cannot open background video: {bg_video}")
            self.bg_cap = cap
        else:
            raise ValueError("Either background image or background video must be provided.")

    def _cover_resize(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        target_w, target_h = self.width, self.height
        scale = max(target_w / max(w, 1), target_h / max(h, 1))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        x1 = (new_w - target_w) // 2
        y1 = (new_h - target_h) // 2
        return resized[y1:y1 + target_h, x1:x1 + target_w]

    def read(self) -> np.ndarray:
        if self.bg_image is not None:
            return self._cover_resize(self.bg_image)

        ok, frame = self.bg_cap.read()
        if not ok:
            self.bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.bg_cap.read()
            if not ok:
                raise RuntimeError("Background video cannot be read.")
        return self._cover_resize(frame)


class MotionEstimator:
    def __init__(self):
        self.prev_gray: Optional[np.ndarray] = None
        self.cumulative_shift = np.array([0.0, 0.0], dtype=np.float32)

    def update(self, frame_bgr: np.ndarray, ignore_mask: Optional[np.ndarray] = None) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return self.cumulative_shift.copy()

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=25,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        if ignore_mask is not None:
            valid = ignore_mask == 0
            if np.any(valid):
                dx = np.median(flow[..., 0][valid])
                dy = np.median(flow[..., 1][valid])
            else:
                dx, dy = 0.0, 0.0
        else:
            dx = np.median(flow[..., 0])
            dy = np.median(flow[..., 1])

        self.cumulative_shift += np.array([dx, dy], dtype=np.float32)
        self.prev_gray = gray
        return self.cumulative_shift.copy()


def parse_color(color: str) -> Tuple[int, int, int]:
    named = {
        "red": (0, 0, 255),
        "blue": (255, 0, 0),
        "green": (0, 255, 0),
        "yellow": (0, 255, 255),
        "black": (16, 16, 16),
        "white": (245, 245, 245),
        "silver": (192, 192, 192),
    }
    color_lower = color.lower().strip()
    if color_lower in named:
        return named[color_lower]

    if color_lower.startswith("#") and len(color_lower) == 7:
        r = int(color_lower[1:3], 16)
        g = int(color_lower[3:5], 16)
        b = int(color_lower[5:7], 16)
        return (b, g, r)

    raise ValueError(f"Unsupported color: {color}. Use named colors or #RRGGBB.")


def parse_subject_classes(subject_classes: str) -> Optional[list[int]]:
    txt = subject_classes.strip().lower()
    if txt in {"auto", "any", "all"}:
        return None

    class_ids: list[int] = []
    for token in txt.split(","):
        token = token.strip()
        if not token:
            continue
        if token.isdigit():
            class_ids.append(int(token))
            continue
        if token in COCO_CLASS_NAME_TO_ID:
            class_ids.append(COCO_CLASS_NAME_TO_ID[token])
            continue
        raise ValueError(
            f"Unknown subject class token: {token}. Use COCO names like person/car/dog or numeric ids."
        )

    if not class_ids:
        return None
    return sorted(set(class_ids))


def stylize_subject(frame_bgr: np.ndarray, subject_mask: np.ndarray, target_bgr: Tuple[int, int, int], material_strength: float) -> np.ndarray:
    out = frame_bgr.copy()
    if subject_mask.max() == 0:
        return out

    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    target_hsv = cv2.cvtColor(np.uint8([[target_bgr]]), cv2.COLOR_BGR2HSV).astype(np.float32)[0, 0]

    m = subject_mask.astype(bool)
    hsv[..., 0][m] = 0.15 * hsv[..., 0][m] + 0.85 * target_hsv[0]
    hsv[..., 1][m] = np.clip(0.25 * hsv[..., 1][m] + 0.75 * max(target_hsv[1], 120), 0, 255)
    hsv[..., 2][m] = np.clip(hsv[..., 2][m] * (1.0 + 0.08 * material_strength), 0, 255)

    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if material_strength > 0:
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (0, 0), 3.0)
        detail = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
        detail = np.clip(detail, 0, 255).astype(np.uint8)
        detail_bgr = cv2.cvtColor(detail, cv2.COLOR_GRAY2BGR)

        alpha = np.clip(0.08 + 0.22 * material_strength, 0.0, 0.5)
        blend = cv2.addWeighted(out, 1.0 - alpha, detail_bgr, alpha, 0)
        out[m] = blend[m]

    return out


def shift_background(bg_bgr: np.ndarray, shift_xy: np.ndarray) -> np.ndarray:
    dx, dy = shift_xy
    h, w = bg_bgr.shape[:2]
    mat = np.array([[1.0, 0.0, -float(dx)], [0.0, 1.0, -float(dy)]], dtype=np.float32)
    moved = cv2.warpAffine(bg_bgr, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return moved


def compose_frame(car_frame: np.ndarray, bg_frame: np.ndarray, car_mask: np.ndarray, feather: int) -> np.ndarray:
    mask_u8 = (car_mask * 255).astype(np.uint8)
    if feather > 0:
        k = feather * 2 + 1
        mask_u8 = cv2.GaussianBlur(mask_u8, (k, k), 0)
    alpha = (mask_u8.astype(np.float32) / 255.0)[..., None]
    comp = car_frame.astype(np.float32) * alpha + bg_frame.astype(np.float32) * (1.0 - alpha)
    return np.clip(comp, 0, 255).astype(np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(description="Keep subject motion, edit subject style, and replace background.")
    parser.add_argument("--input-video", required=True, help="Path to source driving video")
    parser.add_argument("--output-video", required=True, help="Path to output video")
    parser.add_argument("--background-image", default=None, help="Path to replacement background image")
    parser.add_argument("--background-video", default=None, help="Path to replacement background video")
    parser.add_argument("--subject-color", default="red", help="Named color or #RRGGBB")
    parser.add_argument("--car-color", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--subject-classes", default="auto", help="COCO class names or ids, e.g. person or person,car or 0,2; use auto for any")
    parser.add_argument("--preset", choices=["fast", "strict"], default="fast")
    parser.add_argument("--model", default=None, help="YOLO segmentation model, e.g. yolov8n-seg.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--material-strength", type=float, default=0.8)
    parser.add_argument("--mask-feather", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--no-mask-action", choices=["keep-original", "replace-background"], default="keep-original")
    parser.add_argument("--min-mask-area-ratio", type=float, default=0.0003)
    parser.add_argument("--max-mask-area-ratio", type=float, default=0.70)
    parser.add_argument("--save-mask-video", default=None, help="Optional output path for debug car mask video")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input_video):
        raise FileNotFoundError(f"Input video not found: {args.input_video}")
    if (args.background_image is None) == (args.background_video is None):
        raise ValueError("Provide exactly one of --background-image or --background-video")

    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {args.input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-3:
        fps = 24.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames > 0:
        total_frames = min(total_frames, args.max_frames)

    os.makedirs(os.path.dirname(args.output_video) or ".", exist_ok=True)
    writer = cv2.VideoWriter(
        args.output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    mask_writer = None
    if args.save_mask_video:
        os.makedirs(os.path.dirname(args.save_mask_video) or ".", exist_ok=True)
        mask_writer = cv2.VideoWriter(
            args.save_mask_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

    model_name = args.model
    if model_name is None:
        model_name = "yolov8n-seg.pt" if args.preset == "fast" else "yolov8x-seg.pt"

    class_ids = parse_subject_classes(args.subject_classes)
    tracker = CarMaskTracker(
        model_name=model_name,
        conf=args.conf,
        max_missing=18 if args.preset == "strict" else 8,
        class_ids=class_ids,
        min_mask_area_ratio=args.min_mask_area_ratio,
        max_mask_area_ratio=args.max_mask_area_ratio,
    )
    bg_source = BackgroundSource(args.background_image, args.background_video, frame_size=(width, height))
    motion_estimator = MotionEstimator()

    color_arg = args.car_color if args.car_color is not None else args.subject_color
    color_bgr = parse_color(color_arg)
    prev_mask = None

    for _ in tqdm(range(total_frames), desc="Editing"):
        ok, frame = cap.read()
        if not ok:
            break

        mask = tracker.get_mask(frame)

        if args.preset == "strict" and prev_mask is not None:
            mask = (0.35 * prev_mask + 0.65 * mask).astype(np.float32)
            mask = (mask > 0.5).astype(np.uint8)
        prev_mask = mask.copy()

        has_subject = bool(mask.max() > 0)
        if not has_subject:
            if args.no_mask_action == "keep-original":
                comp = frame.copy()
            else:
                comp = bg_source.read()
            writer.write(comp)
            if mask_writer is not None:
                mask_writer.write(np.zeros_like(frame))
            continue

        subject_styled = stylize_subject(frame, mask, color_bgr, material_strength=args.material_strength)

        bg = bg_source.read()
        if args.preset == "strict":
            shift = motion_estimator.update(frame, ignore_mask=mask)
            bg = shift_background(bg, shift_xy=shift)

        comp = compose_frame(subject_styled, bg, mask, feather=args.mask_feather if args.preset == "strict" else 1)
        writer.write(comp)

        if mask_writer is not None:
            mask_vis = (mask * 255).astype(np.uint8)
            mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
            mask_writer.write(mask_vis)

    cap.release()
    writer.release()
    if mask_writer is not None:
        mask_writer.release()

    print(f"Done. Output saved to: {args.output_video}")


if __name__ == "__main__":
    main()
