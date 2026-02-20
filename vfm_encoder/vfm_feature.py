from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torchvision.transforms import Normalize

try:
    from torchvision.io import read_video

    TVIO_AVAILABLE = True
except Exception:
    TVIO_AVAILABLE = False

try:
    import decord

    decord.bridge.set_bridge("torch")
    DECORD_AVAILABLE = True
except Exception:
    DECORD_AVAILABLE = False

try:
    import cv2

    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False


REPO_ROOT = Path(__file__).resolve().parents[1]
VIDEOREPA_ROOT = REPO_ROOT / "VideoREPA"
if str(VIDEOREPA_ROOT) not in sys.path:
    sys.path.append(str(VIDEOREPA_ROOT))
SSL_ROOT = VIDEOREPA_ROOT / "finetune" / "models" / "cogvideox_t2v_align" / "models" / "ssl"
if str(SSL_ROOT) not in sys.path:
    sys.path.append(str(SSL_ROOT))


@dataclass
class ModelSpec:
    name: str
    feature_dim: int
    patch_size: int
    tubelet_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract VFM features for motion_top100 videos")
    parser.add_argument(
        "--video_dir",
        type=Path,
        default=REPO_ROOT / "dataset" / "motion_top100",
        help="Directory that stores motion_top100 *.mp4 files",
    )
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=REPO_ROOT / "dataset" / "motion_top100.csv",
        help="CSV file that stores captions",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=REPO_ROOT / "dataset" / "vfm_features_motion_top100",
        help="Output directory for extracted feature files",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["VideoMAEv2", "VideoMAE", "DINOv2", "VJEPA2"],
        choices=["VideoMAEv2", "VideoMAE", "DINOv2", "VJEPA2"],
        help="VFM models to run",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_frames", type=int, default=48, help="Sampled frames per video before tubelet")
    parser.add_argument("--videomae_resize_h", type=int, default=160)
    parser.add_argument("--videomae_resize_w", type=int, default=240)
    parser.add_argument("--dinov2_resize_h", type=int, default=420)
    parser.add_argument("--dinov2_resize_w", type=int, default=630)
    parser.add_argument("--dinov2_chunk", type=int, default=128, help="Frames per chunk for DINOv2 forward")
    parser.add_argument(
        "--videomaev2_ckpt",
        type=Path,
        default=VIDEOREPA_ROOT / "ckpt" / "VideoMAEv2" / "vit_b_k710_dl_from_giant.pth",
    )
    parser.add_argument(
        "--videomae_ckpt",
        type=Path,
        default=VIDEOREPA_ROOT / "ckpt" / "VideoMAE" / "k400_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0_9_e1600.pth",
    )
    parser.add_argument(
        "--allow_skip_missing",
        action="store_true",
        help="Skip model if checkpoint/dependency is missing instead of hard fail",
    )
    return parser.parse_args()


def load_caption_map(csv_path: Path) -> Dict[str, str]:
    caption_map: Dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for idx, row in enumerate(reader):
            caption_map[f"{idx:04d}"] = row.get("caption", "")
    return caption_map


def sample_video_frames(video_path: Path, num_frames: int) -> torch.Tensor:
    if DECORD_AVAILABLE:
        try:
            vr = decord.VideoReader(video_path.as_posix())
            total = len(vr)
            if total <= 0:
                raise RuntimeError(f"No frames in video: {video_path}")

            if total >= num_frames:
                idxs = torch.linspace(0, total - 1, steps=num_frames).round().long()
            else:
                idxs = torch.linspace(0, total - 1, steps=num_frames).round().long().clamp(0, total - 1)

            frames = vr.get_batch(idxs.tolist())  # [T, H, W, C], uint8
            frames = frames.float() / 255.0
            frames = frames.permute(0, 3, 1, 2).contiguous()  # [T, C, H, W]
            frames = frames.permute(1, 0, 2, 3).contiguous()  # [C, T, H, W]
            return frames
        except Exception:
            pass

    if CV2_AVAILABLE:
        try:
            cap = cv2.VideoCapture(video_path.as_posix())
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {video_path}")

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                raise RuntimeError(f"No frames in video: {video_path}")

            if total >= num_frames:
                idxs = torch.linspace(0, total - 1, steps=num_frames).round().long().tolist()
            else:
                idxs = torch.linspace(0, total - 1, steps=num_frames).round().long().clamp(0, total - 1).tolist()

            frames: List[torch.Tensor] = []
            for frame_idx in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ok, frame_bgr = cap.read()
                if not ok:
                    if len(frames) == 0:
                        cap.release()
                        raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
                    frame_rgb = frames[-1]
                else:
                    frame_rgb_np = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    frame_rgb = torch.from_numpy(frame_rgb_np)
                frames.append(frame_rgb)

            cap.release()
            video = torch.stack(frames, dim=0).float() / 255.0  # [T, H, W, C]
            video = video.permute(0, 3, 1, 2).contiguous()  # [T, C, H, W]
            video = video.permute(1, 0, 2, 3).contiguous()  # [C, T, H, W]
            return video
        except Exception:
            pass

    if TVIO_AVAILABLE:
        try:
            video, _, _ = read_video(video_path.as_posix(), pts_unit="sec")  # [T, H, W, C], uint8
            if video.ndim != 4 or video.shape[0] == 0:
                raise RuntimeError(f"No frames in video: {video_path}")

            total = int(video.shape[0])
            if total >= num_frames:
                idxs = torch.linspace(0, total - 1, steps=num_frames).round().long()
            else:
                idxs = torch.linspace(0, total - 1, steps=num_frames).round().long().clamp(0, total - 1)

            video = video.index_select(0, idxs)
            video = video.float() / 255.0
            video = video.permute(0, 3, 1, 2).contiguous()  # [T, C, H, W]
            video = video.permute(1, 0, 2, 3).contiguous()  # [C, T, H, W]
            return video
        except Exception:
            pass

    raise RuntimeError(f"Failed to decode video with all backends: {video_path}")


def normalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    # x: [N, C, H, W], value in [0,1]
    norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return norm(x)


def _patch_size_int(maybe_patch) -> int:
    if isinstance(maybe_patch, (tuple, list)):
        return int(maybe_patch[0])
    return int(maybe_patch)


def _tensor_from_model_output(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    if isinstance(output, dict):
        priority = [
            "x_norm_patchtokens",
            "last_hidden_state",
            "hidden_states",
            "feat",
            "features",
            "x",
        ]
        for key in priority:
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value
            if isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                return value[0]
    raise RuntimeError(f"Cannot parse tensor from model output type={type(output)}")


def load_model(model_name: str, args: argparse.Namespace, device: torch.device):
    if model_name == "VideoMAEv2":
        if not args.videomaev2_ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {args.videomaev2_ckpt}")
        from VideoMAEv2 import vit_base_patch16_224

        model = vit_base_patch16_224().to(device)
        model.from_pretrained(str(args.videomaev2_ckpt))
        model.eval()
        spec = ModelSpec(name=model_name, feature_dim=768, patch_size=16, tubelet_size=2)
        return model, spec

    if model_name == "VideoMAE":
        if not args.videomae_ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {args.videomae_ckpt}")
        from VideoMAE import vit_base_patch16_224 as videomae_vit

        model = videomae_vit().to(device)
        model.from_pretrained(str(args.videomae_ckpt))
        model.eval()
        patch_size = _patch_size_int(getattr(model, "patch_size", 16))
        tubelet_size = int(getattr(model, "tubelet_size", 2))
        embed_dim = int(getattr(model, "embed_dim", 768))
        spec = ModelSpec(name=model_name, feature_dim=embed_dim, patch_size=patch_size, tubelet_size=tubelet_size)
        return model, spec

    if model_name == "DINOv2":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(device)
        model.eval()
        if hasattr(model, "head"):
            model.head = torch.nn.Identity()
        patch_size = _patch_size_int(getattr(model, "patch_size", 14))
        embed_dim = int(getattr(model, "embed_dim", 768))
        spec = ModelSpec(name=model_name, feature_dim=embed_dim, patch_size=patch_size, tubelet_size=1)
        return model, spec

    if model_name == "VJEPA2":
        model, _ = torch.hub.load("facebookresearch/vjepa2", "vjepa2_vit_large")
        model = model.to(device)
        model.eval()
        if hasattr(model, "norm"):
            model.norm = torch.nn.Identity()
        patch_size = _patch_size_int(getattr(model, "patch_size", 16))
        tubelet_size = int(getattr(model, "tubelet_size", 2))
        embed_dim = int(getattr(model, "embed_dim", 1024))
        spec = ModelSpec(name=model_name, feature_dim=embed_dim, patch_size=patch_size, tubelet_size=tubelet_size)
        return model, spec

    raise NotImplementedError(f"Unsupported model: {model_name}")


def encode_videomae_like(
    frames_c_t_h_w: torch.Tensor,
    model,
    spec: ModelSpec,
    resize_h: int,
    resize_w: int,
    device: torch.device,
) -> torch.Tensor:
    c, t, _, _ = frames_c_t_h_w.shape
    x = frames_c_t_h_w.permute(1, 0, 2, 3).contiguous()  # [T, C, H, W]
    x = F.interpolate(x, size=(resize_h, resize_w), mode="bicubic", align_corners=False)
    x = normalize_imagenet(x)
    x = x.permute(1, 0, 2, 3).unsqueeze(0).to(device)  # [1, C, T, H, W]

    with torch.no_grad():
        tokens = model(x)
        if isinstance(tokens, (tuple, list)):
            tokens = _tensor_from_model_output(tokens)
        if isinstance(tokens, dict):
            tokens = _tensor_from_model_output(tokens)

    tokens = tokens.squeeze(0)  # [N, C]
    t_out = t // spec.tubelet_size
    h_out = resize_h // spec.patch_size
    w_out = resize_w // spec.patch_size
    feat = tokens.reshape(t_out, h_out, w_out, -1).contiguous()  # [T, H, W, C]
    return feat.cpu()


def encode_dinov2(
    frames_c_t_h_w: torch.Tensor,
    model,
    spec: ModelSpec,
    resize_h: int,
    resize_w: int,
    chunk: int,
    device: torch.device,
) -> torch.Tensor:
    x = frames_c_t_h_w.permute(1, 0, 2, 3).contiguous()  # [T, C, H, W]
    x = F.interpolate(x, size=(resize_h, resize_w), mode="bicubic", align_corners=False)
    x = normalize_imagenet(x).to(device)

    outputs: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], chunk):
            batch = x[start : start + chunk]
            out = model.forward_features(batch)
            tok = _tensor_from_model_output(out)  # [B, N, C]
            outputs.append(tok)
    tokens = torch.cat(outputs, dim=0)

    t_out = tokens.shape[0]
    h_out = resize_h // spec.patch_size
    w_out = resize_w // spec.patch_size
    feat = tokens.reshape(t_out, h_out, w_out, -1).contiguous()  # [T, H, W, C]
    return feat.cpu()


def encode_vjepa2(
    frames_c_t_h_w: torch.Tensor,
    model,
    spec: ModelSpec,
    resize_h: int,
    resize_w: int,
    device: torch.device,
) -> torch.Tensor:
    c, t, _, _ = frames_c_t_h_w.shape
    x = frames_c_t_h_w.permute(1, 0, 2, 3).contiguous()  # [T, C, H, W]
    x = F.interpolate(x, size=(resize_h, resize_w), mode="bicubic", align_corners=False)
    x = normalize_imagenet(x)
    x = x.permute(1, 0, 2, 3).unsqueeze(0).to(device)  # [1, C, T, H, W]

    with torch.no_grad():
        out = model(x)
        tok = _tensor_from_model_output(out)

    if tok.ndim == 5:
        # [B, T, H, W, C]
        feat = tok.squeeze(0).contiguous()
        return feat.cpu()

    if tok.ndim != 3:
        raise RuntimeError(f"Unexpected VJEPA2 output shape: {tuple(tok.shape)}")

    tok = tok.squeeze(0)  # [N, C]
    t_out = t // spec.tubelet_size
    h_out = resize_h // spec.patch_size
    w_out = resize_w // spec.patch_size
    feat = tok.reshape(t_out, h_out, w_out, -1).contiguous()  # [T, H, W, C]
    return feat.cpu()


def encode_one_video(
    model_name: str,
    frames_c_t_h_w: torch.Tensor,
    model,
    spec: ModelSpec,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    if model_name in {"VideoMAEv2", "VideoMAE"}:
        return encode_videomae_like(
            frames_c_t_h_w=frames_c_t_h_w,
            model=model,
            spec=spec,
            resize_h=args.videomae_resize_h,
            resize_w=args.videomae_resize_w,
            device=device,
        )
    if model_name == "DINOv2":
        return encode_dinov2(
            frames_c_t_h_w=frames_c_t_h_w,
            model=model,
            spec=spec,
            resize_h=args.dinov2_resize_h,
            resize_w=args.dinov2_resize_w,
            chunk=args.dinov2_chunk,
            device=device,
        )
    if model_name == "VJEPA2":
        return encode_vjepa2(
            frames_c_t_h_w=frames_c_t_h_w,
            model=model,
            spec=spec,
            resize_h=args.videomae_resize_h,
            resize_w=args.videomae_resize_w,
            device=device,
        )
    raise NotImplementedError(model_name)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    if not args.video_dir.exists():
        raise FileNotFoundError(f"video_dir not found: {args.video_dir}")
    if not args.csv_path.exists():
        raise FileNotFoundError(f"csv_path not found: {args.csv_path}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    caption_map = load_caption_map(args.csv_path)
    videos = sorted(args.video_dir.glob("*.mp4"))
    if not videos:
        raise RuntimeError(f"No mp4 files found in {args.video_dir}")

    model_infos: List[Tuple[str, str, int]] = []

    for model_name in args.models:
        print(f"\n[Model] {model_name}")
        try:
            model, spec = load_model(model_name, args, device)
        except Exception as err:
            if args.allow_skip_missing:
                print(f"[Skip] {model_name}: {err}")
                continue
            raise

        out_dir = args.output_root / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        first_shape = None
        failed_videos: List[str] = []
        for video_path in videos:
            stem = video_path.stem
            caption = caption_map.get(stem, "")

            try:
                frames = sample_video_frames(video_path, num_frames=args.num_frames)
                feature = encode_one_video(model_name, frames, model, spec, args, device)
                if first_shape is None:
                    first_shape = tuple(feature.shape)

                save_path = out_dir / f"{stem}.pt"
                torch.save(
                    {
                        "feature": feature,
                        "caption": caption,
                        "video": video_path.name,
                        "model": model_name,
                    },
                    save_path,
                )
            except Exception as err:
                failed_videos.append(f"{video_path.name}: {err}")
                continue

        dim = first_shape[-1] if first_shape is not None else spec.feature_dim
        model_infos.append((model_name, str(first_shape), dim))
        print(f"[Done] {model_name}: {len(videos) - len(failed_videos)}/{len(videos)} files -> {out_dir}")
        if failed_videos:
            fail_log = out_dir / "failed_videos.txt"
            fail_log.write_text("\n".join(failed_videos), encoding="utf-8")
            print(f"[Warn] {model_name}: {len(failed_videos)} failed videos logged at {fail_log}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Dimension report
    report_path = args.output_root / "vfm_dimensions.txt"
    with report_path.open("w", encoding="utf-8") as file:
        file.write("VFM feature dimensions report\n")
        file.write(f"video_dir: {args.video_dir}\n")
        file.write(f"num_videos: {len(videos)}\n")
        file.write(f"num_frames(sampled): {args.num_frames}\n\n")
        for model_name, shape_str, dim in model_infos:
            file.write(f"- {model_name}\n")
            file.write(f"  sample_feature_shape: {shape_str}\n")
            file.write(f"  feature_dimension(C): {dim}\n\n")

    print(f"\nSaved dimension report: {report_path}")
    print(f"Output root: {args.output_root}")


if __name__ == "__main__":
    main()
