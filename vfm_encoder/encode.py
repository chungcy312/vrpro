from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from model import MotionFeatureAutoencoder


def load_feature(path: Path) -> torch.Tensor:
    if path.suffix in {".pt", ".pth"}:
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict):
            if "feature" in data:
                data = data["feature"]
            elif "features" in data:
                data = data["features"]
            else:
                data = data[next(iter(data.keys()))]
        return torch.as_tensor(data, dtype=torch.float32)
    if path.suffix == ".npy":
        return torch.from_numpy(np.load(path)).float()
    raise ValueError(f"Unsupported suffix: {path.suffix}")


def save_tensor(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".npy":
        np.save(path, tensor.cpu().numpy())
    else:
        torch.save(tensor.cpu(), path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode VFM features into compact motion embeddings")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Input .pt/.pth/.npy feature file")
    parser.add_argument("--output", type=Path, required=True, help="Output file (.pt/.pth/.npy)")
    parser.add_argument("--input_dim", type=int, default=768)
    parser.add_argument("--model_dim", type=int, default=384)
    parser.add_argument("--latent_dim", type=int, default=192)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_token", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MotionFeatureAutoencoder(
        input_dim=args.input_dim,
        model_dim=args.model_dim,
        latent_dim=args.latent_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_token=args.use_token,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    feature = load_feature(args.input)
    if feature.ndim not in {3, 4}:
        raise ValueError(f"Expected [T,N,C] or [T,H,W,C], got {tuple(feature.shape)}")

    with torch.no_grad():
        x = feature.unsqueeze(0).to(device)
        output = model(x)
        latent = output.latent.squeeze(0)
        motion_embedding = output.motion_embedding.squeeze(0)

    payload = {
        "latent": latent.cpu(),
        "motion_embedding": motion_embedding.cpu(),
        "compression_ratio": model.compression_ratio,
    }
    save_tensor(args.output, payload)
    print(f"Saved compact features to {args.output}")
    print(f"Compression ratio: {model.compression_ratio:.3f}")


if __name__ == "__main__":
    main()
