from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from model import DSNFeatureExtractor


def load_feature(path: Path) -> torch.Tensor:
    if path.suffix in {".pt", ".pth"}:
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict):
            if "feature" in data:
                data = data["feature"]
            elif "features" in data:
                data = data["features"]
            else:
                first_key = next(iter(data.keys()))
                data = data[first_key]
        x = torch.as_tensor(data, dtype=torch.float32)
    elif path.suffix == ".npy":
        x = torch.from_numpy(np.load(path)).float()
    else:
        raise ValueError(f"Unsupported suffix: {path.suffix}")

    if x.ndim not in {2, 3, 4}:
        raise ValueError(f"Feature should be [T,C] / [T,N,C] / [T,H,W,C], got {tuple(x.shape)}")
    if x.ndim == 2:
        x = x.unsqueeze(1)
    return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode a feature with DSN into shared/private embeddings")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to DSN checkpoint")
    parser.add_argument("--input", type=Path, required=True, help="Input feature file (.pt/.pth/.npy)")
    parser.add_argument("--output", type=Path, required=True, help="Output path for encoded result (.pt)")

    parser.add_argument("--input_dim", type=int, default=768)
    parser.add_argument("--hidden_dim", type=int, default=384)
    parser.add_argument("--shared_dim", type=int, default=128)
    parser.add_argument("--private_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DSNFeatureExtractor(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        shared_dim=args.shared_dim,
        private_dim=args.private_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_classes=0,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    x = load_feature(args.input).unsqueeze(0).to(device)

    with torch.no_grad():
        shared, private = model.encode(x)
        recon = model.reconstruct(shared, private, ref_shape=x.shape)

    torch.save(
        {
            "shared": shared.squeeze(0).cpu(),
            "private": private.squeeze(0).cpu(),
            "reconstruction": recon.squeeze(0).cpu(),
            "input": str(args.input),
            "checkpoint": str(args.checkpoint),
        },
        args.output,
    )

    print(f"Saved DSN encoding to: {args.output}")


if __name__ == "__main__":
    main()
