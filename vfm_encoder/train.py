from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from losses import motion_autoencoder_loss
from model import MotionFeatureAutoencoder


class FeatureDataset(Dataset):
    def __init__(self, feature_files: List[Path]):
        self.feature_files = feature_files

    def __len__(self) -> int:
        return len(self.feature_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.feature_files[idx]
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

        # Expected sample shape: [T, N, C] or [T, H, W, C]
        if x.ndim not in {3, 4}:
            raise ValueError(f"Feature should be [T,N,C] or [T,H,W,C], got {tuple(x.shape)} in {path}")
        return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train motion-focused bottleneck autoencoder on VFM features")
    parser.add_argument("--feature_dir", type=Path, required=True, help="Directory with .pt/.pth/.npy feature files")
    parser.add_argument("--save_dir", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--input_dim", type=int, default=768)
    parser.add_argument("--model_dim", type=int, default=384)
    parser.add_argument("--latent_dim", type=int, default=192)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--recon_weight", type=float, default=1.0)
    parser.add_argument("--motion_weight", type=float, default=1.0)
    parser.add_argument("--smooth_weight", type=float, default=0.05)
    parser.add_argument("--use_token", action="store_true", help="Use [x, delta_x] as encoder input")
    return parser.parse_args()


def collate_features(batch: List[torch.Tensor]) -> torch.Tensor:
    # Assume same shape in a batch (typical for fixed train resolution).
    return torch.stack(batch, dim=0)


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    feature_files = sorted([p for p in args.feature_dir.rglob("*") if p.suffix in {".pt", ".pth", ".npy"}])
    if not feature_files:
        raise RuntimeError(f"No feature files found under: {args.feature_dir}")

    dataset = FeatureDataset(feature_files)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_features,
    )

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"[Info] Compression ratio: {model.compression_ratio:.3f} ({args.input_dim} -> {args.latent_dim})")
    print(f"[Info] Num files: {len(dataset)}")

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in loader:
            x = batch.to(device, non_blocking=True)
            output = model(x)

            loss, logs = motion_autoencoder_loss(
                x,
                output.reconstruction,
                output.latent,
                recon_weight=args.recon_weight,
                motion_weight=args.motion_weight,
                smooth_weight=args.smooth_weight,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += loss.item()
            global_step += 1

            if global_step % 50 == 0:
                print(
                    f"[Epoch {epoch}] step={global_step} "
                    f"loss={logs['loss_total'].item():.4f} "
                    f"recon={logs['loss_recon'].item():.4f} "
                    f"motion={logs['loss_motion'].item():.4f}"
                )

        mean_loss = running / max(1, len(loader))
        ckpt_path = args.save_dir / f"motion_ae_epoch{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "config": vars(args),
                "mean_loss": mean_loss,
            },
            ckpt_path,
        )
        print(f"[Epoch {epoch}] mean loss={mean_loss:.4f}, checkpoint={ckpt_path}")


if __name__ == "__main__":
    main()
