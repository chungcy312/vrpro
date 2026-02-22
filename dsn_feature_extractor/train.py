from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from losses import dsn_pair_loss
from model import DSNFeatureExtractor


def _load_feature(path: Path) -> torch.Tensor:
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
        raise ValueError(f"Feature should be [T,C] / [T,N,C] / [T,H,W,C], got {tuple(x.shape)} from {path}")
    if x.ndim == 2:
        x = x.unsqueeze(1)  # [T,C] -> [T,1,C]
    return x


def _align_pair(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    ta = a.shape[0]
    tb = b.shape[0]
    t = min(ta, tb)
    a = a[:t]
    b = b[:t]

    if a.ndim == 3 and b.ndim == 3 and a.shape[1] != b.shape[1]:
        n = min(a.shape[1], b.shape[1])
        a = a[:, :n]
        b = b[:, :n]

    if a.shape[-1] != b.shape[-1]:
        raise ValueError(f"Channel mismatch: {a.shape[-1]} vs {b.shape[-1]}")
    return a, b


class PairFeatureDataset(Dataset):
    def __init__(self, pairs: List[Tuple[Path, Path]], labels: Optional[List[int]] = None):
        self.pairs = pairs
        self.labels = labels

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        a_path, b_path = self.pairs[idx]
        a = _load_feature(a_path)
        b = _load_feature(b_path)
        a, b = _align_pair(a, b)

        sample: Dict[str, torch.Tensor] = {
            "a": a,
            "b": b,
        }
        if self.labels is not None:
            sample["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample


def collate_pair(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    a = torch.stack([item["a"] for item in batch], dim=0)
    b = torch.stack([item["b"] for item in batch], dim=0)
    out = {"a": a, "b": b}
    if "label" in batch[0]:
        out["label"] = torch.stack([item["label"] for item in batch], dim=0)
    return out


def build_pairs_from_manifest(manifest_path: Path) -> Tuple[List[Tuple[Path, Path]], Optional[List[int]]]:
    pairs: List[Tuple[Path, Path]] = []
    labels: List[int] = []

    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"path_a", "path_b"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("Manifest must contain columns: path_a,path_b[,label]")

        has_label = "label" in (reader.fieldnames or [])
        for row in reader:
            pairs.append((Path(row["path_a"]), Path(row["path_b"])))
            if has_label:
                labels.append(int(row["label"]))

    return pairs, labels if labels else None


def build_pairs_from_dirs(dir_a: Path, dir_b: Path) -> Tuple[List[Tuple[Path, Path]], None]:
    exts = {".pt", ".pth", ".npy"}
    a_files = sorted([p for p in dir_a.rglob("*") if p.suffix in exts])

    pairs: List[Tuple[Path, Path]] = []
    for a in a_files:
        rel = a.relative_to(dir_a)
        b = dir_b / rel
        if b.exists() and b.suffix in exts:
            pairs.append((a, b))

    return pairs, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DSN on paired VFM latents (shared=motion, private=texture)")

    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--pair_manifest", type=Path, help="CSV with columns path_a,path_b[,label]")
    data_group.add_argument("--pair_dirs", nargs=2, type=Path, metavar=("DIR_A", "DIR_B"), help="Two dirs with mirrored file structure")

    parser.add_argument("--save_dir", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--input_dim", type=int, default=768)
    parser.add_argument("--hidden_dim", type=int, default=384)
    parser.add_argument("--shared_dim", type=int, default=128)
    parser.add_argument("--private_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--recon_weight", type=float, default=1.0)
    parser.add_argument("--shared_weight", type=float, default=1.0)
    parser.add_argument("--orth_weight", type=float, default=0.1)
    parser.add_argument("--private_margin_weight", type=float, default=0.1)
    parser.add_argument("--private_margin", type=float, default=1.0)

    parser.add_argument("--use_classifier_loss", action="store_true", help="Enable optional classifier loss if labels exist")
    parser.add_argument("--classifier_weight", type=float, default=0.0)
    parser.add_argument("--num_classes", type=int, default=2)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    if args.pair_manifest is not None:
        pairs, labels = build_pairs_from_manifest(args.pair_manifest)
    else:
        pair_dirs = args.pair_dirs
        assert pair_dirs is not None
        pairs, labels = build_pairs_from_dirs(pair_dirs[0], pair_dirs[1])

    if not pairs:
        raise RuntimeError("No valid feature pairs found")

    if not args.use_classifier_loss:
        labels = None

    dataset = PairFeatureDataset(pairs, labels=labels)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_pair,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DSNFeatureExtractor(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        shared_dim=args.shared_dim,
        private_dim=args.private_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_classes=args.num_classes if args.use_classifier_loss else 0,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in loader:
            x_a = batch["a"].to(device, non_blocking=True)
            x_b = batch["b"].to(device, non_blocking=True)
            labels_t = batch.get("label")
            if labels_t is not None:
                labels_t = labels_t.to(device, non_blocking=True)

            output = model(x_a, x_b)
            loss, logs = dsn_pair_loss(
                x_a=x_a,
                x_b=x_b,
                recon_a=output.recon_a,
                recon_b=output.recon_b,
                shared_a=output.shared_a,
                shared_b=output.shared_b,
                private_a=output.private_a,
                private_b=output.private_b,
                logits_a=output.logits_a,
                logits_b=output.logits_b,
                labels=labels_t,
                recon_weight=args.recon_weight,
                shared_weight=args.shared_weight,
                orth_weight=args.orth_weight,
                private_margin_weight=args.private_margin_weight,
                private_margin=args.private_margin,
                classifier_weight=args.classifier_weight if args.use_classifier_loss else 0.0,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % 50 == 0:
                print(
                    f"[Epoch {epoch}] step={global_step} "
                    f"total={logs['loss_total'].item():.4f} "
                    f"recon={logs['loss_recon'].item():.4f} "
                    f"shared={logs['loss_shared'].item():.4f} "
                    f"orth={logs['loss_orth'].item():.4f} "
                    f"pmargin={logs['loss_private_margin'].item():.4f} "
                    f"clf={logs['loss_classifier'].item():.4f}"
                )

        mean_loss = epoch_loss / max(1, len(loader))
        ckpt_path = args.save_dir / f"dsn_epoch{epoch}.pt"
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
