# VFM Motion Encoder/Decoder

這個模組用來對 VFM 輸出做第二次特徵提取，目標是保留 **motion 相關資訊** 並做維度壓縮。

## 模型選型：Transformer 還是 Convolution？

建議採用 **Hybrid（Temporal Conv + Temporal Transformer）**：

- `Temporal Conv1D`：擅長抓局部短期動作變化（速度、加速度感）。
- `Temporal Transformer`：擅長抓長期時序依賴（跨幀關聯、動作階段）。
- 純 conv 會較難抓長程時序；純 transformer 參數量較高且對局部平滑運動偏不穩。

目前實作：`MotionFeatureAutoencoder`。

## 輸入/輸出格式

- 輸入：
  - `[B, T, N, C]`（token 形式）或
  - `[B, T, H, W, C]`（grid 形式）
- Encoder 輸出：同 layout 的低維 latent，通道維 `C -> latent_dim`
- Decoder 輸出：重建回原始 `C`
- 額外輸出：`motion_embedding`（對空間 token 平均，shape 約為 `[B, T, latent_dim]`）

## 維度建議（壓縮感）

若 VFM 輸出通道是 `C=768`：

- 建議 `latent_dim=192`（1/4）作為第一版
- 追求更強壓縮可試 `128`（1/6）
- 若想保守先穩定可試 `256`（1/3）

## Loss 設計

`motion_autoencoder_loss` 包含三項：

1. `recon loss`：重建原始特徵（避免資訊崩壞）
2. `motion loss`：重建 temporal difference（強化運動訊息）
3. `smooth loss`：限制 latent 在時間上的噪聲抖動

## 快速開始

### 1) 訓練

```bash
cd /home/hpc/ce505215/vrpro/vfm_encoder
python train.py \
  --feature_dir /path/to/vfm_features \
  --save_dir ./checkpoints \
  --input_dim 768 \
  --latent_dim 192 \
  --epochs 10 \
  --batch_size 2 \
  --use_token
```

### 2) 將單一特徵檔壓縮

```bash
python encode.py \
  --checkpoint ./checkpoints/motion_ae_epoch10.pt \
  --input /path/to/one_feature.pt \
  --output /path/to/one_feature_motion.pt \
  --input_dim 768 \
  --latent_dim 192 \
  --use_token
```

## 如何接進 VideoREPA

你現在在 `lora_trainer.py` 裡的 `align_target` 是 VFM 特徵。可在計算 loss 前加一層：

- `align_target -> motion_encoder -> compact_align_target`
- 再讓 `align`（CogVideoX features）對齊 `compact_align_target`

這樣可讓對齊更偏重 motion 相關訊息，同時降低特徵維度。
