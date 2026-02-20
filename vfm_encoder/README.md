# vfm_encoder

本目錄提供一套「先抽 VFM 特徵，再做 motion-focused 壓縮」的工具鏈，包含：

1. 從影片抽取 VFM 特徵（`vfm_feature.py`）
2. 訓練 motion autoencoder（`train.py`）
3. 將特徵轉成低維 latent / motion embedding（`encode.py`）

---

## 檔案說明

- `model.py`
  - 核心模型定義：`MotionFeatureAutoencoder`、`MotionFeatureEncoder`、`MotionFeatureDecoder`
  - 架構為 Temporal Conv + Temporal Transformer
  - 輸入支援 `[B, T, N, C]` 與 `[B, T, H, W, C]`

- `losses.py`
  - `motion_autoencoder_loss`
  - 組成：`reconstruction loss` + `temporal difference loss` + `latent smoothness`

- `train.py`
  - 訓練腳本
  - 讀取 `.pt/.pth/.npy` 特徵檔，並訓練 `MotionFeatureAutoencoder`
  - 預期每個樣本 shape 為 `[T, N, C]` 或 `[T, H, W, C]`

- `encode.py`
  - 推論腳本（單一特徵檔）
  - 輸出 `latent`、`motion_embedding`、`compression_ratio`

- `vfm_feature.py`
  - 影片轉 VFM 特徵腳本（可直接對 `dataset/motion_top100` 使用）
  - 支援模型：`VideoMAEv2`、`VideoMAE`、`DINOv2`、`VJEPA2`
  - 輸出路徑預設為 `dataset/vfm_features_motion_top100/<model_name>/*.pt`
  - 會生成 `vfm_dimensions.txt`（記錄特徵維度）

- `__init__.py`
  - 封裝模組匯出符號

- `envs/environment.train.yaml`
  - `train.py` 使用環境

- `envs/environment.encode.yaml`
  - `encode.py` 使用環境

- `envs/environment.vfm_feature.yaml`
  - `vfm_feature.py` 使用環境（包含影片解碼與 VFM 相關套件）

---

## 環境設定

### 腳本與環境對應

- `train.py` -> `envs/environment.train.yaml`
- `encode.py` -> `envs/environment.encode.yaml`
- `vfm_feature.py` -> `envs/environment.vfm_feature.yaml`

### 建立環境

在 `vfm_encoder` 目錄下執行：

```bash
conda env create -f envs/environment.train.yaml
conda env create -f envs/environment.encode.yaml
conda env create -f envs/environment.vfm_feature.yaml
```

若環境已存在，改用 update：

```bash
conda env update -f envs/environment.train.yaml --prune
conda env update -f envs/environment.encode.yaml --prune
conda env update -f envs/environment.vfm_feature.yaml --prune
```

### 啟用環境

```bash
conda activate vfm-train
conda activate vfm-encode
conda activate vfm-feature
```

說明：

- 若你只跑訓練與編碼，使用 `vfm-train` / `vfm-encode` 即可
- 若你要從影片抽 VFM 特徵，請使用 `vfm-feature`

---

## 資料格式

### `train.py` 可直接讀取的格式

每個檔案可為 `.pt/.pth/.npy`，內容需至少包含特徵 tensor：

- tensor shape：`[T, N, C]` 或 `[T, H, W, C]`
- 若為 dict，優先讀取 key：`feature`、`features`，否則讀第一個 key

`vfm_feature.py` 產生的 `.pt` 內容包含：

- `feature`: `torch.Tensor`
- `caption`: `str`
- `video`: `str`
- `model`: `str`

---

## 使用方法

### 1) 從影片抽 VFM 特徵

在 `vfm_encoder` 目錄下執行：

```bash
conda activate vfm-feature
python vfm_feature.py --allow_skip_missing
```

只抽指定模型（例如 VideoMAEv2）：

```bash
conda activate vfm-feature
python vfm_feature.py \
  --models VideoMAEv2 \
  --allow_skip_missing
```

常用參數：

- `--video_dir`：影片資料夾（預設 `../dataset/motion_top100`）
- `--csv_path`：caption csv（預設 `../dataset/motion_top100.csv`）
- `--output_root`：特徵輸出根目錄
- `--models`：要抽的 VFM 模型列表

### 2) 用 VFM 特徵訓練 motion encoder

```bash
conda activate vfm-train
python train.py \
  --feature_dir ../dataset/vfm_features_motion_top100/VideoMAEv2 \
  --save_dir ./checkpoints_videomaev2 \
  --input_dim 768 \
  --latent_dim 192 \
  --batch_size 1 \
  --epochs 10 \
  --num_workers 0 \
  --use_token
```

### 3) 將單一特徵檔壓縮成 latent

```bash
conda activate vfm-encode
python encode.py \
  --checkpoint ./checkpoints_videomaev2/motion_ae_epoch10.pt \
  --input ../dataset/vfm_features_motion_top100/VideoMAEv2/0000.pt \
  --output ../dataset/vfm_features_motion_top100/VideoMAEv2_motion/0000.pt \
  --input_dim 768 \
  --latent_dim 192 \
  --use_token
```

---

## CLI 參數說明（argparse）

本節整理所有有 `argparse` 的腳本參數。

### `train.py`

- `--feature_dir` (`Path`, 必填)
  - 訓練特徵資料夾，遞迴讀取 `.pt/.pth/.npy`
- `--save_dir` (`Path`, 預設 `./checkpoints`)
  - checkpoint 輸出資料夾
- `--input_dim` (`int`, 預設 `768`)
  - 輸入特徵通道維度 `C`
- `--model_dim` (`int`, 預設 `384`)
  - 編碼器/解碼器中間維度
- `--latent_dim` (`int`, 預設 `192`)
  - 壓縮後維度
- `--num_heads` (`int`, 預設 `8`)
  - Transformer attention heads
- `--num_layers` (`int`, 預設 `2`)
  - Transformer layer 數
- `--dropout` (`float`, 預設 `0.1`)
  - 模型 dropout
- `--epochs` (`int`, 預設 `10`)
  - 訓練 epochs
- `--batch_size` (`int`, 預設 `2`)
  - batch size
- `--lr` (`float`, 預設 `2e-4`)
  - learning rate
- `--weight_decay` (`float`, 預設 `1e-4`)
  - optimizer weight decay
- `--num_workers` (`int`, 預設 `4`)
  - DataLoader workers
- `--recon_weight` (`float`, 預設 `1.0`)
  - reconstruction loss 權重
- `--motion_weight` (`float`, 預設 `1.0`)
  - temporal difference loss 權重
- `--smooth_weight` (`float`, 預設 `0.05`)
  - latent smoothness loss 權重
- `--use_token` (`flag`, 預設關閉)
  - 開啟後 encoder 輸入為 `[x, delta_x]` 拼接

### `encode.py`

- `--checkpoint` (`Path`, 必填)
  - 訓練完成的模型權重檔
- `--input` (`Path`, 必填)
  - 單一輸入特徵檔（`.pt/.pth/.npy`）
- `--output` (`Path`, 必填)
  - 輸出路徑（儲存 `latent`、`motion_embedding`）
- `--input_dim` (`int`, 預設 `768`)
  - 輸入特徵通道維度 `C`
- `--model_dim` (`int`, 預設 `384`)
  - 中間維度
- `--latent_dim` (`int`, 預設 `192`)
  - 壓縮維度
- `--num_heads` (`int`, 預設 `8`)
  - Transformer attention heads
- `--num_layers` (`int`, 預設 `2`)
  - Transformer layer 數
- `--dropout` (`float`, 預設 `0.1`)
  - dropout
- `--use_token` (`flag`, 預設關閉)
  - 與訓練時設定一致時應開啟

### `vfm_feature.py`

- `--video_dir` (`Path`, 預設 `../dataset/motion_top100`)
  - 影片來源資料夾
- `--csv_path` (`Path`, 預設 `../dataset/motion_top100.csv`)
  - caption CSV（用於寫入輸出 metadata）
- `--output_root` (`Path`, 預設 `../dataset/vfm_features_motion_top100`)
  - 特徵輸出根目錄
- `--models` (`str[]`, 預設 `VideoMAEv2 VideoMAE DINOv2 VJEPA2`)
  - 要抽取的 VFM 模型列表
- `--device` (`str`, 預設自動：有 GPU 用 `cuda`，否則 `cpu`)
  - 推論裝置
- `--num_frames` (`int`, 預設 `48`)
  - 每支影片抽樣幀數
- `--videomae_resize_h` (`int`, 預設 `160`)
  - VideoMAE/VideoMAEv2/VJEPA2 前處理高度
- `--videomae_resize_w` (`int`, 預設 `240`)
  - VideoMAE/VideoMAEv2/VJEPA2 前處理寬度
- `--dinov2_resize_h` (`int`, 預設 `420`)
  - DINOv2 前處理高度
- `--dinov2_resize_w` (`int`, 預設 `630`)
  - DINOv2 前處理寬度
- `--dinov2_chunk` (`int`, 預設 `128`)
  - DINOv2 分塊推論大小（避免 OOM）
- `--videomaev2_ckpt` (`Path`, 預設 `VideoREPA/ckpt/VideoMAEv2/vit_b_k710_dl_from_giant.pth`)
  - VideoMAEv2 權重路徑
- `--videomae_ckpt` (`Path`, 預設 `VideoREPA/ckpt/VideoMAE/k400_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0_9_e1600.pth`)
  - VideoMAE 權重路徑
- `--allow_skip_missing` (`flag`, 預設關閉)
  - 模型缺權重/缺依賴時改為跳過，不中斷整批

---

## 維度建議

當 VFM channel dimension 為 `C=768`：

- `latent_dim=192`：1/4 壓縮（推薦起始）
- `latent_dim=128`：更強壓縮
- `latent_dim=256`：較保守、重建通常較穩定

---

## 常見問題

- 影片解碼失敗：
  - `vfm_feature.py` 會嘗試多種 backend，失敗影片會記錄在各模型輸出資料夾的 `failed_videos.txt`

- 某些 VFM 被 `--allow_skip_missing` 跳過：
  - 通常是 checkpoint 或相依套件未安裝

- `vfm_dimensions.txt` 內容被覆蓋：
  - 目前行為是每次執行重寫；若需保留歷史結果可再擴充成 append 模式
