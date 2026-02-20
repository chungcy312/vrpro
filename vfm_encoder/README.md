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
/home/hpc/ce505215/.conda/envs/cogvideox/bin/python vfm_feature.py --allow_skip_missing
```

只抽指定模型（例如 VideoMAEv2）：

```bash
/home/hpc/ce505215/.conda/envs/cogvideox/bin/python vfm_feature.py \
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
/home/hpc/ce505215/.conda/envs/cogvideox/bin/python train.py \
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
/home/hpc/ce505215/.conda/envs/cogvideox/bin/python encode.py \
  --checkpoint ./checkpoints_videomaev2/motion_ae_epoch10.pt \
  --input ../dataset/vfm_features_motion_top100/VideoMAEv2/0000.pt \
  --output ../dataset/vfm_features_motion_top100/VideoMAEv2_motion/0000.pt \
  --input_dim 768 \
  --latent_dim 192 \
  --use_token
```

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
