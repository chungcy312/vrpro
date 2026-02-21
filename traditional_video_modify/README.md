# traditional_video_modify

本目錄提供傳統影像風格擾動工具，用來生成「動作一致、外觀不同」的訓練影片。

核心方法：

1. Random Color Jitter（brightness/contrast/saturation/hue）
2. Color Temperature shift（暖色/冷色）
3. RGB channel intensity gains（R/G/B 通道增益）

---

## 檔案說明

- `traditional_augment.py`
  - 批次處理 `.mp4`
  - 產生多個 variant（`v01/v02/...`）
  - 輸出 manifest（`jsonl + csv`）

- `requirements.txt`
  - pip 套件列表

- `environment.yaml`
  - conda 環境設定檔

---

## 環境設定

在 `traditional_video_modify` 目錄：

```bash
conda env create -f environment.yaml
conda activate traditional-video-modify
pip install -r requirements.txt
```

若環境已存在：

```bash
conda env update -f environment.yaml --prune
```

---

## 使用方法

### 1) random5 快速測試

```bash
python traditional_augment.py \
  --input-dir /home/hpc/ce505215/vrpro/dataset/motion_top100_random5 \
  --output-dir /home/hpc/ce505215/vrpro/dataset/motion_top100_random5_traditional \
  --strength mild \
  --variants 2 \
  --max-frames 120 \
  --temporal-jitter 0.04 \
  --force-visible-change \
  --overwrite
```

### 2) 跑整個 motion_top100

```bash
python traditional_augment.py \
  --input-dir /home/hpc/ce505215/vrpro/dataset/motion_top100 \
  --output-dir /home/hpc/ce505215/vrpro/dataset/motion_top100_traditional \
  --strength mild \
  --variants 2 \
  --temporal-jitter 0.04 \
  --force-visible-change \
  --overwrite
```

---

## 輸出格式

- `output_dir/v01/*.mp4`, `output_dir/v02/*.mp4`, ...
- `output_dir/manifest.jsonl`
- `output_dir/manifest.csv`

`manifest` 會記錄每個輸出影片使用的 base 風格參數（brightness/contrast/saturation/hue_shift/temperature/red_gain/green_gain/blue_gain）。

---

## CLI 參數說明（argparse）

### `traditional_augment.py`

- `--input-dir` (`Path`, 必填)
  - 輸入影片資料夾

- `--output-dir` (`Path`, 必填)
  - 輸出資料夾

- `--glob` (`str`, 預設 `*.mp4`)
  - 搜尋輸入影片的 glob pattern

- `--strength` (`mild|medium`, 預設 `mild`)
  - 隨機參數範圍強度

- `--variants` (`int`, 預設 `2`)
  - 每支影片產生幾個風格版本

- `--max-videos` (`int`, 預設 `-1`)
  - 限制處理影片數量；`-1` 代表全部

- `--max-frames` (`int`, 預設 `-1`)
  - 限制每支輸出影片的幀數；`-1` 代表全部

- `--seed` (`int`, 預設 `2026`)
  - 隨機種子（可重現）

- `--temporal-jitter` (`float`, 預設 `0.04`)
  - 每幀參數微擾強度；越小越穩定、越保留動作可辨識性

- `--overwrite` (`flag`, 預設關閉)
  - 開啟後覆蓋已存在輸出

- `--force-visible-change` (`flag`, 預設關閉)
  - 強制參數與原片保持最小可見差異，避免看起來「幾乎沒變」

- `--min-change-level` (`float`, 預設 `1.0`)
  - `--force-visible-change` 的強度倍率；可設 `1.2~1.5` 提高差異

---

## 動作保留建議

- 優先使用 `--strength mild`
- 建議 `--temporal-jitter` 在 `0.02 ~ 0.06`
- 若要極度保守，可把 `--temporal-jitter` 降到 `0.01`
- 若需要肉眼明顯差異，請加 `--force-visible-change --min-change-level 1.2`
