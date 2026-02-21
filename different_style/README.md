# different_style

本目錄提供「主體動作保留 + 主體外觀替換 + 背景替換」工具鏈。

---

## 檔案說明

- `edit_video_style.py`
  - 以 segmentation mask 保留主體運動軌跡
  - 可替換主體顏色材質感與背景
  - 支援 `fast / strict` 兩種模式

- `requirements.txt`
  - pip 套件列表

- `environment.yaml`
  - conda 環境設定檔

---

## 環境設定

在 `different_style` 目錄下執行：

```bash
conda env create -f environment.yaml
conda activate different-style
pip install -r requirements.txt
```

若環境已存在：

```bash
conda env update -f environment.yaml --prune
```

---

## 使用方法

### 1) 快速預覽（fast）

```bash
python edit_video_style.py \
  --input-video /home/hpc/ce505215/vrpro/dataset/motion_top100/0000.mp4 \
  --background-video /home/hpc/ce505215/vrpro/dataset/motion_top100/0001.mp4 \
  --output-video /home/hpc/ce505215/vrpro/different_style/out_fast.mp4 \
  --subject-color red \
  --subject-classes auto \
  --preset fast \
  --max-frames 120
```

### 2) 人像例子（strict）

```bash
python edit_video_style.py \
  --input-video /path/to/talking_person.mp4 \
  --background-video /home/hpc/ce505215/vrpro/dataset/motion_top100/0002.mp4 \
  --output-video /home/hpc/ce505215/vrpro/different_style/out_person_strict.mp4 \
  --subject-classes person \
  --subject-color '#d9b36c' \
  --preset strict \
  --material-strength 0.9 \
  --no-mask-action keep-original \
  --save-mask-video /home/hpc/ce505215/vrpro/different_style/mask_debug.mp4
```

---

## CLI 參數說明（argparse）

### `edit_video_style.py`

- `--input-video` (`str`, 必填)
  - 輸入影片路徑

- `--output-video` (`str`, 必填)
  - 輸出影片路徑

- `--background-image` (`str`, 預設 `None`)
  - 替換背景圖片；需與 `--background-video` 二擇一

- `--background-video` (`str`, 預設 `None`)
  - 替換背景影片；需與 `--background-image` 二擇一

- `--subject-color` (`str`, 預設 `red`)
  - 主體顏色，支援 `red/blue/green/yellow/black/white/silver` 或 `#RRGGBB`

- `--subject-classes` (`str`, 預設 `auto`)
  - 主體類別；可用 COCO 類別名稱或 id
  - 例：`person`、`person,car`、`0,2`、`auto`

- `--preset` (`fast|strict`, 預設 `fast`)
  - `fast`：速度優先
  - `strict`：時序穩定優先（含背景位移估計與遮罩平滑）

- `--model` (`str`, 預設 `None`)
  - YOLO segmentation 權重檔；未指定時依 `preset` 自動選擇

- `--conf` (`float`, 預設 `0.25`)
  - 偵測置信度門檻

- `--material-strength` (`float`, 預設 `0.8`)
  - 主體材質感/細節強化強度

- `--mask-feather` (`int`, 預設 `3`)
  - `strict` 模式遮罩羽化強度

- `--max-frames` (`int`, 預設 `-1`)
  - 最多處理幀數；`-1` 代表全部

- `--no-mask-action` (`keep-original|replace-background`, 預設 `keep-original`)
  - 抓不到主體時如何處理該幀

- `--min-mask-area-ratio` (`float`, 預設 `0.0003`)
  - 主體遮罩最小面積比，低於此值視為偵測失敗

- `--max-mask-area-ratio` (`float`, 預設 `0.70`)
  - 主體遮罩最大面積比，高於此值視為偵測失敗

- `--save-mask-video` (`str`, 預設 `None`)
  - 輸出除錯用 mask 影片

---

## 注意事項

- 第一次執行可能會下載 YOLO 權重。
- 這套方法主要是「動作保持 + 外觀替換」，不是完整身份語意重建。