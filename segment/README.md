# subject_motion_segment

這個模組把影片做成「主體導向」前處理，目標是讓後續 encoder 更聚焦在主體與其 motion。

核心流程：
1. 讀取 caption（`*.txt`）
2. 小型 LM 抽主體詞（只保留可框選實體）
3. GroundingDINO 找候選框
4. 用運動量 + 時序一致性挑主體框（抑制大場景、字幕）
5. SAM 做 keyframe segmentation
6. 產生完整長度 `overlay.mp4`：即使每 `N` 幀/每 `x` 秒才偵測一次，也會在中間幀對主體框做線性插值與平滑

---

## 功能重點

- **主體優先**：用 `primary_words` 先挑主體，再可選少量 context。
- **字幕/大框抑制**：對底部超寬框（常見字幕）與過大框（整片草原/球場）做懲罰。
- **motion-aware 選框**：用 optical flow 的框內運動量做分數。
- **跨幀平滑**：
  - ranking 加入前一幀 IoU
  - EMA 平滑框座標
  - 短暫漏檢時沿用前一框
- **完整影片可視化輸出**：`overlay.mp4` 為原影片幀率與幀數，中間幀框由前後 keyframe 內插。

---

## 檔案輸出

每支影片輸出：

```text
<output_dir>/<video_stem>/
  frames/000123.jpg ...     # keyframe 可視化圖（依抽樣規則）
  masks/000123.png ...      # keyframe mask（可選）
  frames.jsonl              # keyframe偵測紀錄
  summary.json              # 該影片摘要
  overlay.mp4               # 完整影片長度可視化（框平滑內插）
```

批次摘要：

```text
<output_dir>/run_summary.json
```

---

## 環境建立

```bash
cd /home/hpc/ce505215/vrpro/segment
conda env create -f environment.yaml
conda activate vrpro-segment
```

若已存在：

```bash
conda env update -f environment.yaml --prune
```

---

## 程式用法

### 1) 每 0.2 秒取樣（但輸出完整 overlay 影片）

```bash
cd /home/hpc/ce505215/vrpro
python segment/subject_motion_segment.py \
  --video_dir dataset/motion_top100_random5 \
  --caption_dir dataset/motion_top100_random5 \
  --video_glob "*.mp4" \
  --output_dir dataset/motion_top100_segmented_random5_tuned_0p2s_full \
  --subject_mode lm \
  --sample_interval_sec 0.2 \
  --save_mask_png \
  --save_overlay_video
```

### 2) 每 0.5 秒取樣

```bash
python segment/subject_motion_segment.py \
  --video_dir dataset/motion_top100_random5 \
  --caption_dir dataset/motion_top100_random5 \
  --output_dir dataset/motion_top100_segmented_random5_tuned_0p5s_full \
  --sample_interval_sec 0.5 \
  --save_mask_png --save_overlay_video
```

---

## Arg Parser 說明

### I/O
- `--video_dir`：影片資料夾
- `--caption_dir`：caption 資料夾（預設同 `video_dir`）
- `--output_dir`：輸出根目錄
- `--video_glob`：影片匹配規則（預設 `*.mp4`）

### 資料量
- `--max_videos`：最多處理幾支（0=全部）
- `--max_frames_per_video`：最多處理幾個 sampled 幀（0=全部）

### 抽樣
- `--frame_stride`：每 N 幀取一幀
- `--sample_interval_sec`：每幾秒取一幀（>0 時覆蓋 `frame_stride`）

### 主體詞抽取
- `--subject_mode`：`lm` 或 `heuristic`
- `--small_lm`：預設 `google/flan-t5-small`

### 偵測/分割模型
- `--detector_model`：GroundingDINO 模型
- `--sam_model`：SAM 模型
- `--box_threshold`、`--text_threshold`：GroundingDINO 門檻

### 主體過濾
- `--min_box_area_ratio`：太小框過濾
- `--max_box_area_ratio`：太大框懲罰/抑制
- `--subtitle_bottom_ratio`：底部區域比例
- `--subtitle_aspect_ratio`：字幕型寬高比閾值

### Motion + 時序穩定
- `--motion_weight`：motion 分數權重
- `--temporal_iou_weight`：與前框 IoU 權重
- `--min_motion_score`：低 motion 的懲罰門檻
- `--smooth_alpha`：EMA 平滑強度
- `--max_center_jump_ratio`：限制跨幀中心跳動
- `--track_hold_frames`：短暫漏檢持續沿用前框幀數

### 輸出
- `--save_mask_png`：輸出 mask png
- `--save_overlay_video`：輸出 `overlay.mp4`
- `--device`：`cuda`/`cpu`

---

## 建議參數（減少草原/字幕誤框）

```text
--max_box_area_ratio 0.35
--subtitle_bottom_ratio 0.25
--subtitle_aspect_ratio 4.0
--motion_weight 0.9
--temporal_iou_weight 0.8
--min_motion_score 0.12
--smooth_alpha 0.6
--max_center_jump_ratio 0.18
--track_hold_frames 5
```
