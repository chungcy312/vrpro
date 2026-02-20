# Caption-guided subject segmentation for Panda-70M motion top100

這個流程會做：
1. 讀取每支影片的 caption (`*.txt`)。
2. 用小型 LM（預設 `google/flan-t5-small`）抽主體與可互動背景（例如球 + 地板）。
3. 用 GroundingDINO 逐幀找框（bbox）。
4. 用 SAM 依 bbox 逐幀分割 mask。
5. 輸出可視化幀與 frame-level json，方便後續餵給 motion encoder。

## 安裝

```bash
cd /home/hpc/ce505215/vrpro
pip install -r segment/requirements.txt
```

## 跑 `motion_top100`（你目前下載好的 4 位數命名資料）

```bash
python segment/subject_motion_segment.py \
  --video_dir /home/hpc/ce505215/vrpro/dataset/motion_top100 \
  --caption_dir /home/hpc/ce505215/vrpro/dataset/motion_top100 \
  --video_glob "*.mp4" \
  --output_dir /home/hpc/ce505215/vrpro/dataset/motion_top100_segmented \
  --subject_mode lm \
  --frame_stride 1 \
  --save_mask_png \
  --save_overlay_video
```

## 跑 `panda70m_motion_top100/00000`（8 位數命名資料）

```bash
python segment/subject_motion_segment.py \
  --video_dir /home/hpc/ce505215/vrpro/dataset/panda70m_motion_top100/00000 \
  --caption_dir /home/hpc/ce505215/vrpro/dataset/panda70m_motion_top100/00000 \
  --video_glob "*.mp4" \
  --output_dir /home/hpc/ce505215/vrpro/dataset/panda70m_motion_top100_segmented \
  --subject_mode lm \
  --save_mask_png \
  --save_overlay_video
```

## 輸出格式

每支影片輸出到：

```text
<output_dir>/<video_stem>/
  frames/000000.jpg ...        # bbox + mask 疊圖
  masks/000000.png ...         # (可選) binary mask
  frames.jsonl                 # 每幀 bbox / 主體命中資訊
  summary.json                 # 該影片摘要（主體詞、命中率）
```

整批摘要：

```text
<output_dir>/run_summary.json
```

## 建議參數

- `--frame_stride 2`：速度更快，對長片段較穩定。
- `--context_topk 1`：保留一個互動背景（如 floor、court）。
- `--box_threshold 0.25 --text_threshold 0.20`：預設可先用，若漏檢可降低到 `0.2/0.15`。

## 備註

- 若小 LM 載入失敗，程式會自動 fallback 到 heuristic 主詞抽取。
- 主體「必抓」是靠 `primary_words` 優先選最高分框，若該幀沒有任何主體偵測，`primary_found=false`。
