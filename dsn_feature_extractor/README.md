# dsn_feature_extractor

本目錄提供一個以 DSN (Domain Separation Network) 為核心的特徵分解器：

- `shared` 分支：提取跨 domain 穩定資訊（此專案設定為偏向 motion）
- `private` 分支：提取 domain-specific 資訊（此專案設定為偏向 texture / appearance）

目前版本先完成 **classifier loss 以外** 的訓練流程，並保留 classifier 介面可隨時打開。

---

## 檔案說明

- `model.py`
  - `DSNFeatureExtractor`：shared/private encoder + decoder
  - 輸入支援 `[B,T,C]`、`[B,T,N,C]`、`[B,T,H,W,C]`
  - 若輸入有 token/spatial 維度，會先做平均池化到 `[B,T,C]`

- `losses.py`
  - `dsn_pair_loss`
  - 組成：
    - reconstruction loss
    - shared consistency loss（pair 的 shared 要相近）
    - shared/private orthogonality penalty
    - private margin loss（pair 的 private 要有差異）
    - optional classifier loss（預設關閉）

- `train.py`
  - 用 paired feature 訓練 DSN
  - 支援兩種輸入：
    1. `--pair_manifest`：CSV 指定 `path_a,path_b[,label]`
    2. `--pair_dirs DIR_A DIR_B`：兩個鏡像目錄，自動配對相對路徑

- `encode.py`
  - 對單一 feature 檔輸出 `shared/private` embedding 與 reconstruction

- `__init__.py`
  - 匯出 `DSNFeatureExtractor`, `DSNPairOutput`, `dsn_pair_loss`

- `envs/environment.train.yaml`
  - `train.py` 使用環境

- `envs/environment.encode.yaml`
  - `encode.py` 使用環境

---

## 環境設定

### 腳本與環境對應

- `train.py` -> `envs/environment.train.yaml`
- `encode.py` -> `envs/environment.encode.yaml`

### 建立環境

```bash
conda env create -f envs/environment.train.yaml
conda env create -f envs/environment.encode.yaml
```

### 啟用環境

```bash
conda activate dsn-train
conda activate dsn-encode
```

---

## 資料格式

### paired manifest（`--pair_manifest`）

CSV 至少包含欄位：

- `path_a`
- `path_b`

可選欄位：

- `label`（僅在啟用 classifier loss 時使用）

### feature tensor 格式

每個 feature 檔（`.pt/.pth/.npy`）可為：

- `[T, C]`
- `[T, N, C]`
- `[T, H, W, C]`

`[T, C]` 會自動轉成 `[T, 1, C]`。

---

## 使用方法

### 1) 訓練 DSN（不含 classifier loss）

```bash
conda activate dsn-train
python train.py \
  --pair_dirs ../dataset/vfm_features_styleA ../dataset/vfm_features_styleB \
  --save_dir ./checkpoints \
  --input_dim 768 \
  --shared_dim 128 \
  --private_dim 128
```

### 2) 訓練 DSN（保留介面，啟用 classifier loss）

```bash
conda activate dsn-train
python train.py \
  --pair_manifest ./pairs_with_labels.csv \
  --save_dir ./checkpoints_clf \
  --input_dim 768 \
  --use_classifier_loss \
  --classifier_weight 0.1 \
  --num_classes 2
```

### 3) 對單檔做 shared/private 抽取

```bash
conda activate dsn-encode
python encode.py \
  --checkpoint ./checkpoints/dsn_epoch20.pt \
  --input ../dataset/vfm_features_styleA/0001.pt \
  --output ../dataset/dsn_features/0001.pt \
  --input_dim 768 \
  --shared_dim 128 \
  --private_dim 128
```

---

## CLI 參數說明（argparse）

### `train.py`

- `--pair_manifest` (`Path`)
  - pair 清單 CSV，欄位需有 `path_a,path_b`，可選 `label`
- `--pair_dirs` (`Path Path`)
  - 兩個鏡像資料夾，依相對路徑自動配對
- `--save_dir` (`Path`, 預設 `./checkpoints`)
  - checkpoint 輸出資料夾
- `--input_dim` (`int`, 預設 `768`)
  - 輸入特徵維度
- `--hidden_dim` (`int`, 預設 `384`)
  - encoder/decoder 中間維度
- `--shared_dim` (`int`, 預設 `128`)
  - shared branch 維度（偏 motion）
- `--private_dim` (`int`, 預設 `128`)
  - private branch 維度（偏 texture）
- `--num_heads` (`int`, 預設 `8`)
  - Transformer attention heads
- `--num_layers` (`int`, 預設 `2`)
  - Transformer layer 數
- `--dropout` (`float`, 預設 `0.1`)
  - 模型 dropout
- `--epochs` (`int`, 預設 `20`)
  - 訓練回合
- `--batch_size` (`int`, 預設 `4`)
  - batch size
- `--lr` (`float`, 預設 `2e-4`)
  - learning rate
- `--weight_decay` (`float`, 預設 `1e-4`)
  - AdamW weight decay
- `--num_workers` (`int`, 預設 `4`)
  - DataLoader workers
- `--recon_weight` (`float`, 預設 `1.0`)
  - reconstruction loss 權重
- `--shared_weight` (`float`, 預設 `1.0`)
  - shared consistency loss 權重
- `--orth_weight` (`float`, 預設 `0.1`)
  - orthogonality penalty 權重
- `--private_margin_weight` (`float`, 預設 `0.1`)
  - private margin loss 權重
- `--private_margin` (`float`, 預設 `1.0`)
  - private 距離下界（hinge margin）
- `--use_classifier_loss` (`flag`, 預設關閉)
  - 是否啟用 classifier loss
- `--classifier_weight` (`float`, 預設 `0.0`)
  - classifier loss 權重
- `--num_classes` (`int`, 預設 `2`)
  - classifier 類別數

### `encode.py`

- `--checkpoint` (`Path`, 必填)
  - DSN 模型權重檔
- `--input` (`Path`, 必填)
  - 單一輸入特徵檔
- `--output` (`Path`, 必填)
  - 輸出路徑（儲存 shared/private/reconstruction）
- `--input_dim` (`int`, 預設 `768`)
  - 輸入特徵維度
- `--hidden_dim` (`int`, 預設 `384`)
  - 中間維度
- `--shared_dim` (`int`, 預設 `128`)
  - shared 維度
- `--private_dim` (`int`, 預設 `128`)
  - private 維度
- `--num_heads` (`int`, 預設 `8`)
  - Transformer attention heads
- `--num_layers` (`int`, 預設 `2`)
  - Transformer layer 數
- `--dropout` (`float`, 預設 `0.1`)
  - dropout

---

## Physion supervision 註記

在目前 workspace 的 `dataset/physion_data/Physion/Physion/labels.csv` 可看到：

- 欄位為 `ground truth outcome`（True/False）
- 這可以作為 binary classifier loss 的監督來源

因此 classifier loss 並非不可用，而是可視實驗分兩階段：

1. 先用重建 + 分解 loss 做 representation disentanglement（本版預設）
2. 再加上 outcome classifier 做任務導向微調
