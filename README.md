# CvT for Plant Disease Classification

A complete, dataset-agnostic PyTorch pipeline for classifying plant diseases from leaf images with
**CvT (Convolutional vision Transformer)**. CvT combines convolutional token embeddings and convolutional
projections inside a hierarchical, three-stage transformer, and has no positional embeddings. The model
and ImageNet-pretrained weights come from [Hugging Face Transformers](https://huggingface.co/docs/transformers/model_doc/cvt)
(`microsoft/cvt-*`), since CvT is not part of timm. The pipeline covers training, evaluation, prediction and
ONNX export, and runs on CPU, one GPU, or many GPUs.

## Features

- **Any image-folder dataset**: PlantVillage, PlantDoc, PlantWild or your own field images. The number of classes, class names, input size and normalisation are detected automatically.
- **CvT-13 / CvT-21 / CvT-W24** and their 384 px and ImageNet-22k variants, or any other CvT checkpoint on the Hub or on disk.
- **Leakage-safe splitting**: stratified by default, or grouped by leaf/plant ID with `--group_regex`.
- **Scalable training**: multi-GPU (DDP via `torchrun`), mixed precision (fp16/bf16), gradient accumulation, EMA, resume, early stopping.
- **Imbalance and robustness tools**: class-weighted loss, label smoothing, Mixup/CutMix, optional stronger augmentation.
- **Research-ready outputs**: per-class precision/recall/F1, macro-F1, confusion matrices, training curves, saved splits and arguments.
- **Self-contained checkpoints**: weights, model config, class names and preprocessing are stored together, so evaluation, prediction and export need no Hub access.
- **Same CLI and splits as the MobileViT version of this repository**, so the two models can be compared fairly with the same `--seed`.

## Project structure

```
cvt-plant-disease/
├── README.md
├── requirements.txt
├── train.py                  # entry point: train
├── evaluate.py               # entry point: evaluate a checkpoint on a folder
├── predict.py                # entry point: predict on an image / folder
├── export_onnx.py            # entry point: export to ONNX
├── plant_cls/
│   ├── config.py             # command-line arguments
│   ├── data.py               # dataset discovery, splits, transforms
│   ├── model.py              # CvT (Hugging Face) wrapper, EMA, checkpoint loading
│   ├── engine.py             # training loop (AMP, DDP, accumulation, resume)
│   ├── metrics.py            # evaluation, reports, plots
│   ├── inference.py          # evaluate / predict / export logic
│   └── utils.py              # seeding, device, distributed helpers
├── scripts/
│   ├── train_single_gpu.sh
│   ├── train_multi_gpu.sh
│   └── make_dummy_dataset.py # tiny synthetic dataset for smoke tests
└── tests/
    ├── test_splits.py
    └── test_model.py
```

## Installation

Python 3.9+ is required.

```bash
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt
```

For GPU training, install the PyTorch build matching your CUDA version first
(see https://pytorch.org/get-started/locally/), then `pip install -r requirements.txt`.
Pretrained weights are downloaded from the Hugging Face Hub on first use.

## Dataset format

Point `--data_dir` at either layout (detected automatically):

**A. Already split**
```
data/
├── train/<class_name>/*.jpg
├── val/<class_name>/*.jpg        # "valid" or "validation" also work
└── test/<class_name>/*.jpg       # optional
```

**B. Single folder**: split automatically with `--val_split` (default 0.15) and `--test_split` (default 0.15)
```
data/
├── Tomato___Early_blight/*.jpg
├── Tomato___healthy/*.jpg
└── ...
```

### Avoiding data leakage (important)

Datasets such as PlantVillage contain several photos of the same leaf. A random split can put
near-duplicates in both train and test and inflate accuracy. If your filenames encode the leaf or
plant ID, use `--group_regex` with **one capture group** that extracts it from the file name
(without extension). Images with the same ID are then always kept in the same split:

```bash
# files like  leaf0042_3.jpg  ->  group "leaf0042"
python train.py --data_dir data/PlantVillage --group_regex '^(.+)_\d+$'
```

## Quick start

```bash
# 1. Train (downloads ImageNet-pretrained CvT-13 weights on first run)
python train.py --data_dir /path/to/data --model cvt-13 --epochs 30

# 2. Evaluate on a held-out folder
python evaluate.py --checkpoint runs/cvt-13/best.pt --data_dir /path/to/test

# 3. Predict
python predict.py --checkpoint runs/cvt-13/best.pt --input leaf.jpg --topk 3
python predict.py --checkpoint runs/cvt-13/best.pt --input folder/ --output_csv preds.csv

# 4. Export for deployment
python export_onnx.py --checkpoint runs/cvt-13/best.pt
```

When the pretrained model loads you will see a Transformers warning that the classification head was newly
initialised. That is expected: the 1000-class ImageNet head is replaced by one for your classes.

### Multi-GPU

```bash
torchrun --nproc_per_node=4 train.py --data_dir /path/to/data --scale_lr
# or: NGPUS=4 bash scripts/train_multi_gpu.sh /path/to/data
```

`--batch_size` is **per GPU**. Use `--scale_lr` to scale the learning rate with the global batch size,
and `--accum_steps N` to simulate larger batches on small GPUs.

### Smoke test without real data

```bash
python scripts/make_dummy_dataset.py --out data/dummy --classes 4 --per_class 60
python train.py --data_dir data/dummy --model cvt-13 --no-pretrained --epochs 2 --batch_size 8 --workers 0
pytest -q          # optional: unit tests for splitting and the model (needs pytest, no downloads)
```

## Choosing a model

| `--model`          | Hub checkpoint                     | Pretraining       | Input | Params (approx.) |
|--------------------|------------------------------------|-------------------|-------|------------------|
| `cvt-13` (default) | `microsoft/cvt-13`                 | ImageNet-1k       | 224   | 20 M             |
| `cvt-13-384`       | `microsoft/cvt-13-384`             | ImageNet-1k       | 384   | 20 M             |
| `cvt-13-384-22k`   | `microsoft/cvt-13-384-22k`         | ImageNet-22k to 1k | 384  | 20 M             |
| `cvt-21`           | `microsoft/cvt-21`                 | ImageNet-1k       | 224   | 32 M             |
| `cvt-21-384`       | `microsoft/cvt-21-384`             | ImageNet-1k       | 384   | 32 M             |
| `cvt-21-384-22k`   | `microsoft/cvt-21-384-22k`         | ImageNet-22k to 1k | 384  | 32 M             |
| `cvt-w24`          | `microsoft/cvt-w24-384-22k`        | ImageNet-22k to 1k | 384  | 277 M            |

You can also pass any Hub id (`--model some-org/some-cvt`) or a local folder saved with `save_pretrained`.
CvT is heavier than MobileViT; for 384 px models or CvT-W24 lower `--batch_size` and use `--accum_steps`.

`--img_size` overrides the input size. CvT has no positional embeddings, so other resolutions generally work,
although accuracy is best near the size the checkpoint was trained at.

`--no-pretrained` trains from scratch using built-in architecture presets for `cvt-13`, `cvt-21` and `cvt-w24`
(no download needed).

## Useful options

| Option | Purpose |
|--------|---------|
| `--lr 1e-4` (default) | AdamW learning rate for fine-tuning; raise it for from-scratch training |
| `--class_weights` | inverse-frequency loss weights for imbalanced classes |
| `--aug strong` | adds TrivialAugment on top of the default augmentation |
| `--mixup 0.2 --cutmix 1.0` | Mixup / CutMix (off by default; CutMix can hide small lesions) |
| `--drop_path 0.2` | stochastic depth for the last stage (default: the checkpoint's 0.1) |
| `--ema_decay 0.999` | evaluate and save an EMA of the weights |
| `--amp_dtype bf16` | bfloat16 instead of fp16 (Ampere GPUs and newer); more stable for transformers |
| `--opt sgd --lr 0.01` | any timm optimizer |
| `--patience 10` | early stopping on validation macro-F1 (0 disables) |
| `--resume runs/<exp>/last.pt` | continue an interrupted run |
| `--no-pretrained` | train from scratch (or if weights cannot be downloaded) |

Run `python train.py --help` for the full list.

## Outputs

Every run writes to `runs/<exp_name>/` (default `exp_name` is the model name):

| File | Content |
|------|---------|
| `best.pt` | best weights by validation macro-F1, plus model config, class names and preprocessing (used by evaluate/predict/export) |
| `last.pt` | full training state for `--resume` |
| `val_report.txt/.json`, `test_report.txt/.json` | per-class precision, recall, F1, support |
| `*_confusion_matrix.csv/.png` | confusion matrices |
| `training_curves.png`, `history.json` | loss, accuracy, macro-F1 per epoch |
| `results.json` | final metrics summary |
| `splits.json`, `class_names.json`, `args.json` | everything needed to reproduce the run |

## Tips for plant-disease research

- **Report macro-F1 and per-class metrics**, not only accuracy, since classes are usually imbalanced.
- **Lab-to-field gap**: models trained on clean lab images (PlantVillage) often drop sharply on field photos. Evaluate on a separate field dataset such as PlantDoc, use `--aug strong`, and consider mixing in field images.
- **Compare models fairly**: keep the dataset, `--seed` and split settings fixed and change only `--model`; use `--exp_name` to keep runs separate. The MobileViT repository uses the same CLI and split logic.
- **Reproducibility**: splits depend on `--seed`; `splits.json` stores the exact files used.

## Troubleshooting

- **CUDA out of memory**: lower `--batch_size` and raise `--accum_steps`, choose `cvt-13`, or use 224 px checkpoints.
- **Pretrained weights cannot be downloaded** (offline machine): use `--no-pretrained`, or download a checkpoint on another machine, save it with `save_pretrained`, copy the folder over and pass it as `--model /path/to/folder`.
- **Loss becomes NaN with fp16**: use `--amp_dtype bf16`, or `--no-amp`, or lower `--lr`.
- **`Validation set is empty`**: provide a `val/` folder or increase `--val_split`.
- **Odd batch size error with Mixup/CutMix**: use an even `--batch_size`.
- **ONNX export fails**: update `torch`, or try another `--opset`. CvT uses `einsum`/reshape operations that some exporter versions handle differently.
- **Slow data loading**: raise `--workers`, and keep the dataset on fast local storage.

## References

- H. Wu et al., *CvT: Introducing Convolutions to Vision Transformers*, ICCV 2021 (arXiv:2103.15808).
- Hugging Face Transformers, [CvT model documentation](https://huggingface.co/docs/transformers/model_doc/cvt).
