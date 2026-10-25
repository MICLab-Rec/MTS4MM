# MTS4MM

Code repository for **Three Minds, One Student: Online Multi-Teacher Knowledge Distillation for Multimodal Recommenders**.

**Hangtong Xu, Yuanbo Xu, and En Wang** · **IJCAI 2026**, Main Track, pp. 3133–3141.

[Paper](https://ijcai.org/proceedings/2026/348) · [DOI](https://doi.org/10.24963/ijcai.2026/348) · [License](LICENSE)

The method is named **MTS²-4MM** in the paper; the repository, Python class, and command-line model identifier use **MTS4MM**.

## Overview

MTS²-4MM learns complementary preference signals from collaborative IDs, images, and text through online multi-teacher distillation. Modality-specific teachers guide a unified student, which produces recommendations at inference time. The training pipeline builds on [MMRec](https://github.com/enoche/MMRec).

This repository contains the MTS4MM model, training and checkpoint-evaluation utilities, dataset configurations, and preprocessing notebooks. Dataset configurations are provided for **Baby, Sports, Clothing, Electronics, and MicroLens**.

## Installation

The following environment follows [MMRec's legacy dependency versions](https://github.com/enoche/MMRec/blob/master/requirements.txt) and includes the additional packages imported by this repository. It is a starting setup rather than a separately validated environment lockfile.

```bash
conda create -n mts4mm python=3.7.11 -y
conda activate mts4mm

# PyTorch / torchvision for CUDA 11.3
python -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install numpy==1.21.5 pandas==1.3.5 scipy==1.7.3 PyYAML==6.0 matplotlib lmdb Pillow
```

For a CPU environment, replace the PyTorch installation command with:

```bash
python -m pip install torch==1.11.0+cpu torchvision==0.12.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
```

Other CUDA builds are listed in the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v1110). Training consumes precomputed features; feature extraction through the notebooks may require additional dependencies.

## Data preparation

Place each processed dataset under `data/<dataset>/`:

```text
data/
└── baby/
    ├── baby.inter
    ├── image_feat.npy
    └── text_feat.npy
```

The same layout applies to the other datasets:

| Dataset | Command-line name | Interaction file |
| --- | --- | --- |
| Amazon Baby | `baby` | `baby.inter` |
| Amazon Sports | `sports` | `sports.inter` |
| Amazon Clothing | `clothing` | `clothing.inter` |
| Amazon Electronics | `elec` | `elec.inter` |
| MicroLens | `microlens` | `microlens.inter` |

Processed Baby, Sports, and Electronics downloads are linked from [MMRec's data page](https://github.com/enoche/MMRec/tree/master/data). See the [MicroLens project](https://github.com/westlake-repl/MicroLens) for its original data and [data/README.md](data/README.md) for an additional processed-data link. Clothing must be prepared in the same format. Match the paper's processed subset and split when comparing results; downloading the original dataset alone does not establish an identical evaluation protocol.

### Required format

- **Interactions:** a tab-separated file with a header. The supplied configurations require `userID`, `itemID`, and `x_label`. Split labels are `0` for training, `1` for validation, and `2` for testing. Additional columns such as `rating` and `timestamp` may be retained.
- **IDs:** use integer IDs reindexed from zero. Feature row `i` must correspond to interaction item ID `i`.
- **Features:** numeric NumPy arrays with shapes `(num_items, image_dim)` and `(num_items, text_dim)`. Both arrays must use the same item ordering. The Baby files in this checkout have dimensions `4096` and `384`, respectively.
- **Configuration:** filenames and field names are defined in `src/configs/dataset/<dataset>.yaml`. With the default configuration, training reads data from `../data/` relative to `src/`.

The model supports an available subset of content modalities, but both image and text features are needed for the three-view setting. ID mapping CSVs and `user_graph_dict.npy` are not required by MTS4MM.

For raw-data preparation, the notebooks in `preprocessing/` cover filtering/reindexing, interaction splitting, feature reindexing, and feature encoding, in this order:

```text
0rating2inter.ipynb → 1splitting.ipynb → 2reindex-feat.ipynb → 3feat-encoder.ipynb
```

Adapt their dataset paths before execution, then update the corresponding dataset YAML file.

## Quick start

**Run training and evaluation from `src/`.** Configuration paths are resolved from the current working directory.

### One configuration on Baby

From the repository root, run:

```bash
cd src
python -c "from utils.quick_start import quick_start; quick_start(model='MTS4MM', dataset='baby', config_dict={'gpu_id': 0, 'reg_weight': 1e-7, 'kd_tau': 1.0, 'n_ui_layers': 2, 'cl_loss': 0.005, 'kd_rank_weight': 0.1}, save_model=True)"
```

This selects one configuration and uses the default seed `999`. These values demonstrate the interface; they are not claimed to be the optimal settings for every dataset. The run trains, selects the best validation checkpoint, and evaluates the selected student on the test split. The default limit is 1,000 epochs, with early stopping based on validation `Recall@20`.

For a brief CPU check, add `'use_gpu': False, 'epochs': 2` to `config_dict` above. A two-epoch check covers loading, training, and evaluation; ranking distillation begins after the 10-epoch warm-up.

### Default hyperparameter search

From `src/`:

```bash
python main.py --model MTS4MM --dataset baby --gpu_id 0
```

The shipped YAML configuration evaluates **54 combinations** for the default seed. This command runs the full search. Replace `baby` with `sports`, `clothing`, `elec`, or `microlens` after preparing that dataset. The shorthand `-m MTS4MM -d baby` is also supported.

The CLI accepts `--model`, `--dataset`, and `--gpu_id`. Set other options in YAML or through `quick_start(..., config_dict=...)`; arbitrary extra CLI options are not forwarded to the configuration. In particular, `--use_kd` is explicitly rejected for MTS4MM.

## Configuration and implementation notes

Configuration is loaded in the order `overall.yaml` → dataset YAML → model YAML → `config_dict`. Hyperparameter lists define a Cartesian-product search, with the seed included. MTS4MM then enforces its fixed teacher settings in code.

| Setting | Shipped value / search space |
| --- | --- |
| Embedding size | `64` |
| Modality graph neighbors (`knn_k`) | `10` |
| Modality graph layers (`n_layers`) | `1` (required by this implementation) |
| Collaborative graph layers (`n_ui_layers`) | `[2, 1, 3]` |
| Learning rate | `0.001` |
| Regularization (`reg_weight`) | `[1e-7, 1e-6]` |
| Contrastive loss weight (`cl_loss`) | `[0.005, 0.001, 0.01]` |
| Distillation temperature (`kd_tau`) | `[1.0, 0.5, 2.0]` |
| Ranking distillation weight (`kd_rank_weight`) | `[0.1]` |
| Seed | `[999]` |

The current code trains view-specific experts with BPR, maintains EMA teachers with momentum `0.995`, and applies pairwise ranking distillation after 10 warm-up epochs with a 10-epoch ramp. Teacher signals are selected using positive/negative score margins. `full_sort_predict` uses the student representations.

## Evaluation and outputs

Training selects the checkpoint by validation `Recall@20` and evaluates the test split after selecting the best configuration. Default metrics are **Recall, NDCG, Precision, and MAP** at `K = 5, 10, 20, 50`.

With the default configuration, outputs are:

| Output | Location relative to the repository root |
| --- | --- |
| Training and evaluation log | `src/csmmsr/MTS4MM-<dataset>-<timestamp>.log` |
| Selected checkpoint | `src/saved/mts4mm__<dataset>.pth` |
| Configuration and validation metadata | `src/saved/mts4mm__<dataset>.pth.metadata.json` |
| Test top-50 recommendations | `src/recommend_topk/MTS4MM-<dataset>-idx0-top50-<timestamp>.csv` |
| Cached modality graphs | `data/<dataset>/mts4mm_*_cosine_weighted_v1_k*.pt` |

The checkpoint filename contains **two underscores** before the dataset name because the configured prefix already ends with `_`. Repeated runs for the same dataset overwrite that checkpoint and its metadata; copy them elsewhere to retain separate runs. Recommendation files use tab separators despite the `.csv` extension and contain internal integer IDs.

To evaluate a saved Baby checkpoint without retraining, run from `src/`:

```bash
python -c "from utils.load_and_eval import evaluate_from_checkpoint; evaluate_from_checkpoint(model='MTS4MM', dataset='baby', config_dict={'gpu_id': 0}, ckpt_path='saved/mts4mm__baby.pth')"
```

Keep the `.metadata.json` file beside its checkpoint so evaluation can restore the selected hyperparameters. The same processed interaction and feature files are still required.

## Repository layout

```text
MTS4MM/
├── data/                     # Processed interactions and feature arrays
├── preprocessing/            # Raw-data preprocessing notebooks
├── evaluation/               # Inherited MMRec evaluation notes
├── src/
│   ├── main.py               # Training / grid-search entry point
│   ├── models/mts4mm.py      # Student, experts, and EMA teachers
│   ├── configs/              # Overall, dataset, and model settings
│   ├── common/               # Recommender interfaces and trainer
│   └── utils/                # Data loading, metrics, and checkpoint evaluation
├── LICENSE
└── README.md
```

The root README documents this release. Inherited notes under `evaluation/` and `preprocessing/` may refer to baseline models that are not included here.

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{ijcai2026p348,
  title     = {Three Minds, One Student: Online Multi-Teacher Knowledge Distillation for Multimodal Recommenders},
  author    = {Xu, Hangtong and Xu, Yuanbo and Wang, En},
  booktitle = {Proceedings of the Thirty-Fifth International Joint Conference on Artificial Intelligence, IJCAI-26},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  pages     = {3133--3141},
  year      = {2026},
  doi       = {10.24963/ijcai.2026/348},
  url       = {https://doi.org/10.24963/ijcai.2026/348}
}
```

## Acknowledgements and license

We thank [MMRec](https://github.com/enoche/MMRec) for the recommendation framework and the dataset providers for their resources. Please also acknowledge the upstream framework and datasets when using them.

The repository includes an [MIT license](LICENSE). Reused third-party components and datasets remain subject to their respective licenses and terms.
