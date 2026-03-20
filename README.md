# CellSighter Baseline

ResNet-50 based cell type classification for multiplexed imaging data, adapted from [CellSighter](https://github.com/KerenLab/CellSighter).

**Paper**: Yael Amitay et al., "CellSighter: a neural network to classify cells in highly multiplexed images", *Nature Communications* 14, 4302 (2023). DOI: [10.1038/s41467-023-40066-7](https://doi.org/10.1038/s41467-023-40066-7)

**Original code**: https://github.com/KerenLab/CellSighter

## Architecture

ResNet-50 with a CIFAR-style stem (3x3 stride-1 conv, no maxpool) adapted for 32x32 cell patches from multiplexed imaging. Input channels are globally aligned marker intensities (269 channels) plus cell mask and neighbor mask (271 total).

## Installation

Requires the `deepcelltypes` main package for data loading and configuration:

```bash
pip install -e .
```

This will install `deepcelltypes` from GitHub as a dependency. Alternatively, install `deepcelltypes` locally first:

```bash
pip install -e /path/to/deepcelltypes-cell-type-assignment-pytorch
pip install -e .
```

## Usage

```bash
python -m cellsighter --model_name cs_0 --device_num cuda:0 --split_file splits/fov_split_v7.json
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model_name` | `cellsighter_0` | Name for saved model and output files |
| `--device_num` | `cuda:0` | Device to use |
| `--zarr_dir` | `$DATA_DIR/tissuenet-caitlin-labels.zarr` | Path to TissueNet zarr archive |
| `--split_file` | None | Pre-computed FOV split JSON |
| `--epochs` | 50 | Number of training epochs |
| `--learning_rate` | 0.001 | Learning rate (constant, matching original) |
| `--batch_size` | 256 | Batch size |
| `--model_size` | `resnet50` | ResNet variant (`resnet50` or `resnet18`) |
| `--pretrained` | False | Use ImageNet pretrained weights |
| `--val_every_n_epochs` | 10 | Validation frequency |
| `--no_amp` | False | Disable automatic mixed precision |
| `--no_compile` | False | Disable torch.compile |
| `--min_channels` | 3 | Min non-DAPI channels per dataset |
| `--enable_wandb` | False | Enable Weights & Biases logging |

## Known Adaptations from Original

- **CIFAR-style stem** (3x3 stride-1, no maxpool) instead of ImageNet stem -- necessary for 32x32 patches vs original 60x60.
- **Global 269-channel sparse alignment** via `marker2idx` (original uses dense per-dataset channels).
- **Raw intensity masked by cell mask** (original uses unmasked crops).
- **Cell mask + neighbor mask channels** (original uses `all_cells_mask` + `cell_mask`).
