# FLASH

Official research code accompanying the Expert Systems with Applications article **“Physically consistent battery charging trajectory generation under mixed protocols using flow-matched diffusion transformers.”**

FLASH is a simulation-free conditional generative model for lithium-ion battery charging trajectories. Given a battery state of health (SOH) and a charging protocol, it learns an ODE velocity field that transports Gaussian noise to a three-channel trajectory containing current (`I`), voltage (`V`), and charged capacity (`Qc`).

## Method at a glance

- **Conditional flow matching:** learns deterministic noise-to-data transport without running an electrochemical simulator during generation.
- **Full-resolution input projection:** maps all 2,048 aligned time points directly into the latent space, avoiding input downsampling.
- **SOH- and protocol-conditioned generation:** combines flow time, scalar SOH, and a 100-dimensional SOC-aligned protocol vector through AdaLN-Zero modulation.
- **Factorized dual-stream attention:** uses local intra-patch attention for abrupt fluctuations and axial inter-patch attention for long-range charging trends.
- **Multivariate output:** jointly generates `I`, `V`, and `Qc` trajectories.

The abstract reports validation on 70 batteries covering 29 unseen mixed charging protocols, with average RMSEs of `0.184 C`, `0.010 V`, and `0.002 Ah` for current, voltage, and capacity, respectively. Elsewhere, the article describes a cleaned data pool of 388 batteries across 80 protocols and supplementary 64/16 protocol-split robustness experiments. These scopes should not be conflated. They are **paper-reported results**, not metrics automatically reproduced by the current public evaluation command; see [Reproducibility status](#reproducibility-status).

## Repository layout

```text
.
├── configs/flash_attention_dit.json  # public model/training configuration
├── docs/                             # dataset, checkpoint, and run notes
├── flash_battery/
│   ├── dataloader.py                 # pickle loading and normalization
│   ├── model.py                      # AttentionDiT and conditioning modules
│   ├── train_loop.py                 # conditional flow-matching objective
│   ├── eval_loop.py                  # ODE sampling
│   ├── train.py                      # training/evaluation entry point
│   └── utils.py                      # metrics and visualizations
├── scripts/train.sh
├── scripts/eval.sh
├── requirements.txt
└── LICENSE
```

## Installation

Python 3.10 or newer is recommended. The experiments described in the supplementary material used PyTorch 2.6.

```bash
python -m venv .venv
source .venv/bin/activate

# Install a PyTorch build appropriate for your CUDA/CPU environment first.
# See https://pytorch.org/get-started/locally/

pip install -r requirements.txt
```

Run commands from the repository root so that the `flash_battery` package is importable.

## Data preparation

Raw datasets, preprocessing scripts, processed pickle files, and pretrained checkpoints are not included in this repository. The loader expects trajectories that have already been resampled and padded to a common length, normally 2,048 points.

Each pickle file must contain a list of cell records:

```python
[
    {
        "charge_policy": "5.2(20%)-6(40%)-4.8(60%)-3.759(80%)-1(100%)",
        "label": "Healthy",  # optional; only relevant to CFG training
        "cycles": [
            {
                "I":  np.ndarray,  # shape: (2048,)
                "V":  np.ndarray,  # shape: (2048,)
                "Qc": np.ndarray,  # shape: (2048,)
                "SOH": float,
            }
        ],
    }
]
```

The preprocessing described in the paper uses a global fixed time step derived from the longest charging segment, cubic-spline resampling, zero padding for current, and last-value extension for voltage and capacity. Users must currently implement that preprocessing and the protocol-level train/validation split outside this repository.

Charging policies must end at 100% SOC and use monotonically increasing boundaries. For example, the string above is converted into a 100-dimensional vector whose entries describe the commanded C-rate in each 1% SOC interval. A conventional constant-current policy can be represented as `0.5(100%)`.

### Configure paths

Dataset paths are resolved **relative to the directory containing the JSON config**, not relative to the repository root. If the data are stored in `<repo>/data/`, use:

```json
{
  "dataset_params": {
    "battery_files": ["../data/train.pkl"],
    "valid_files": ["../data/valid.pkl"]
  }
}
```

The paths currently shipped in `configs/flash_attention_dit.json` (`./data/...`) therefore refer to `<repo>/configs/data/` unless changed.

More details are available in [`docs/dataset.md`](docs/dataset.md).

## Training

One-epoch sanity run:

```bash
python -m flash_battery.train \
  --config_file ./configs/flash_attention_dit.json \
  --output_dir ./outputs \
  --test_run
```

`--test_run` stops after the first training epoch and the first evaluation batch; it does not shorten that training epoch.

Single GPU:

```bash
python -m flash_battery.train \
  --config_file ./configs/flash_attention_dit.json \
  --output_dir ./outputs
```

Distributed training:

```bash
torchrun --nproc_per_node=6 -m flash_battery.train \
  --config_file ./configs/flash_attention_dit.json \
  --output_dir ./outputs
```

The supplementary material states that the reported experiment used six RTX 2080 Ti GPUs, 200 epochs, and a per-GPU batch size of 32. The public JSON currently contains 700 epochs and batch size 128, so it is a public run configuration rather than an exact record of the paper run. To request the manuscript values without editing the JSON:

```bash
torchrun --nproc_per_node=6 -m flash_battery.train \
  --config_file ./configs/flash_attention_dit.json \
  --output_dir ./outputs \
  --epochs 200 \
  --batch_size 32
```

Checkpoints and matching normalization statistics are written under:

```text
outputs/<timestamp>_<experiment>_eval_False/epoch<N>/
├── checkpoint.pth
├── normalization.pkl
└── generated_data.pkl
```

Periodic sampling during training uses `battery_files`; it is a training-set diagnostic, not held-out validation. Use `--eval_only` to load `valid_files`.

## Evaluation

```bash
python -m flash_battery.train \
  --config_file ./configs/flash_attention_dit.json \
  --resume ./outputs/<training-run>/epoch<N>/checkpoint.pth \
  --normalization_file ./outputs/<training-run>/epoch<N>/normalization.pkl \
  --output_dir ./outputs \
  --eval_only
```

Equivalent helper:

```bash
bash scripts/eval.sh \
  ./configs/flash_attention_dit.json \
  ./outputs/<training-run>/epoch<N>/checkpoint.pth \
  ./outputs/<training-run>/epoch<N>/normalization.pkl
```

Evaluation creates a new `..._eval_True/epoch1/` directory containing generated samples and analysis figures. By default, generated trajectories are trimmed to the real trajectory length before pointwise metrics are calculated. Pass `--keep_padded_length` to retain all 2,048 points.

## Reproducibility status

This release exposes the core model, flow-matching training objective, ODE sampler, normalization, and plotting code. It does **not yet constitute an end-to-end reproduction package** for the manuscript.

The following items are still required for exact reproduction:

1. scripts that download/parse the cited raw datasets and reproduce the paper's cleaning, global time alignment, padding, and protocol-level splits;
2. the exact split manifests and manuscript training configuration;
3. the paper checkpoint and its normalization statistics;
4. full-validation evaluation and numerical aggregation matching the 11,190 samples reported in the manuscript.

The current `--eval_only` path samples at most 288 examples per distributed rank, regardless of the configured `test_samples`, and produces plots rather than a machine-readable table of the reported global metrics. In addition, its default trimming uses the real trajectory's termination length. These behaviors should be revised before using the public evaluator to claim exact reproduction of the headline RMSE values.

## Paper data sources

The manuscript uses fast-charging LFP data from the Stanford/MIT closed-loop charging studies and an additional NCA dataset for cross-chemistry evaluation. Please cite the original dataset publications when using those data:

- Attia et al., *Closed-loop optimization of fast-charging protocols for batteries with machine learning*, Nature (2020), https://doi.org/10.1038/s41586-020-1994-5
- Severson et al., *Data-driven prediction of battery cycle life before capacity degradation*, Nature Energy (2019), https://doi.org/10.1038/s41560-019-0356-8
- Zhu et al., *Data-driven capacity estimation of commercial lithium-ion batteries from voltage relaxation* (dataset), https://doi.org/10.5281/zenodo.6405084

## Citation

The article is at the proof stage. Add the DOI, volume, and article number once the final journal record is available.

```bibtex
@article{li2026flash,
  title   = {Physically consistent battery charging trajectory generation under mixed protocols using flow-matched diffusion transformers},
  author  = {Li, Xiaotian and Huang, Xinghao and Tao, Shengyu and Liang, Chen and Wang, Runhua and Wang, Junpeng and Zhang, Jiale and Deng, Yuchen and Liao, Zhichao and Xia, Bizhong},
  journal = {Expert Systems with Applications},
  year    = {2026},
  note    = {In press}
}
```

## License

This code is released under the [MIT License](LICENSE). Dataset licenses and terms are governed by their original providers.
