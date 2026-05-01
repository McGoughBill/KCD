# KCD — Automated Small Kidney Cancer Detection in Non-Contrast CT

Code accompanying the paper *Automated Small Kidney Cancer Detection in Non-Contrast Computed Tomography*.

> **Pre-print:** [arXiv:2312.05258](https://arxiv.org/abs/2312.05258)

This repository contains the training, inference, and evaluation pipelines used to produce the results in the paper. The original investigation included a number of ablation studies and cohort-specific experiments; for the public release, we have kept the canonical pipeline scripts and removed cohort/site-specific variants. Background notes from the development of this work live in [`docs/literature_review.md`](docs/literature_review.md).

## Repository layout

```
KCD/
├── Detection/             # 2D / 3D / MTL slice classifiers — main detection pipeline
│   ├── Dataloaders/       #   per-modality dataloaders
│   ├── ModelGenerator/    #   ResNeXt 2D/3D MTL, EfficientNet V2 L 3D, Merlin
│   ├── Preprocessing/     #   axial slice + object preprocessing
│   ├── Training/          #   2d/, 3d_vanilla/, MTL/, finetuning_dira/, shape/
│   ├── Inference/         #   slice / shape / merlin inference entry points
│   └── Evaluation/        #   metrics + ROC utilities
├── Segmentation/          # ovseg-based segmentation backbone (pretrain → finetune → infer)
│   ├── ovseg/             #   the ovseg library (vendored)
│   ├── Inference/         #   End2End.py, End2End_ROC.py, EnsembleSeg, Postprocess_Testing
│   └── HPC_Scripts/       #   canonical SLURM examples for pretrain/prep/finetune/infer
├── coregistration/        # NCCT ↔ CECT coregistration (TCIA + ARTIST)
├── ne2ceCT/               # NCCT → CECT cross-modal synthesis / distillation
├── sLDCT_synthesis/       # synthetic low-dose CT generation
├── da_ft/                 # domain-adaptation fine-tuning experiments
├── screening_sim/         # clinical screening cost-effectiveness simulation
├── tcia_download/         # TCIA download + crop helpers
├── InferenceEvaluation/   # cross-pipeline result aggregation, ROC plotting, EigenCAM
├── utils/paper_plots/     # figure-generation scripts (testsetv3 = paper figures)
└── config.py              # central path resolver — see "Configuration" below
```

Top-level helper scripts (`mask_*.py`, `convert_*.py`, `prepare_AA.py`, `compile_ds.py`, `unpack_ykst.py`, …) are one-shot data-prep utilities invoked by the pipelines above.

## Setup

```bash
git clone https://github.com/McGoughBill/KCD.git
cd KCD
conda create -n kcd python=3.10
conda activate kcd
pip install -e .
```

CUDA-enabled PyTorch is required for training. The `setup_KCD_env.sh` script is a Baskerville HPC reference; users on other clusters should adapt the SLURM directives at the top of that file.

## Configuration

All paths are resolved through [`KCD/config.py`](KCD/config.py). Two equivalent ways to configure:

**Option A — YAML (recommended for local development):**
```bash
cp configs/paths.example.yaml configs/paths.yaml
# Edit configs/paths.yaml to point to your local data, weights, predictions
```

**Option B — environment variables (recommended for HPC / containers):**
```bash
export KCD_DATA_ROOT=/scratch/$USER/kcd_data
export KCD_OV_DATA_BASE=$KCD_DATA_ROOT/seg_data
export KCD_RAW_DATA=$KCD_DATA_ROOT/seg_data/raw_data
export KCD_PREDICTIONS=$KCD_DATA_ROOT/predictions_nii
export KCD_MODEL_WEIGHTS=$KCD_DATA_ROOT/model_weights
```

Environment variables override the YAML values. Missing keys raise `FileNotFoundError` — there is no silent fallback.

> **Note on legacy scripts.** The main training/inference entry points listed in *Reproduce paper results* below call `get_paths()` automatically. Many secondary scripts (older preprocessing utilities, HPC orchestration scripts, profiling code) still contain the literal string `/path/to/kcd_data` as a placeholder. To run those as-is, either edit the placeholders inline or run a one-shot replace:
>
> ```bash
> grep -rl '/path/to/kcd_data' . | xargs sed -i '' "s|/path/to/kcd_data|$KCD_DATA_ROOT|g"
> ```

## Data

Public datasets used for pre-training and benchmarking:

| Dataset | Purpose | Source |
|---|---|---|
| KiTS23 | Kidney + tumour segmentation pre-training | https://kits-challenge.org/kits23/ |
| KiTS23 NCCT | Synthetic NCCT counterparts of KiTS23 CECTs | (see paper Section 2.1) |
| ORCA | Paired NCCT/CECT (40 patients) | https://orcascore.grand-challenge.org/Data/ |
| COLTEA | Paired NCCT/CECT (100 patients) | https://github.com/ristea/cycle-transformer |
| AbdomenAtlas | Multi-organ pre-training | (see paper Section 2.2) |
| BTCV | Multi-organ baseline | https://www.synapse.org/#!Synapse:syn3193805 |

The clinical evaluation cohorts (CamUSS, ARTIST, YKST, OxCAM) are not publicly redistributable; the loaders, preprocessing, and inference scripts that operate on them are released so the same protocol can be reproduced on equivalent local data.

Expected layout under `${KCD_DATA_ROOT}`:
```
${KCD_DATA_ROOT}/
├── seg_data/raw_data/<dataset>/{images,labels}/*.nii.gz
├── predictions_nii/
└── model_weights/
```

## Reproduce paper results

The full pipeline is `data prep → segmentation pre-training → detection training → end-to-end inference → ROC analysis`. A minimal reproduction uses the canonical entry points:

**1. Data prep (NCCT slice extraction):**
```bash
python KCD/Detection/Preprocessing/AxialSlices/create3D_pretrain.py
```

**2. Segmentation pre-training (CECT, cancer-only labels):**
```bash
sbatch KCD/Segmentation/HPC_Scripts/pretrain_seg/pretrain_cect_canceronly_ch.py
# or run as a plain `python` script if not on SLURM
```

**3. Detection — train each model variant:**
```bash
# 2D slice-tile classifier
python KCD/Detection/Training/2d/train_slice_models.py 0   # fold 0

# 3D vanilla patch classifier with KiTS23 pre-training
python KCD/Detection/Training/3d_vanilla/train_slice_models_pretrain_3d.py 100 0 0.001

# 3D MTL (segmentation + classification heads) with KiTS23 pre-training
python KCD/Detection/Training/MTL/train_slice_models_pretrain_MTL.py 100 0
```

**4. End-to-end inference (segmentation + detection on a held-out cohort):**
```bash
python KCD/Segmentation/Inference/End2End.py
python KCD/Detection/Inference/slice_model_inference.py
```

**5. ROC analysis and paper figures:**
```bash
python KCD/Segmentation/Inference/End2End_ROC.py
python KCD/utils/paper_plots/testsetv3/fd_test_ROC.py
```

Trained model weights are not bundled in the repo. Contact the authors or download from the release artefact (TBD link) once available.

## Citation

```bibtex
@article{mcgough2023kcd,
    title   = {Automated Small Kidney Cancer Detection in Non-Contrast Computed Tomography},
    author  = {McGough, William and others},
    journal = {arXiv preprint arXiv:2312.05258},
    year    = {2023},
    url     = {https://arxiv.org/abs/2312.05258}
}
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Acknowledgements

The segmentation backbone is built on the [`ovseg`](https://github.com/ThomasBudd/ovseg) library (vendored under `KCD/Segmentation/ovseg/`).
