# KCD — Kidney cancer screening in non-contrast CT via domain-adaptive deep learning

Code accompanying the paper *Kidney cancer screening in non-contrast computed tomography via domain-adaptive deep learning* (manuscript under review).

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

### Pretrained weights

Five-fold Stage 1 and Stage 2 checkpoints used to produce the paper's ROC results are archived on Zenodo:

> **DOI:** [10.5281/zenodo.21836814](https://doi.org/10.5281/zenodo.21836814)
> **Direct link:** https://zenodo.org/records/21836814
> **File:** `NCCT_LDCT_pipeline_v3.tar.gz` (3.2 GB)
> **Licence:** CC BY-NC-ND 4.0

The files are access-controlled. To obtain them, sign in to Zenodo, open the record and submit an access request stating your name, institution, institutional email and intended use. Access is granted to researchers at non-profit and academic institutions for non-commercial research.

Setup:

```bash
# 1. Once access is granted, download the archive from Zenodo, then extract into your model weights root
mkdir -p $KCD_MODEL_WEIGHTS
tar -xzvf NCCT_LDCT_pipeline_v3.tar.gz -C $KCD_MODEL_WEIGHTS

# 2. Confirm structure — should look like:
#    $KCD_MODEL_WEIGHTS/NCCT_LDCT_pipeline/coreg_v3_noised/.../fold_{0..4}
#    $KCD_MODEL_WEIGHTS/NCCT_LDCT_pipeline/masked_coreg_v3_noised/.../fold_{0..4}
```

`End2End_ROC.py` will look for the weights under `KCD_MODEL_WEIGHTS/NCCT_LDCT_pipeline/…` automatically once this variable (or the `model_weights` key in `configs/paths.yaml`) is set.

Optimizer state (`opt_parameters`) has been stripped from the release; it is only needed to resume training and is not required for inference.

## Citation

Manuscript under review. A formal citation will be added once the paper is accepted.

## License

The KCD code is released under the **PolyForm Strict License 1.0.0** — see [LICENSE](LICENSE) for the full terms, or https://polyformproject.org/licenses/strict/1.0.0.

Required Notice: Copyright (c) 2026 Crispin Lab (https://github.com/McGoughBill/KCD)

In brief:

- **Noncommercial use only** — you may use the code for noncommercial purposes, including academic and non-profit research.
- **No redistribution, no derivative works** — the licence does not permit distributing the code or making changes or new works based on it.
- **Commercial use requires a separate licence** — please contact the corresponding authors of the paper.
- **Clinical use is not authorised** — the code and weights are research artefacts, not medical devices. Do not use them for diagnosis or treatment.

Third-party components included in this repository (for example the vendored `ovseg` library) remain under their original licences; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The pretrained weights are licensed separately from the code, under **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)**, and are available on request through the Zenodo record ([DOI: 10.5281/zenodo.21836814](https://doi.org/10.5281/zenodo.21836814)).

## Acknowledgements

The segmentation backbone is built on the [`ovseg`](https://github.com/ThomasBudd/ovseg) library (vendored under `KCD/Segmentation/ovseg/`).
