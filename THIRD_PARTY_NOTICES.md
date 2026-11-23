# Third-party notices

The PolyForm Strict License 1.0.0 in [`LICENSE`](LICENSE) applies to the original KCD code in this repository. It does not apply to the third-party components listed below, which are included in this repository and remain under the terms set by their original authors. Copyright and licence headers in individual files take precedence over this summary.

| Path | Origin | Upstream licence |
|---|---|---|
| `KCD/Segmentation/ovseg/` | [ovseg](https://github.com/ThomasBudd/ovseg) by Thomas Buddenkotte, with modifications | MIT |
| `KCD/Segmentation/ovseg/networks/mask2former_modeling/` | [Mask2Former](https://github.com/facebookresearch/Mask2Former) and [DETR](https://github.com/facebookresearch/detr), Copyright (c) Facebook, Inc. and its affiliates | MIT (Mask2Former), Apache-2.0 (DETR) |
| `KCD/Segmentation/ovseg/networks/swin_utils.py`, `KCD/Segmentation/ovseg/networks/swinunetr_heads.py` | Copyright (c) ByteDance, Inc. and its affiliates | See the file headers and the upstream project |
| `KCD/Detection/Training/finetuning_dira/segmentation_models_pytorch/` | [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) by Pavel Iakubovskii, as adapted in [DiRA](https://github.com/fhaghighi/DiRA); some files carry their own notices (for example BSD 3-Clause, Copyright (c) Soumith Chintala 2016, and Copyright (c) Facebook, Inc. and its affiliates) | MIT (segmentation_models.pytorch); see file headers and the DiRA repository for the rest |

If you believe a component is missing from this list or attributed incorrectly, please open an issue.
