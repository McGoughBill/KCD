# Literature Review and Lab Notes

> Background notes accumulated during the development of *Kidney cancer screening in non-contrast computed tomography via domain-adaptive deep learning* (accepted in Nature Health). Kept here for context — not user-facing documentation. See the project [README](../README.md) for setup and reproduction instructions.

## Related work

### Segmentation-based detection

[Toda et al](https://pubmed.ncbi.nlm.nih.gov/34935652/) use a volume thresholding, with an implicit confidence thresholding of 0.5.

[Pancreatic PANDA framework](https://www.nature.com/articles/s41591-023-02640-w) uses a multi-task CNN for segmentation and classification. They use a Dice loss for segmentation and a cross-entropy loss for classification.
This allows them to generate an ROC based on the classification head's probability output.

### NCCT and CECT paired datasets

[ORCA](https://orcascore.grand-challenge.org/Data/) - 40 patients, 40 scans in each modality
[COLTEA](https://github.com/ristea/cycle-transformer?tab=readme-ov-file) - 100 patients, 100 scans in each modality

### Cross-modality distillation

Here is the first relevant paper I have found: [Cross Modal Distillation for Supervision Transfer](https://arxiv.org/abs/1507.00448) - they simply use euclidian loss in latent feature space

This paper ([Weakly supervised segmentation with cross-modality equivariant constraints](https://www.sciencedirect.com/science/article/pii/S1361841522000275)) uses the Kullback-Leibler Divergence Loss over the softmaxed output probabilities

This paper ([Deep cross‐modality (MR‐CT) educed distillation learning for cone beam CT lung tumor segmentation](https://aapm.onlinelibrary.wiley.com/doi/abs/10.1002/mp.14902)) operates on unpaired MRI-CT datasets, and distills a pretrained MRI segmentation algorithm for segmentation in CT. They do this using Cycle GAN, which is a great approach for synthesizing paired data from unpaired datasets. Their cross modal self-distillation is simply L2 loss over latent features in the last two layers.

[Improving T1w MRI-based brain tumor segmentation using cross-modal distillation](https://spiedigitallibrary.org/conference-proceedings-of-spie/11596/115960Z/Improving-T1w-MRI-based-brain-tumor-segmentation-using-cross-modal/10.1117/12.2581067.full) uses Euclidean distance on latent features and a fairly convoluted entropy loss, that includes the ground truth (in one modality) and the soft AND hard segmentation outputs

### Additional datasets

CPTAC - TCIA
Acrin-6698


### Downstream fine-tune tasks

Seg and Detection for:

Hemmorhage using RSNA CT dataset (https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection?tab=readme-ov-file)
This emulates the downstream task from SparK paper and 100mill medical images paper

Lung Cancer / Covid-19 detection using LIDC-IDRI dataset and Covid-19 CT dataset (https://ieee-dataport.org/documents/lung-image-database-consortium-image-collection-lidc-idri)
(https://bmcresnotes.biomedcentral.com/articles/10.1186/s13104-021-05592-x) haghihi semantic genesis

Kidney cancer - my short paper dataset.

Semantic Genesis does finetuning from pretrain. MoCo uses Prototypical Networks for few shot classification?
SparK uses a simple linear layer on top of the pretrained model, trains only the final added linear layer for 10 epochs.

Perform a similar reduction in training set size as SparK! See how this performs as training dataset size
is reduced.

For seg downstream, we can copy vox2vec. For detection, we can copy SparK.

## Lab notes (development log)

1) Tried just plain training of 2D axial tile and 3D axial patch model on coreg_ncct dataset. Got cross-val AUC of 0.845 / 0.81 respectively. Tested both model on unseen data, got sensitivity of about 25%, whereas 3D model got 45%. We believe this is due to the integration technique of converting many slice-wise inferences into a kidney-wise prediction; 3D model is simply top-1, whereas we employed top-10 with 2D. We should test 2D in top-1, to ensure this comparison is fair and valid. 2D algorithm used pretraining ResNeXT weights, 3D was randomly initialised.
2) Given that the 2D model had some pretraining, we wanted to give the 3D model pretraining too. Thus, we ran it through kits23 and kits23sncct data before finetuning. We observed that the 3D model's cross-val performance increased to 0.84AUC.  I should do this pretraining on 2D classifier, to see if it improves its accuracy, too.
3) Then wanted to evaluate the effect of multi-task learning. We coded up a decoder branch to ResNeXT and trained it via segmentation and classification on the Kits23 dataset, before regular classification training on coreg_ncct. We observed that the cross-val performance of the 3D model increased to 0.86AUC. We should do this on the 2D model, too.

Tasks:

- [ ] Train 2D model in top-1 mode
- [ ] Train 2D model on coreg_ncct dataset after pretraining on kits23
- [ ] Train 2D model with decoder branch on kits23 MTL

After these tasks are complete, I should investigate further MTL improvements via CECT synthesis/prediction, based on this paper: https://www.medrxiv.org/content/10.1101/2022.01.12.22269120v1


