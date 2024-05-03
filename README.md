*Automated Small Kidney Cancer Detection in Non-Contrast Computed Tomography*

[Link to paper](https://arxiv.org/abs/2312.05258)



For segmentation-based detection:

[Toda et al](https://pubmed.ncbi.nlm.nih.gov/34935652/) use a volume thresholding, with an implicit confidence thresholding of 0.5.

[Pancreatic PANDA framework](https://www.nature.com/articles/s41591-023-02640-w) uses a multi-task CNN for segmentation and classification. They use a Dice loss for segmentation and a cross-entropy loss for classification.
This allows them to generate an ROC based on the classification head's probability output.

For NCCT and CECT paired datasets:

[ORCA](https://orcascore.grand-challenge.org/Data/) - 40 patients, 40 scans in each modality
[COLTEA](https://github.com/ristea/cycle-transformer?tab=readme-ov-file) - 100 patients, 100 scans in each modality

Cross modality distillation papers:

Here is the first relevant paper I have found: [Cross Modal Distillation for Supervision Transfer](https://arxiv.org/abs/1507.00448) - they simply use euclidian loss in latent feature space

This paper ([Weakly supervised segmentation with cross-modality equivariant constraints](https://www.sciencedirect.com/science/article/pii/S1361841522000275)) uses the Kullback-Leibler Divergence Loss over the softmaxed output probabilities

This paper ([Deep cross‐modality (MR‐CT) educed distillation learning for cone beam CT lung tumor segmentation](https://aapm.onlinelibrary.wiley.com/doi/abs/10.1002/mp.14902)) operates on unpaired MRI-CT datasets, and distills a pretrained MRI segmentation algorithm for segmentation in CT. They do this using Cycle GAN, which is a great approach for synthesizing paired data from unpaired datasets. Their cross modal self-distillation is simply L2 loss over latent features in the last two layers.

[Improving T1w MRI-based brain tumor segmentation using cross-modal distillation](https://spiedigitallibrary.org/conference-proceedings-of-spie/11596/115960Z/Improving-T1w-MRI-based-brain-tumor-segmentation-using-cross-modal/10.1117/12.2581067.full) uses Euclidean distance on latent features and a fairly convoluted entropy loss, that includes the ground truth (in one modality) and the soft AND hard segmentation outputs

More datasets:

CPTAC - TCIA
Acrin-6698


Downstream finetune tasks:

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