import os
import numpy as np
import pandas as pd

def eval_ZS_seg(pred, gt, num_classes):
    """
    Evaluate segmentation results for zero-shot setting.
    Args:
        pred: prediction results, shape (H, W)
        gt: ground truth, shape (H, W)
        num_classes: number of classes
    Returns:
        Dice: dice coefficient - number
        Sensitivity: recall - number
        Specificity: precision - number
    """
    # Check if the input is valid
    if pred.shape != gt.shape:
        raise ValueError("Shape mismatch between prediction and ground truth")

    # Initialize variables
    TP = np.zeros(num_classes)
    FP = np.zeros(num_classes)
    FN = np.zeros(num_classes)
    TN = np.zeros(num_classes)

    # Calculate confusion matrix
    for i in range(num_classes):
        TP[i] = np.sum(np.logical_and(pred == i, gt == i))
        FP[i] = np.sum(np.logical_and(pred == i, gt != i))
        FN[i] = np.sum(np.logical_and(pred != i, gt == i))
        TN[i] = np.sum(np.logical_and(pred != i, gt != i))

    # Calculate Dice coefficient
    Dice = np.sum(2*TP) / (np.sum(2*TP) + np.sum(FP) + np.sum(FN))

    # Calculate Sensitivity
    Sensitivity = np.sum(TP) / (np.sum(TP) + np.sum(FN))

    # Calculate Specificity
    Specificity = np.sum(TN) / (np.sum(TN) + np.sum(FP))

    return Dice, Sensitivity, Specificity

NCCT_path = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/predictions_npy/coreg_ncct/4,4,4mm_AA_False'
CECT_path = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/predictions_npy/coreg_cect/4,4,4mm_AA_True'
labels_path = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/preprocessed/coreg_ncct/4mm_binary/labels'
save_path = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/predictions_npy/coreg_ncct'

#extract files from each loc that ends in .npy
NCCT_files = [f for f in os.listdir(NCCT_path) if f.endswith('.npy')]
CECT_files = [f for f in os.listdir(CECT_path) if f.endswith('.npy')]
labels_files = [f for f in os.listdir(labels_path) if f.endswith('.npy')]
results = []
for i in range(len(NCCT_files)):
    NCCT = np.load(os.path.join(NCCT_path, NCCT_files[i]))
    CECT = np.load(os.path.join(CECT_path, CECT_files[i]))
    labels = np.load(os.path.join(labels_path, labels_files[i]))

    Dice_NCCT, Sensitivity_NCCT, Specificity_NCCT = eval_ZS_seg(NCCT, labels, 2)
    Dice_CECT, Sensitivity_CECT, Specificity_CECT = eval_ZS_seg(CECT, labels, 2)

    print(f"NCCT: Dice = {Dice_NCCT}, Sensitivity = {Sensitivity_NCCT}, Specificity = {Specificity_NCCT}")
    print(f"CECT: Dice = {Dice_CECT}, Sensitivity = {Sensitivity_CECT}, Specificity = {Specificity_CECT}")
    print("\n")
    entry_NCCT = {'case': NCCT_files[i], 'Dice': Dice_NCCT, 'Sensitivity': Sensitivity_NCCT, 'Specificity': Specificity_NCCT}
    entry_CECT = {'case': CECT_files[i], 'Dice': Dice_CECT, 'Sensitivity': Sensitivity_CECT, 'Specificity': Specificity_CECT}
    results.append(entry_NCCT)
    results.append(entry_CECT)

df = pd.DataFrame(results)
df.to_csv(os.path.join(save_path, 'results.csv'))
print("Results saved to csv file")
print(f'Casewise average Dice for NCCT: {df["Dice"].iloc[::2].mean()}')
print(f'Casewise average Dice for CECT: {df["Dice"].iloc[1::2].mean()}')
print(f'Casewise average Sensitivity for NCCT: {df["Sensitivity"].iloc[::2].mean()}')
print(f'Casewise average Sensitivity for CECT: {df["Sensitivity"].iloc[1::2].mean()}')
print(f'Casewise average Specificity for NCCT: {df["Specificity"].iloc[::2].mean()}')
print(f'Casewise average Specificity for CECT: {df["Specificity"].iloc[1::2].mean()}')