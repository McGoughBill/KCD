import numpy as np
import pandas as pd
import os
from sklearn.metrics import auc

fp = '/Users/mcgoug01/Dropbox (Personal)/PhD/thesis_plots/V1segdetection/largest_pred.csv'
label_fp = '/Users/mcgoug01/Dropbox (Personal)/PhD/thesis_plots/V1segdetection/labels.csv'
sizes_fp = '/Users/mcgoug01/Dropbox (Personal)/PhD/thesis_plots/V1segdetection/sizes_test.csv'

if not os.path.exists(fp):
    print('File not found:', fp)

predictions = pd.read_csv(fp,index_col=False)
label = pd.read_csv(label_fp,index_col=False)
sizes = pd.read_csv(sizes_fp,index_col=False)

def calculate_roc(predictions, label):
    thresholds = np.linspace(0, 1, 100)
    tpr = []
    fpr = []

    for threshold in thresholds:
        preds_binary = (predictions[['RightCancer', 'LeftCancer']] >= threshold).astype(int)
        tp = ((preds_binary['RightCancer'] == 1) & (label['RightCancer'] == 1)).sum()
        tn = ((preds_binary['RightCancer'] == 0) & (label['RightCancer'] == 0)).sum()
        fp = ((preds_binary['RightCancer'] == 1) & (label['RightCancer'] == 0)).sum()
        fn = ((preds_binary['RightCancer'] == 0) & (label['RightCancer'] == 1)).sum()

        tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)

    return np.array(tpr), np.array(fpr)
use_sizes = True  # Set to True if you want to use sizes for filtering



for use_sizes in [True, False]:
    if use_sizes:
        # filter CaseID in predictions and label based on sizes: make sure sizes < 335120
        predictions_small = predictions[predictions['CaseID'].isin(sizes[sizes['size'] < 33510]['CaseID'])]
        label_small = label[label['CaseID'].isin(sizes[sizes['size'] < 33510]['CaseID'])]
        tpr, fpr = calculate_roc(predictions_small, label_small)
        tnr = 1 - fpr  # True Negative Rate (TNR) is 1 - False Positive Rate (FPR)
        small_cancer_results = np.array([tpr, tnr])*100
    else:
        tpr, fpr = calculate_roc(predictions, label)
        tnr = 1 - fpr
        all_cancer_results = np.array([tpr, tnr])*100


print(all_cancer_results.shape)

#find point where tnr and tpr are maximised in all_cancer_results
max_index = np.argmax(all_cancer_results[0] + all_cancer_results[1])
operating_point = np.linspace(0, 1, 100)[max_index]
sensitivity = all_cancer_results[0][max_index]
specificity = all_cancer_results[1][max_index]
print(f'Operating point: {operating_point}, Sensitivity: {sensitivity}, Specificity: {specificity}')

# Calculate AUC for both cases
auc_all = auc(100 - all_cancer_results[1], all_cancer_results[0])/1e4
auc_small = auc(100 - small_cancer_results[1], small_cancer_results[0])/1e4

import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')

fig = plt.figure(figsize=(8,5))
plt.plot(100-all_cancer_results[1],all_cancer_results[0],label='Segmentation detection ({:.3f})'.format(auc_all),c='k')
plt.plot(100-small_cancer_results[1],small_cancer_results[0],label='Segmentation detection (Small RCs, {:.3f})'.format(auc_small),c='k',linestyle='--')
plt.xlabel('100 - Specificity / %',fontdict={'fontsize':20})
plt.ylabel('Sensitivity / %',fontdict={'fontsize':20})
plt.legend(fontsize=15,loc='lower right')
#change x and y tick labels to 12
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

plt.xlim([-2, 100])
plt.ylim([0, 102])
plt.savefig('/Users/mcgoug01/Dropbox (Personal)/PhD/thesis_plots/V1segdetection/testroc_seg.png')

plt.show()
