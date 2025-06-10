import os
import pandas as pd
import numpy as np
import nibabel as nib
from skimage.measure import label, regionprops

meta_folder = '/Users/mcgoug01/Dropbox (Personal)/PhD/thesis_plots/V1segdetection'
sub_folders = [os.path.join(meta_folder, f,'predictions.csv') for f in os.listdir(meta_folder) if os.path.isdir(os.path.join(meta_folder, f))]
names_sub_folders = [float(f.split('/')[-2]) for f in sub_folders]

# sort sub_folders by names_sub_folders
sub_folders = [x for _,x in sorted(zip(names_sub_folders,sub_folders))]
names_sub_folders = sorted(names_sub_folders)

truth_csv_fp = '/Users/mcgoug01/Dropbox (Personal)/PhD/thesis_plots/V1segdetection/labels.csv'
sizes_fp = '/Users/mcgoug01/Dropbox (Personal)/PhD/thesis_plots/V1segdetection/sizes_test.csv'
if not os.path.exists(sizes_fp):
    seg_path = '/Users/mcgoug01/Downloads/untitled folder'
    with open(sizes_fp, 'w') as f:
        f.write('CaseID,size\n')
    for file in os.listdir(seg_path):
        if file.endswith('.nii.gz'):
            if file.startswith('Rcc'):
                label_targ = 1
            else:
                label_targ = 2
            nii = nib.load(os.path.join(seg_path, file))
            vox_size = np.prod(nii.header.get_zooms()[:3])
            # get largest contiguous region
            data = nii.get_fdata() == label_targ
            data = data.astype(bool)
            labeled_data = label(data)
            regions = regionprops(labeled_data)
            if len(regions) == 0:
                size = 0
            else:
                largest_region = max(regions, key=lambda r: r.area)
                size = largest_region.area * vox_size
            with open(sizes_fp, 'a') as f:
                f.write(f'{file},{size}\n')

sizes_df = pd.read_csv(sizes_fp)

names = names_sub_folders
folders = sub_folders
small_ai_results = [[100,0]]
all_ai_results = [[100,0]]
for size_constraint in [True,False]:
    for name,pred_csv_fp in zip(names,folders):

        # pred and columns have 'RightCancer', 'LeftCancer' columns - trust has answers, pred has predictions
        #calc sens and spec
        pred = pd.read_csv(pred_csv_fp)
        truth = pd.read_csv(truth_csv_fp)

        #rename truth RightCancer and LeftCancer to RightCancerLabel and LeftCancerLabel
        truth.rename(columns={'RightCancer': 'RightCancerLabel', 'LeftCancer': 'LeftCancerLabel'}, inplace=True)
        #merge pred and truth on CaseID
        pred = pred.merge(truth[['CaseID', 'RightCancerLabel', 'LeftCancerLabel']], on='CaseID', how='left')
        #merge sizes_df on CaseID
        pred = pred.merge(sizes_df, on='CaseID', how='left')
        # exclude Rcc_112.nii.gz
        pred = pred[pred['CaseID'] != 'Rcc_112.nii.gz']


        # filter out patients with size greater than 31500
        if size_constraint:
            pred = pred[pred['size'] < 33510]

        # get the truth values
        truth_right = pred['RightCancerLabel'].values.astype(bool)
        truth_left = pred['LeftCancerLabel'].values.astype(bool)

        # get the predicted values
        pred_right = pred['RightCancer'].values.astype(bool)
        pred_left = pred['LeftCancer'].values.astype(bool)



        # get the number of patients
        num_patients = len(truth_right)

        # get the number of true positives, true negatives, false positives, false negatives
        tp_right = np.sum(np.logical_and(truth_right, pred_right))
        tn_right = np.sum(np.logical_and(np.logical_not(truth_right), np.logical_not(pred_right)))
        fp_right = np.sum(np.logical_and(np.logical_not(truth_right), pred_right))
        fn_right = np.sum(np.logical_and(truth_right, np.logical_not(pred_right)))

        tp_left = np.sum(np.logical_and(truth_left, pred_left))
        tn_left = np.sum(np.logical_and(np.logical_not(truth_left), np.logical_not(pred_left)))
        fp_left = np.sum(np.logical_and(np.logical_not(truth_left), pred_left))
        fn_left = np.sum(np.logical_and(truth_left, np.logical_not(pred_left)))

        #print dataframe rows for all false negatives
        fn_right_indices = np.where(np.logical_and(truth_right, np.logical_not(pred_right)))[0]
        fn_left_indices = np.where(np.logical_and(truth_left, np.logical_not(pred_left)))[0]


        tp = tp_right + tp_left
        tn = tn_right + tn_left
        fp = fp_right + fp_left
        fn = fn_right + fn_left

        sens = 100*tp/(tp+fn)
        spec = 100*tn/(tn+fp)
        print(f'{name}:')
        #print sens and spec to 2dp
        print(f'Sensitivity: {sens:.2f}%')
        print(f'Specificity: {spec:.2f}%')
        print()

        if name == 'Iztok' and name != 'Cathal':
            iztok = [sens,spec]
        elif name == 'Cathal':
            cathal = [sens,spec]
        elif name == 'Hania':
            hania = [sens,spec]
        else:
            if name == 0.120:
                high_sens = [sens,spec]
            elif name == 0.550:
                high_acc = [sens,spec]
            elif name == 0.994:
                high_spec = [sens,spec]
            if size_constraint:
                small_ai_results.append([sens,spec])
            else:
                all_ai_results.append([sens,spec])

    if size_constraint:
        small_ai_results.append([0,100])
        small_ai_results = np.array(small_ai_results)
        # sort by ascending first column and descending second column - merge sort by second column first
        small_ai_results = small_ai_results[small_ai_results[:,1].argsort(kind='mergesort')[::-1]]
        small_ai_results = small_ai_results[small_ai_results[:,0].argsort(kind='mergesort')]
        auc_small = np.abs(np.trapz(small_ai_results[:,0],small_ai_results[:,1])/10000)
    else:
        all_ai_results.append([0,100])
        all_ai_results = np.array(all_ai_results)
        # sort by ascending first column and descending second column - merge sort by second column first
        all_ai_results = all_ai_results[all_ai_results[:,1].argsort(kind='mergesort')[::-1]]
        all_ai_results = all_ai_results[all_ai_results[:,0].argsort(kind='mergesort')]
        #calculate AUC for ai_results
        auc_all = np.abs(np.trapz(all_ai_results[:,0],all_ai_results[:,1])/10000)

#make marker star
marker = '*'

#plot ROC curve, and plot the point for Iztok and Cathal
# also, plot the operating points for the AI models (high sensitivity, high accuracy, high specificity)
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,5))
plt.switch_backend('TkAgg')
plt.plot(100-all_ai_results[:,1],all_ai_results[:,0],label='Segmentation detection ({:.3f})'.format(auc_all),c='k')
plt.plot(100-small_ai_results[:,1],small_ai_results[:,0],label='Segmentation detection (Small RCs, {:.3f})'.format(auc_small),c='k',linestyle='--')
plt.xlabel('100 - Specificity / %',fontdict={'fontsize':20})
plt.ylabel('Sensitivity / %',fontdict={'fontsize':20})
plt.legend(fontsize=15,loc='lower right')
#change x and y tick labels to 12
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

plt.xlim([-2,100])
plt.ylim([0,102])

plt.savefig('/Users/mcgoug01/Downloads/testroc_seg.png')
plt.show()

