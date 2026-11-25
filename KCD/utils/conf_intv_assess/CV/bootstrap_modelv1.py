import numpy as np
import pandas as pd
import os
import nibabel as nib
from skimage.measure import label, regionprops

if not os.path.exists('/path/to/kcd_data/Downloads/valid_v1/sizes.csv'):
    nii_label_loc = '/path/to/kcd_data/Downloads/labels_v1'
    with open('/path/to/kcd_data/Downloads/valid_v1/sizes.csv', 'w') as f:
        f.write('file,size\n')
    for file in os.listdir(nii_label_loc):
        if file.endswith('.nii.gz'):
            nii = nib.load(os.path.join(nii_label_loc, file))
            vox_size = np.prod(nii.header.get_zooms()[:3])
            # get largest contiguous region
            data = nii.get_fdata()
            data = data.astype(bool)
            labeled_data = label(data)
            regions = regionprops(labeled_data)
            largest_region = max(regions, key=lambda r: r.area)
            size = largest_region.area * vox_size
            with open('/path/to/kcd_data/Downloads/valid_v1/sizes.csv', 'a') as f:
                f.write(f'{file},{size}\n')



sizes_fp = '/path/to/kcd_data/Downloads/valid_v1/sizes.csv'
conf_per_patient_fp = '/path/to/kcd_data/Downloads/valid_v1/results_confwise.npy'
sizes = np.genfromtxt(sizes_fp, delimiter=',')
print(min(sizes), max(sizes))
large = sizes > ((20**3)*np.pi*4/3)
conf_per_patient = np.load(conf_per_patient_fp)
print(conf_per_patient.shape, sizes.shape)


confidence_thresholds = np.append(np.arange(0,0.1,0.01),np.arange(0.1, 0.9, 0.05))
confidence_thresholds = np.append(confidence_thresholds,np.arange(0.9,0.98,0.02))
confidence_thresholds = np.append(confidence_thresholds,np.arange(0.98,1.005,0.005))
conf_intervals = confidence_thresholds

samples_per_bootstrap = len(sizes)
number_samples = 10000
num_conf_intervals = len(conf_intervals)
high_spec = 0.98
low_spec = 0.5**0.5

small_ratio = 43/43
# small_ratio = 1
# https://link.springer.com/article/10.1007/s00261-017-1376-0

results = []
nonbootstrapped_results = []
for conf in range(num_conf_intervals):
    conf_res = []
    for sample in range(number_samples):
        relevant_interval = conf_per_patient[conf,:]

        small_tumours = relevant_interval[~large]
        large_tumours = relevant_interval[large]

        small_indices = np.random.randint(0,len(small_tumours), size=int(samples_per_bootstrap*small_ratio))
        large_indices = np.random.randint(0,len(large_tumours), size=int(samples_per_bootstrap*(1-small_ratio)))

        small_sample = small_tumours[small_indices]
        large_sample = large_tumours[large_indices]

        sample = np.concatenate((small_sample, large_sample))

        tp, fp, fn, tn = sample.sum(axis=0)
        sens = tp/(tp+fn)
        spec = (tn/(tn+fp))
        conf_res.append((sens, spec))
    conf_res = np.array(conf_res)
    results.append(conf_res)
    # calculate non-bootstrapped results from the mean of the conf_per_patient
    tp, fp, fn, tn = conf_per_patient[conf,:].sum(axis=0)
    nonbs_sens = tp/(tp+fn)
    nonbs_spec = (tn/(tn+fp))
    nonbootstrapped_results.append((nonbs_sens, nonbs_spec))

results = np.array(results)
means = results.mean(axis=1)
stds = results.std(axis=1)

nonbootstrapped_results = np.array(nonbootstrapped_results)

# plot an ROC with means, with error bars of 1 std
sens = means[:,0]
spec = means[:,1]
sens_std = stds[:,0]
spec_std = stds[:,1]


conf_intervals = np.array(conf_intervals)
#prepend and append 0 to sens_std and spec_std

#find AUC
AUC = np.trapz(sens, spec)

import matplotlib.pyplot as plt
from matplotlib import gridspec

plt.switch_backend('TkAgg')
# subplot of two figures in one row - one with width 7, other with width 3, makign a cumulative
# width of 10. both with height 8
fig = plt.figure(figsize=(8,5))

#plot the non-bootstrapped ROC curve from nonbootstrapped_results
nonbootstrapped_results = nonbootstrapped_results.T
print(nonbootstrapped_results.shape)

# sort by ascending first column and descending second column - merge sort by second column first
# nonbootstrapped_results = nonbootstrapped_results[:,nonbootstrapped_results[1].argsort(kind='mergesort')[::-1]]
nonbs_sens, nonbs_spec = nonbootstrapped_results


plt.plot(100 - nonbs_spec*100, nonbs_sens*100, c = 'black', label='Segmentation detection ({:.3f})'.format(-1*np.trapz(nonbs_sens, nonbs_spec)),
        linestyle='-')
#plot semi-transparent error line with fill between
# ax.set_xlabel('100 - Specificity / %', fontsize=20,fontdict={'family':'Helvetica'})
# ax.set_ylabel('Sensitivity / %', fontsize=20,fontdict={'family':'Helvetica'})
#do above using plt
plt.xlabel('100 - Specificity / %', fontsize=20)
plt.ylabel('Sensitivity / %', fontsize=20)


#xlim and ylim -2 to 102
plt.xlim(-2, 102)
plt.ylim(-2, 102)
#convert x and y ticks to fontsize 12
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# set xticks and yticks

#plot legend
plt.legend(fontsize=15, loc='lower right')
print(AUC)
plt.tight_layout()
#change width of ax[1] to 3
plt.savefig('/path/to/kcd_data/Downloads/cvroc_seg.png')
plt.show()
plt.close()
