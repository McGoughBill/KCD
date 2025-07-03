import numpy as np
import pandas as pd
import os
import nibabel as nib
from skimage.measure import label, regionprops
import numpy as np
import os
import torch
from scipy.ndimage.morphology import binary_fill_holes
import scipy.ndimage as spim
from skimage.measure import regionprops
from KCD.Detection.Preprocessing.ObjectFiles import object_creation_utils as ocu
from skimage.segmentation import watershed
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc


nii_label_loc = '/Users/mcgoug01/Downloads/labels_v1'

if not os.path.exists('/Users/mcgoug01/Downloads/valid_v1/sizes.csv'):

    with open('/Users/mcgoug01/Downloads/valid_v1/sizes.csv', 'w') as f:
        f.write('file,size\n')
    for file in os.listdir(nii_label_loc):
        if file.endswith('.nii.gz'):
            nii = nib.load(os.path.join(nii_label_loc, file))
            vox_size = np.prod(nii.header.get_zooms()[:3])
            print(file,vox_size,nii.header.get_zooms()[:3])
            # get largest contiguous region
            data = nii.get_fdata()
            labeled_data = label(data==2)
            regions = regionprops(labeled_data)
            try:
                largest_region = max(regions, key=lambda r: r.area)
                size = largest_region.area * vox_size
            except ValueError:
                size = 0

            with open('/Users/mcgoug01/Downloads/valid_v1/sizes.csv', 'a') as f:
                f.write(f'{file},{size}\n')


sizes_fp = '/Users/mcgoug01/Downloads/valid_v1/sizes.csv'
sizes = pd.read_csv(sizes_fp, index_col=False)
path = '/Users/mcgoug01/Downloads/valid_v1'
cancer_infp = '/Users/mcgoug01/Downloads/continuous_v1'

CV_results_fp = '/Users/mcgoug01/Downloads/valid_v1/results_CV.csv'

confidence_thresholds = np.append(np.arange(0,0.1,0.005),np.arange(0.1, 0.9, 0.1))
confidence_thresholds = np.append(confidence_thresholds,np.arange(0.9,1.02,0.01))
vol = 500

results =[]
percase_results =[]
small_results = []

if os.path.exists(CV_results_fp):
    results = pd.read_csv(CV_results_fp, index_col=False)
    print('Loaded previous results from:', CV_results_fp)
else:
    #only load files that end in .nii.gz
    for conf in confidence_thresholds:
        conf_results = []
        tp, fp, fn, tn = 0, 0, 0, 0
        small_tp, small_fp, small_fn, small_tn = 0, 0, 0, 0
        for file in [path for path in os.listdir(nii_label_loc) if path.endswith('.nii.gz')]:
            patient_results = [0,0,0,0] #tp, fp, fn, tn
            try:
                cancer_inf = nib.load(os.path.join(cancer_infp, file))
                cancer_label = nib.load(os.path.join(nii_label_loc, file))
                cancer_inf = nib.as_closest_canonical(cancer_inf)
                cancer_label = nib.as_closest_canonical(cancer_label)

            except:
                continue

            size = sizes[sizes['file'] == file]['size'].values[0]
            is_large = size > ((20**3)*np.pi*4/3)

            print(file,size,is_large)
            # assert all images have same shape
            assert cancer_inf.shape == cancer_label.shape
            #find spacing
            spacing = cancer_inf.header['pixdim'][1:4]

            #resize all to 2mm spacing using torch
            try:
                cancer_inf = torch.from_numpy(cancer_inf.get_fdata()).unsqueeze(0).unsqueeze(0)
                cancer_label = torch.from_numpy(cancer_label.get_fdata()).unsqueeze(0).unsqueeze(0)
            except:
                continue
            cancer_inf = torch.nn.functional.interpolate(cancer_inf, scale_factor=tuple(spacing/2), mode='trilinear')
            cancer_label = torch.nn.functional.interpolate(cancer_label, scale_factor=tuple(spacing/2), mode='trilinear', align_corners=False)

            #find cancerous voxels
            cancer_voxels = cancer_inf > conf
            # fill holes in cancerous voxels
            cancer_voxels = binary_fill_holes(cancer_voxels)
            #filter out regions with less than vol voxels using regionprops and spim
            cancer_voxels = spim.label(cancer_voxels.squeeze())[0]
            cancer_regions = np.array([region for region in regionprops(cancer_voxels) if region.area > vol])

            # reconstruct cancerous voxels from regions over vol
            cancer_voxels = np.zeros_like(cancer_voxels)
            for region in cancer_regions:
                cancer_voxels[region.coords[:,0],region.coords[:,1],region.coords[:,2]] = 2

            # colour kidney labels in 2 if cancerous, 1 if not
            travel_mask1 = cancer_label.squeeze().numpy().copy()
            #set 0 values to -1, set 1 values to 0 in travel mask
            travel_mask1[travel_mask1==0] = -1
            travel_mask1[travel_mask1==1] = 0
            cancer_kidney_labels = watershed(travel_mask1, markers=(cancer_label.numpy().squeeze()==2).astype(int),
                                      mask = (cancer_label.numpy().squeeze()>0).astype(int))
            cancer_kidney_labels[cancer_kidney_labels==1] = 2
            cancer_kidney_labels[(cancer_kidney_labels==0) & (cancer_label.numpy().squeeze()>0)] = 1

            # reconstruct labels voxels from cancer_kidney_labels, removing tiny regions
            label_voxels = np.zeros_like(cancer_kidney_labels)
            for label in [1,2]:
                for region in regionprops(spim.label(cancer_kidney_labels==label)[0]):
                    if region.area > 20:
                        label_voxels[region.coords[:,0],region.coords[:,1],region.coords[:,2]] = label

            # loop through all cancer predicted regions without a corresponding kidney label,
            # and add a false positive
            for region in regionprops(spim.label(cancer_voxels == 2)[0]):
                kid_reg = cancer_label[0,0,region.coords[:, 0], region.coords[:, 1], region.coords[:, 2]]
                if kid_reg.max() == 0:
                    fp += 1
                    patient_results[1] += 1
                    if is_large:
                        small_fp += 1

            # loop through all cancerous labels - checking if true or false positive
            for region in regionprops(spim.label(label_voxels==2)[0]):
                prediction_region = cancer_voxels[region.coords[:,0],region.coords[:,1],region.coords[:,2]]
                if prediction_region.max() <2:
                    fn+=1
                    patient_results[2] += 1
                    if is_large:
                        small_fn += 1
                else:
                    tp+=1
                    patient_results[0] += 1
                    if is_large:
                        small_tp += 1

            # loop through all healthy labels - checking if true or false negative
            for region in regionprops(spim.label(label_voxels==1)[0]):
                prediction_region = cancer_voxels[region.coords[:,0],region.coords[:,1],region.coords[:,2]]
                if prediction_region.max() <2:
                    tn+=1
                    patient_results[3] += 1
                    if is_large:
                        small_tn += 1
                else:
                    fp+=1
                    patient_results[1] += 1
                    if is_large:
                        small_fp += 1
            conf_results.append(patient_results)

        percase_results.append(conf_results)
        entry = {'confidence_threshold': conf, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                    'small_tp': small_tp, 'small_fp': small_fp, 'small_fn': small_fn, 'small_tn': small_tn}
        sensitivity = tp/(tp+fn+1e-6)
        specificity = tn/(tn+fp+1e-6)
        precision = tp/(tp+fp+1e-6)
        accuracy = (tp+tn)/(tp+tn+fp+fn)

        entry['small_sensitivity'] = small_tp/(small_tp+small_fn+1e-6)
        entry['small_specificity'] = small_tn/(small_tn+small_fp+1e-6)
        entry['small_precision'] = small_tp/(small_tp+small_fp+1e-6)
        entry['small_accuracy'] = (small_tp+small_tn)/(small_tp+small_tn+small_fp+small_fn)
        entry['sensitivity'] = sensitivity
        entry['specificity'] = specificity
        entry['precision'] = precision
        entry['accuracy'] = accuracy
        entry['dice'] = 2*tp/(2*tp+fp+fn+1e-6)

        print(entry)
        results.append(entry)

    results = pd.DataFrame(results)
    results.to_csv(CV_results_fp, index=False)
tpr = results['sensitivity'].values * 100
tnr = results['specificity'].values * 100

#sort tnr and tpr by tnr
sorted_indices = np.argsort(tpr)
tpr = tpr[sorted_indices]
tnr = tnr[sorted_indices]
auc = auc(tpr, tnr) / 10000

#plot sensitivity vs specificity
plt.figure(figsize=(8, 5))
plt.plot(100-tnr,tpr,label='Segmentation detection ({:.3f})'.format(auc),c='k')
plt.xlabel('100 - Specificity / %',fontdict={'fontsize':20})
plt.ylabel('Sensitivity / %',fontdict={'fontsize':20})
plt.legend(fontsize=15,loc='lower right')
#change x and y tick labels to 12
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

plt.xlim([-2, 100])
plt.ylim([0, 102])
plt.savefig('/Users/mcgoug01/Downloads/valid_v1/CVroc_seg.png')

plt.show()
