import os
import pandas as pd
import numpy as np
import nibabel as nib
from torch.nn.functional import interpolate
import torch

import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')

meta_folder = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/AI/predictions/lowdose_ROC'
sub_folders = [os.path.join(meta_folder, f,'predictions.csv') for f in os.listdir(meta_folder) if os.path.isdir(os.path.join(meta_folder, f))]
names_sub_folders = [float(f.split('/')[-2]) for f in sub_folders]

# sort sub_folders by names_sub_folders
sub_folders = [x for _,x in sorted(zip(names_sub_folders,sub_folders))]
names_sub_folders = sorted(names_sub_folders)

highacc_pred_csv_fp = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/AI/predictions/lowdose_test/high_acc/predictions.csv'
iztok_pred_csv_fp = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/iztokLD.csv'
cathal_pred_csv_fp = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/cathalLD.csv'
truth_csv_fp = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/Bill/FD/internaltest_metadata.csv'
hania_pred_csv_fp = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/haniaLD.csv'

highacc_fd_pred_home = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/AI/predictions/fulldose_test/high_acc'
fd_continuous_home = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/AI/predictions/full_dose_continuous'
ld_continuous_home = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/AI/predictions/low_dose_continuous'
highacc_ld_pred_home = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/AI/predictions/lowdose_test/high_acc'
lowdose_image_path = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/Bill/LD/anon_noised/'
fulldose_image_path = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/raw_data/test_set_v2/images/'
add_cect_path = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/raw_data/add'
kits_cect_path = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/raw_data/kits23/'
# find the StudyID where cancer was missed by each radiologist that were found by the AI


save_loc = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/nat_paper'
# read in the AIpredictions
highacc_pred = pd.read_csv(highacc_pred_csv_fp)

# read in the radiologist predictions
iztok_pred = pd.read_csv(iztok_pred_csv_fp)
cathal_pred = pd.read_csv(cathal_pred_csv_fp)
hania_pred = pd.read_csv(hania_pred_csv_fp)

# plot cases ID23 (posneg), ID42 (negneg), ID12 (pospos), None (negpos)

# match the StudyID above to find the OriginalID in truth_csv
# read in the truth
truth = pd.read_csv(truth_csv_fp)

ids = ['ID23','ID42','ID12']
original_ids = []
for i,row in highacc_pred.iterrows():
    if row['StudyID'] in ids:
        original_ids.append(truth[truth['StudyID'] == row['StudyID']]['OriginalID'].values[0])


original_ids = [original_ids[1],original_ids[2],original_ids[0]]

#load the images, AI preds, and labels
low_dose_images = []
fulldose_images = []
contrastenhanced_images = []
lowdose_ai_preds = []
fulldose_ai_preds = []
labels = []

for id,ai,studyid in zip(original_ids,['pos','neg','pos'],ids):
    print(id,ai,studyid)
    ldim_path = os.path.join(lowdose_image_path, studyid+'.nii.gz')
    fdim_path = os.path.join(fulldose_image_path, id)
    if id.startswith('RCC'):
        cectim_path = os.path.join(add_cect_path, 'images','Rcc'+id[3:])
        cectlb_path = os.path.join(add_cect_path, 'labels','Rcc'+id[3:])
    else:
        cectim_path = os.path.join(kits_cect_path,'images', 'case_'+id[5:])
        cectlb_path = os.path.join(kits_cect_path,'labels', 'case_'+id[5:])

    if ai == 'pos':
        ldpred_path = os.path.join(highacc_ld_pred_home, 'positives_nii',studyid+'.nii.gz')
        fdpred_path = os.path.join(highacc_fd_pred_home, 'positives_nii',studyid+'.nii.gz')
    else:
        ldpred_path = os.path.join(highacc_ld_pred_home, 'negatives_nii',studyid+'.nii.gz')
        fdpred_path = os.path.join(highacc_fd_pred_home, 'negatives_nii',studyid+'.nii.gz')

    low_dose_images.append(nib.load(ldim_path).get_fdata())
    fulldose_images.append(nib.load(fdim_path).get_fdata())
    contrastenhanced_images.append(nib.load(cectim_path).get_fdata())
    lowdose_ai_preds.append(nib.load(ldpred_path).get_fdata())
    try:
        fulldose_ai_preds.append(nib.load(fdpred_path).get_fdata())
    except:
        fulldose_ai_preds.append(nib.load( os.path.join(highacc_fd_pred_home, 'positives_nii',studyid+'.nii.gz')).get_fdata())
    labels.append(nib.load(cectlb_path).get_fdata())


# plot each in a 3x3 grid - top row is pos neg, with 3 images, lowdose (pred overlay), fulldose (pred overlay), contrast enhanced (label overlay)
# middle row is neg neg, with 3 images, lowdose (pred overlay), fulldose (pred overlay), contrast enhanced (label overlay)
# bottom row is pos pos, with 3 images, lowdose (pred overlay), fulldose (pred overlay), contrast enhanced (label overlay)

fig, ax = plt.subplots(3,4,figsize=(20,20))

pie_sizes= [13.56,13.56,72.89]
pie_names = [['13.56%','',''],['','13.56%',''],['','','72.89%']]
for i in range(3):
    # if i<2, rotate all images by 90 degrees and sample slice from the last axis, otherwise
    # sample slice from the first axis

    low_dose = np.flip(low_dose_images[i].T)
    fulldose = np.flip(fulldose_images[i].T)
    contrastenhanced = contrastenhanced_images[i].T
    lowdose_pred = np.flip(lowdose_ai_preds[i].T)
    fulldose_pred = np.flip(fulldose_ai_preds[i].T)

    if (i==0) or (i==2):
        label = labels[i].T ==1
        low_dose = np.flip(low_dose,axis=2)
        fulldose = np.flip(fulldose,axis=2)
        lowdose_pred = np.flip(lowdose_pred,axis=2)
        # fulldose_pred = np.flip(fulldose_pred,axis=2)
    else:
        label = labels[i] ==2
        contrastenhanced = contrastenhanced_images[i]

    if i==0:
        label_slice = np.argmax(np.sum(label, axis=(1, 2)))-1
        pred_slice = np.argmax(np.sum(lowdose_pred, axis=(1, 2)))
        upper_left = 270,280
        lower_right = 370,380
    elif i==2:
        upper_left = 110,260
        lower_right = 210,360
        label_slice = np.argmax(np.sum(label, axis=(1, 2)))
        pred_slice = int(low_dose.shape[0]/2) +8
    else:
        upper_left = 115,190
        lower_right = 215,290
        #flip low dose and preds updown
        low_dose = np.flip(np.flip(low_dose,axis=1),2)
        fulldose = np.flip(np.flip(fulldose,axis=1),2)
        lowdose_pred = np.flip(np.flip(lowdose_pred,axis=1),2)
        # fulldose_pred = np.flip(np.flip(fulldose_pred,axis=1),2)
        label_slice = np.argmax(np.sum(label, axis=(1, 2)))
        pred_slice = int(low_dose.shape[0]/2)-1

    #interpolate the fulldose_pred to match the size of the 3D images
    fulldose_pred = interpolate(torch.tensor(fulldose_pred.copy()).unsqueeze(0).unsqueeze(0),size = fulldose.shape,mode='trilinear').squeeze().numpy()

    if i==1:
        pred_slice = np.argmax(np.sum(fulldose_pred, axis=(1, 2)))

        ax[i,0].imshow(low_dose[pred_slice-5][upper_left[1]:lower_right[1],upper_left[0]:lower_right[0]],cmap='gray',
                       vmin=-200,vmax=200)
        ax[i,1].imshow(fulldose[pred_slice-5][upper_left[1]:lower_right[1],upper_left[0]:lower_right[0]],cmap='gray',
                         vmin=-200,vmax=200)
    else:
        ax[i,0].imshow(low_dose[pred_slice][upper_left[1]:lower_right[1],upper_left[0]:lower_right[0]],cmap='gray',
                       vmin=-200,vmax=200)
        ax[i,1].imshow(fulldose[pred_slice][upper_left[1]:lower_right[1],upper_left[0]:lower_right[0]],cmap='gray',
                         vmin=-200,vmax=200)
    ax[i,2].imshow(contrastenhanced[label_slice][upper_left[1]:lower_right[1],upper_left[0]:lower_right[0]],cmap='gray',
                     vmin=-200,vmax=200)

    # #overlay the continuous predictions as heatmaps, and set transparency according to the value of the heatmap
    # ax[i,0].imshow(ld_cont[pred_slice][upper_left[1]:lower_right[1],upper_left[0]:lower_right[0]]/100000,cmap='hot',alpha=0.2)
    # ax[i,1].imshow(fd_cont[pred_slice][upper_left[1]:lower_right[1],upper_left[0]:lower_right[0]]/100000,cmap='hot',alpha=0.2)

    #draw contour around where pred>0.7
    ax[i,0].contour(lowdose_pred[pred_slice][upper_left[1]:lower_right[1],upper_left[0]:lower_right[0]]>0,colors='red',linewidths=5,linestyles='dashed')
    ax[i,1].contour(fulldose_pred[pred_slice][upper_left[1]:lower_right[1],upper_left[0]:lower_right[0]]>0,colors='red',linewidths=5,linestyles='dashed')
    #print sum of both slices in contour above
    print(np.sum(lowdose_pred[pred_slice][upper_left[1]:lower_right[1],upper_left[0]:lower_right[0]]>0))
    print(np.sum(fulldose_pred[pred_slice][upper_left[1]:lower_right[1],upper_left[0]:lower_right[0]]>0))

    # turn of all grids
    for j in range(3):
        ax[i,j].grid(False)
        ax[i,j].axis('off')

    #overlay label on contrast enhanced as contours
    ax[i,2].contour(label[label_slice][upper_left[1]:lower_right[1],upper_left[0]:lower_right[0]],colors='green',linewidths=5)

    #draw pie chart - highlight only ith name and size in the pie chart
    #set explode to 0.1 for the ith slice
    #set text to be larger and bold on the ith slice
    explode = [0.07 if j==i else 0 for j in range(3)]
    ax[i,3].pie(pie_sizes,labels =pie_names[i],startangle=0,explode = explode,
                shadow={'ox': -0.04, 'edgecolor': 'grey', 'shade': 0.9},
                colors=['red' if j==i else 'slategrey' for j in range(3)],
                textprops={'fontsize': 20, 'fontweight': 'bold'})
# set small spacing between each column and row
plt.subplots_adjust(wspace=0.05,hspace=0.05)
plt.tight_layout()
plt.savefig(os.path.join(save_loc,'lowdose_assessment.png'))
plt.show(block=True)
print(original_ids)