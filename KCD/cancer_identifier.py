

import os
os.environ['OV_DATA_BASE'] = "/home/wcm23/rds/rds-artist-sDuALntK11g"
import shutil
import numpy as np
from scipy.ndimage import label
import matplotlib.pyplot as plt
import nibabel as nib

# define the folders to copy from and to
src_dir = '/home/wcm23/rds/rds-artist-sDuALntK11g/raw_data/masked_final_pairedpretrain_nc/images'
pred_dir = '/home/wcm23/rds/rds-artist-sDuALntK11g/predictions_nii/masked_final_pairedpretrain_ce/cancer_[2 2 2]mm'
dest_dir = '/home/wcm23/rds/rds-artist-sDuALntK11g/raw_data/masked_final_pairedpretrain_nc_literallyall/'

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
    os.makedirs(os.path.join(dest_dir, 'images'))
    os.makedirs(os.path.join(dest_dir, 'labels'))

# loop through predictions, if cancer (more than 1 voxel ==1) is detected, copy the image and label to the new folder
for file in os.listdir(pred_dir):
    if file.endswith('.nii.gz'):
        pred_path = os.path.join(pred_dir, file)
        im_path = os.path.join(src_dir, file)
        dest_im_path = os.path.join(dest_dir, 'images', file)
        dest_label_path = os.path.join(dest_dir, 'labels', file)
        # pred = nib.load(pred_path).get_fdata()

        # if np.sum(pred) > 1:
        shutil.copy(im_path, dest_im_path)
        shutil.copy(pred_path, dest_label_path)
        print('Cancer detected in', file)