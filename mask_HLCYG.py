import nibabel as nib
import numpy as np
import os
from scipy.ndimage import binary_dilation
import sys
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
#take images 'nii.gz' from dir_A, mask with images from dir_B and save in dir_C.
# dilate mask by 25mm, using the header information from the original image

# directory A
# dir_A = '/rds/project/rds-sDuALntK11g/raw_data/nc_reg/images'
# dir_B = '/rds/project/rds-sDuALntK11g/predictions_nii/ce/[2 2 2]mm'
# dir_C = '/rds/project/rds-sDuALntK11g/raw_data/masked_nc_reg/images'

dir_B = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/predictions/coreg_ncct_v2/4mm_kidneybinary_customPP/HLCYG_fulldose/cross_validation'
raw_home = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/raw_data'
# dirs = ['coreg_ncct_v2','coreg_v2_HLCYG_0.1','coreg_v2_HLCYG_0.2','coreg_v2_HLCYG_0.3','coreg_v2_HLCYG_0.4','coreg_v2_HLCYG_0.45','coreg_v2_HLCYG_0.5','coreg_v2_HLCYG_0.55',
#         'coreg_v2_HLCYG_0.6','coreg_v2_HLCYG_0.65','coreg_v2_HLCYG_0.7','coreg_v2_HLCYG_0.75','coreg_v2_HLCYG_0.8','coreg_v2_HLCYG_0.85','coreg_v2_HLCYG_0.9',
#         'coreg_v2_HLCYG_0.95']

dirs = ['coreg_v2_HLCYG_0.2']
for im_fold in dirs:
    imfp = os.path.join(raw_home,im_fold,'images')
    lbfp = os.path.join(raw_home,im_fold,'labels')
    save_dir = os.path.join(raw_home,'AImasked_'+im_fold)
    save_dir_im = os.path.join(save_dir,'images')
    save_dir_lb = os.path.join(save_dir,'labels')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(os.path.join(save_dir_im))
        os.makedirs(os.path.join(save_dir_lb))

    # get all the files in directory A
    files_A = [f for f in os.listdir(imfp) if os.path.isfile(os.path.join(imfp, f)) and f.endswith('.nii.gz')]

    for file in files_A:
        if os.path.exists(os.path.join(save_dir_im, file)):
            continue
        # load the image
        image = nib.load(os.path.join(imfp, file)).get_fdata()
        # load the mask
        mask = nib.load(os.path.join(dir_B, file.replace('_noised',''))).get_fdata()
        shape = mask.shape
        header = nib.load(os.path.join(imfp, file)).header
        affine = nib.load(os.path.join(imfp, file)).affine

        #do a simple masked operations using the niftis. interpolate mask into 1x1x1, dilate mask by 20, then interpolate back to original resolution
        mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0)

        spacing = torch.tensor(header.get_zooms())
        new_spacing = torch.tensor([1,1,1])
        scale_factor = tuple((spacing/new_spacing).numpy())
        mask = F.interpolate(mask, scale_factor=scale_factor, mode='trilinear')

        #ditate the mask by 20mm
        mask = mask.squeeze().numpy()
        mask = binary_dilation(mask, iterations=20).astype(float)

        #interpolate back to original resolution
        mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0)
        mask = F.interpolate(mask, size=shape, mode='trilinear',align_corners=False)
        mask = mask.squeeze().numpy().astype(np.uint8)

        masked_image = np.where(mask, image, -500)


        # save the masked image
        masked_image_nii = nib.Nifti1Image(masked_image, affine)
        nib.save(masked_image_nii, os.path.join(save_dir_im, file))

        #copy the label
        os.system('cp '+os.path.join(lbfp,file)+' '+os.path.join(save_dir_lb,file))
