# this script will load the registered ncct and cect images (in .nii.gz format),
# crop the first and last two slices off each image, then crop any slices that are
# entirely equal to -500 HU. The cropped images will be saved in the different directory

import os
import numpy as np
import nibabel as nib

# define the path to the registered images
registered_ncct_path = '/path/to/kcd_data/data/seg_data/raw_data/masked_tcia_ncct/images'
registered_cect_path = '/path/to/kcd_data/data/seg_data/raw_data/masked_tcia_cect/images'

# coregistered images have identical names, so we can use the same list for both
registered_ncct_ims = [f for f in os.listdir(registered_ncct_path) if f.endswith('.nii.gz')]

# define the path to save the cropped images
cropped_ncct_path = '/path/to/kcd_data/data/seg_data/raw_data/cropped_tcia_ncct/images'
cropped_cect_path = '/path/to/kcd_data/data/seg_data/raw_data/cropped_tcia_cect/images'

# create the directories if they don't exist
if not os.path.exists(cropped_ncct_path):
    os.makedirs(cropped_ncct_path)
if not os.path.exists(cropped_cect_path):
    os.makedirs(cropped_cect_path)

# loop through the registered images
for im in registered_ncct_ims:
    # load the image
    ncct_im = nib.load(os.path.join(registered_ncct_path, im))
    cect_im = nib.load(os.path.join(registered_cect_path, im))

    # get the image data
    ncct_data = ncct_im.get_fdata()
    cect_data = cect_im.get_fdata()

    print(ncct_data.shape)

    # crop the first and last two slices
    ncct_data = ncct_data[:,:,2:-2]
    cect_data = cect_data[:,:,2:-2]

    # find the slices that are entirely -500 HU
    ncct_mask = np.all(ncct_data == -500, axis=(0,1))

    # crop the slices that are entirely -500 HU
    ncct_data = ncct_data[:,:,~ncct_mask]
    cect_data = cect_data[:,:,~ncct_mask]

    # save the cropped images
    ncct_cropped = nib.Nifti1Image(ncct_data, ncct_im.affine)
    cect_cropped = nib.Nifti1Image(cect_data, cect_im.affine)

    nib.save(ncct_cropped, os.path.join(cropped_ncct_path, im))
    nib.save(cect_cropped, os.path.join(cropped_cect_path, im))