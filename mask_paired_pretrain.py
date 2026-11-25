import nibabel as nib
import numpy as np
import os
from scipy.ndimage import binary_dilation
import sys
#take images 'nii.gz' from dir_A, mask with images from dir_B and save in dir_C.
# dilate mask by 25mm, using the header information from the original image

# directory A
# dir_A = '/path/to/kcd_data/raw_data/nc_reg/images'
# dir_B = '/path/to/kcd_data/predictions_nii/ce/[2 2 2]mm'
# dir_C = '/path/to/kcd_data/raw_data/masked_nc_reg/images'

dir_A = '/path/to/kcd_data/raw_data/final_pairedpretrain_ce/images'
dir_B = '/path/to/kcd_data/predictions_nii/final_pairedpretrain_ce/[2 2 2]mm'
dir_C = '/path/to/kcd_data/raw_data/masked_final_pairedpretrain_ce/images'

if not os.path.exists(dir_C):
    os.makedirs(dir_C)

# get all the files in directory A
files_A = [f for f in os.listdir(dir_A) if os.path.isfile(os.path.join(dir_A, f)) and f.endswith('.nii.gz')]
files_B = [f for f in os.listdir(dir_B) if os.path.isfile(os.path.join(dir_B, f)) and f.endswith('.nii.gz')]

# loop through all the files in directory A
for file in files_A:
    if file not in files_B:
        print(f"File {file} not in directory B")
        continue
    # load the image from directory A
    image = nib.load(os.path.join(dir_A, file))
    # load the mask from directory B
    mask = nib.load(os.path.join(dir_B, file))
    # get the data from the image
    data = image.get_fdata()
    # get the data from the mask
    mask_data, header = mask.get_fdata(), mask.header
    #flip mask_data along axial plane
    print(file, mask_data.shape)
    sys.stdout.flush()
    # mask_data = np.flip(mask_data, axis=2)
    # dilate the mask by 25mm according to in-plane resolution at every slice
    mask_data = binary_dilation(mask_data, iterations=10)
    masked_image = np.where(mask_data, data, -500)
    # multiply the image data by the mask data
    masked_image = nib.Nifti1Image(masked_image, affine=image.affine)
    nib.save(masked_image, os.path.join(dir_C, file))


dir_A = '/path/to/kcd_data/raw_data/final_pairedpretrain_nc/images'
dir_B = '/path/to/kcd_data/predictions_nii/final_pairedpretrain_ce/[2 2 2]mm'
dir_C = '/path/to/kcd_data/raw_data/masked_final_pairedpretrain_nc/images'

if not os.path.exists(dir_C):
    os.makedirs(dir_C)

# get all the files in directory A
files_A = [f for f in os.listdir(dir_A) if os.path.isfile(os.path.join(dir_A, f)) and f.endswith('.nii.gz')]
files_B = [f for f in os.listdir(dir_B) if os.path.isfile(os.path.join(dir_B, f)) and f.endswith('.nii.gz')]

# loop through all the files in directory A
for file in files_A:
    if file not in files_B:
        print(f"File {file} not in directory B")
        continue
    # load the image from directory A
    image = nib.load(os.path.join(dir_A, file))
    # load the mask from directory B
    mask = nib.load(os.path.join(dir_B, file))
    # get the data from the image
    data = image.get_fdata()
    # get the data from the mask
    mask_data, header = mask.get_fdata(), mask.header
    #flip mask_data along axial plane
    print(file, mask_data.shape)
    sys.stdout.flush()
    # mask_data = np.flip(mask_data, axis=2)
    # dilate the mask by 25mm according to in-plane resolution at every slice
    mask_data = binary_dilation(mask_data, iterations=10)
    masked_image = np.where(mask_data, data, -500)
    # multiply the image data by the mask data
    masked_image = nib.Nifti1Image(masked_image, affine=image.affine)
    nib.save(masked_image, os.path.join(dir_C, file))
