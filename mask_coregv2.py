import nibabel as nib
import numpy as np
import os
from scipy.ndimage import binary_dilation
import sys
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
plt.switch_backend('TkAgg')
#take images 'nii.gz' from dir_A, mask with images from dir_B and save in dir_C.
# dilate mask by 25mm, using the header information from the original image

# directory A
# dir_A = '/rds/project/rds-sDuALntK11g/raw_data/nc_reg/images'
# dir_B = '/rds/project/rds-sDuALntK11g/predictions_nii/ce/[2 2 2]mm'
# dir_C = '/rds/project/rds-sDuALntK11g/raw_data/masked_nc_reg/images'

dir_A = '/Users/mcgoug01/Downloads/masking/images'
dir_B = '/Users/mcgoug01/Downloads/masking/cect_[4 4 4]mm'
dir_C = '/Users/mcgoug01/Downloads/masking/masked_images'

if not os.path.exists(dir_C):
    os.makedirs(dir_C)

# get all the files in directory A
files_A = [f for f in os.listdir(dir_A) if os.path.isfile(os.path.join(dir_A, f)) and f.endswith('.nii.gz')]

for file in files_A:
    if os.path.exists(os.path.join(dir_C, file)):
        continue
    # load the image
    image = nib.load(os.path.join(dir_A, file)).get_fdata()
    # load the mask
    mask = nib.load(os.path.join(dir_B, file.replace('_noised_0.7','_noised_0'))).get_fdata()
    shape = mask.shape
    header = nib.load(os.path.join(dir_A, file)).header
    affine = nib.load(os.path.join(dir_A, file)).affine

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

    #plot the image and the mask
    # slice is the index of the axial slice with largest number of non-zero pixels

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    if file.startswith('KiTS'):
        slice = np.argmax(np.sum(mask, axis=(1,2)))
        ax.imshow(image[slice], cmap='gray',vmin= -200, vmax=200)
        ax.imshow(mask[slice], cmap='jet', alpha=0.5)
    else:
        slice = np.argmax(np.sum(mask, axis=(0, 1)))
        ax.imshow(image[:,:,slice], cmap='gray',vmin= -200, vmax=200)
        ax.imshow(mask[:,:,slice], cmap='jet', alpha=0.5)

    ax.set_title('{} image'.format(file))
    fig.canvas.manager.window.wm_geometry("+200+500")
    #set plot location on monitor
    plt.show(block=True)


    # save the masked image
    masked_image_nii = nib.Nifti1Image(masked_image, affine)
    nib.save(masked_image_nii, os.path.join(dir_C, file))
