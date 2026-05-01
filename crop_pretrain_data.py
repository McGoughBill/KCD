# open corresponding ce and nc files ('.nii.gz') from the two directories, and crop them to based
# on the number of voxels exceeding -200 HU in the z-axis. Save the cropped files to a new directory.
import os
import shutil
import nibabel as nib
import numpy as np

ce_dir = '/path/to/kcd_data/raw_data/paired_combined_cect/images'
nc_dir = '/path/to/kcd_data/raw_data/paired_combined_ncct/images'

ce_save_dir = '/path/to/kcd_data/raw_data/cropped_paired_combined_cect/images'
nc_save_dir = '/path/to/kcd_data/raw_data/cropped_paired_combined_ncct/images'

if not os.path.exists(ce_save_dir):
    os.makedirs(ce_save_dir)

if not os.path.exists(nc_save_dir):
    os.makedirs(nc_save_dir)

nc_files = [f for f in os.listdir(nc_dir) if f.endswith('.nii.gz')]

for nc_file in nc_files[::-1]:
    print(nc_file)
    nc_img = nib.load(os.path.join(nc_dir, nc_file))
    nc_data = nc_img.get_fdata()
    print(nc_data.shape)
    aff = nc_img.affine

    ce_file = nc_file
    ce_img = nib.load(os.path.join(ce_dir, ce_file))
    ce_data = ce_img.get_fdata()

    # crop the images based on the number of voxels exceeding -200 HU in the z-axis
    z_indices = np.where(np.sum(nc_data > -200, axis=(0, 1)) > 1000)[0]
    z_min = z_indices[0]
    z_max = z_indices[-1] + 1

    x_indices = np.where(np.sum(nc_data > -200, axis=(1, 2)) > 1000)[0]
    x_min = x_indices[0]
    x_max = x_indices[-1] + 1

    y_indices = np.where(np.sum(nc_data > -200, axis=(0, 2)) > 1000)[0]
    y_min = y_indices[0]
    y_max = y_indices[-1] + 1

    print(x_min, x_max, y_min, y_max, z_min, z_max)
    print()
    nc_data_cropped = nc_data[x_min:x_max, y_min:y_max, z_min:z_max]
    ce_data_cropped = ce_data[x_min:x_max, y_min:y_max, z_min:z_max]


    nc_img_cropped = nib.Nifti1Image(nc_data_cropped, aff)
    ce_img_cropped = nib.Nifti1Image(ce_data_cropped, aff)

    nib.save(nc_img_cropped, os.path.join(nc_save_dir, nc_file))
    nib.save(ce_img_cropped, os.path.join(ce_save_dir, ce_file))