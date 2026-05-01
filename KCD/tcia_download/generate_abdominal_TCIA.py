import os
import numpy as np
import nibabel as nib

# define the path to the registered images
registered_ncct_path = '/path/to/kcd_data/data/seg_data/raw_data/tcia_ncct_reg/images'
registered_cect_path = '/path/to/kcd_data/data/seg_data/raw_data/tcia_cect/images'

# coregistered images have identical names, so we can use the same list for both
registered_ncct_ims = [f for f in os.listdir(registered_ncct_path) if f.endswith('.nii.gz')]

# define the path where cropped and registered images were saved
cropped_ncct_path = '/path/to/kcd_data/data/seg_data/raw_data/cropped_tcia_ncct/images'
cropped_cect_path = '/path/to/kcd_data/data/seg_data/raw_data/cropped_tcia_cect/images'

# load the cropped images from ncct
cropped_ncct_ims = [f for f in os.listdir(cropped_ncct_path) if f.endswith('.nii.gz')]

# filter registered_ncct_ims to only include images that are in cropped_ncct_ims
registered_ncct_ims = [im for im in registered_ncct_ims if im in cropped_ncct_ims]

#define path where masked ncct images are saved
masked_ncct_path = '/path/to/kcd_data/data/seg_data/raw_data/masked_tcia_ncct/images'

# define the path where to save abdominal coregistered images
abdominal_ncct_path = '/path/to/kcd_data/data/seg_data/raw_data/abdominal_tcia_ncct/images'
abdominal_cect_path = '/path/to/kcd_data/data/seg_data/raw_data/abdominal_tcia_cect/images'


# create the directories if they don't exist
if not os.path.exists(abdominal_ncct_path):
    os.makedirs(abdominal_ncct_path)
if not os.path.exists(abdominal_cect_path):
    os.makedirs(abdominal_cect_path)

# loop through the registered images
for im in registered_ncct_ims:
    # load the image
    ncct_im = nib.load(os.path.join(registered_ncct_path, im))
    cect_im = nib.load(os.path.join(registered_cect_path, im))
    masked_ncct_im = nib.load(os.path.join(masked_ncct_path, im))

    # get the image data
    ncct_data = ncct_im.get_fdata()
    cect_data = cect_im.get_fdata()
    masked_ncct_data = np.flip(np.flip(np.rot90(masked_ncct_im.get_fdata(),3),0),2)

    assert ncct_data.shape == cect_data.shape == masked_ncct_data.shape

    # crop the first and last two slices
    ncct_data = ncct_data[:,:,2:-2]
    cect_data = cect_data[:,:,2:-2]
    masked_ncct_data = masked_ncct_data[:,:,2:-2]

    # find the slices that are entirely -500 HU
    ncct_mask = np.all(masked_ncct_data == -500, axis=(0,1))

    # crop the slices that are entirely -500 HU
    ncct_data = ncct_data[:,:,~ncct_mask]
    cect_data = cect_data[:,:,~ncct_mask]

    # save the cropped images
    ncct_cropped = nib.Nifti1Image(ncct_data, ncct_im.affine)
    cect_cropped = nib.Nifti1Image(cect_data, cect_im.affine)

    nib.save(ncct_cropped, os.path.join(abdominal_ncct_path, im))
    nib.save(cect_cropped, os.path.join(abdominal_cect_path, im))