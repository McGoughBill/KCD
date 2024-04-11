import nibabel as nib
import numpy as np
import os

#this script is used for preparing data from the AbdomenAtlas dataset
#it reads in the images and masks, and saves them as nifti files.
# the file structure is as follows:

# path/to/AbdomenAtlas
#     -<patient_id1>
#         -ct.nii.gz
#         -segmentations
#             -aorta.nii.gz
#             -liver.nii.gz
#             -spleen.nii.gz
#             -kidney_left.nii.gz
#             -kidney_right.nii.gz
#             -pancreas.nii.gz
#             -stomach.nii.gz
#             -gall_bladder.nii.gz
#             -postcava.nii.gz
#     -<patient_id2>
#         -ct.nii.gz
#         ....

# the script creates a new directory, 'nifti', and saves the images and masks in this directory
# the images are saved as 'patient_id.nii.gz' in the /nifti/images directory
# the segmentation masks are all collated into one mask, indexed by the organ, and saved as 'patient_id.nii.gz' in the /nifti/labels directory

# the script also creates a csv file, 'nifti/labels.csv', which contains the mapping of organ to index in the mask

path_to_AbdomenAtlas = '/Users/mcgoug01/Downloads/AbdomenAtlas/AbdomenAtlasMini1.0'
save_path = '/Users/mcgoug01/Downloads/AbdomenAtlas/nifti'
im_path = os.path.join(save_path, 'images')
label_path = os.path.join(save_path, 'labels')

if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(im_path):
    os.makedirs(im_path)
if not os.path.exists(label_path):
    os.makedirs(label_path)

# create a dictionary to map organ to index
organ_dict = {'aorta': 1, 'liver': 2, 'spleen': 3, 'kidney_left': 4, 'kidney_right': 5, 'pancreas': 6, 'stomach': 7, 'gall_bladder': 8, 'postcava': 9}

# create a csv file to store the mapping
import csv
with open(os.path.join(save_path, 'labels.csv'), 'w') as f:
    writer = csv.writer(f)
    for key, value in organ_dict.items():
        writer.writerow([key, value])

# loop through the patients
for patient in os.listdir(path_to_AbdomenAtlas):
    # load the image
    im = nib.load(os.path.join(path_to_AbdomenAtlas, patient, 'ct.nii.gz'))
    im_data = im.get_fdata()
    im_affine = im.affine
    im_header = im.header
    mask_save = os.path.join(label_path, patient + '.nii.gz')

    if os.path.exists(mask_save):
        continue

    # save the image
    im_nifti = nib.Nifti1Image(im_data, im_affine, im_header)
    nib.save(im_nifti, os.path.join(im_path, patient + '.nii.gz'))

    # create a mask
    mask = np.zeros(im_data.shape)
    for organ in os.listdir(os.path.join(path_to_AbdomenAtlas, patient, 'segmentations')):
        organ_name = organ.split('.')[0]
        organ_index = organ_dict[organ_name]
        organ_mask = nib.load(os.path.join(path_to_AbdomenAtlas, patient, 'segmentations', organ))
        organ_mask_data = organ_mask.get_fdata()
        mask[organ_mask_data == 1] = organ_index

    # save the mask
    mask_nifti = nib.Nifti1Image(mask, im_affine, im_header)
    nib.save(mask_nifti, os.path.join(label_path, patient + '.nii.gz'))
