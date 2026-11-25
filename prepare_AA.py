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

path_to_AbdomenAtlas = '/path/to/kcd_data/Downloads/AbdomenAtlas/AbdomenAtlasMini1.0'
save_path = '/path/to/kcd_data/Downloads/AbdomenAtlas/nifti'
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
    if patient == '.DS_Store': continue
    mask_save = os.path.join(label_path, patient + '.nii.gz')
    print(patient)


    # create a mask


    organ_list = [os.path.join(path_to_AbdomenAtlas, patient, 'segmentations',organ) for organ in os.listdir(os.path.join(path_to_AbdomenAtlas, patient, 'segmentations')) if organ != '.DS_Store']
    starter = nib.load(organ_list[0])
    im_affine = starter.affine
    im_header = starter.header
    starter_data = starter.get_fdata().astype(np.int16)

    for organ in organ_list[1:]:
        organ_name = organ.split('/')[-1].split('.')[0]
        organ_index = organ_dict[organ_name]
        organ_mask = nib.load(organ)
        organ_mask_data = organ_mask.get_fdata()
        starter_data[organ_mask_data >0] = organ_index

    # save the mask
    mask_nifti = nib.Nifti1Image(starter_data, im_affine, im_header)
    nib.save(mask_nifti, os.path.join(label_path, patient + '.nii.gz'))
