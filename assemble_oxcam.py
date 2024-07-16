# source dir contains a set of sub_dir, name after patient ID.
# in each subdir, there is a file 'ce.nii.gz'. Copy the file to destination, renaming it to the patient ID.

import os

source_dir = '/rds/project/rds-sDuALntK11g/oxcam_nifti'
dest_dir = '/rds/project/rds-sDuALntK11g/raw_data/oxcam/images'

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file == 'ce.nii.gz':
            source_file = os.path.join(root, file)
            dest_file = os.path.join(dest_dir, os.path.basename(root)+'.nii.gz')
            os.rename(source_file, dest_file)
            print('Copied:', dest_file)