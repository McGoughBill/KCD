# go through 'images' and 'labels' directories and list all files that exist in one
# directory but not in the other directory

import os
import shutil

images = '/home/wcm23/rds/rds-early_rcc_ldct-HSfXDTkHyC0/data/seg_data/raw_data/coreg_v4/images'
labels = '/home/wcm23/rds/rds-early_rcc_ldct-HSfXDTkHyC0/data/seg_data/raw_data/coreg_v4/labels'

# List of files in the source directory
images_files = [f for f in os.listdir(images) if os.path.isfile(os.path.join(images, f)) and f.endswith('.nii.gz')]
labels_files = [f for f in os.listdir(labels) if os.path.isfile(os.path.join(labels, f)) and f.endswith('.nii.gz')]


# List of files that exist in one directory but not in the other directory
images_not_in_labels = [f for f in images_files if f not in labels_files]
labels_not_in_images = [f for f in labels_files if f not in images_files]

print('Images not in labels:', images_not_in_labels)
print('Labels not in images:', labels_not_in_images)
