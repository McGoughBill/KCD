# we wish to pull images from four different folders together.
# all images will be copied into a central folder. first, though, the exact filepaths of the images will be stored in a list,
# and a new study ID will be assigned to each image. the study ID will be randomised for each image - this will be done by
# shuffling the list of filepaths, and using the index of each filepath in the shuffled list as the study ID.
# the study ID will be stored in a metadata csv file, along with the original filepath of the image, at the level of the
# central folder. all images are .nii.gz

#also, all will have all air regions cropped and be clipped to [-300,300] HU

import os
import pandas as pd
import random
import nibabel as nib
import numpy as np
import sys

#set random seeds
random.seed(42)
np.random.seed(42)

nc_path = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/raw_data/sldct_vs_ldct_test/nc/images/'
eighty_pc_path = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/raw_data/sldct_vs_ldct_test/80pc_sldct/images/'
ninety_pc_path = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/raw_data/sldct_vs_ldct_test/90pc_sldct/images/'
LDCT_path = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/raw_data/sldct_vs_ldct_test/ldct/images/'

central_folder = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/raw_data/sldct_vs_ldct_test/final_quality_ds/'
metadata = []
#first, get all the filepaths
nc_files = [os.path.join(nc_path,f) for f in os.listdir(nc_path) if f.endswith('.nii.gz')]
eighty_pc_files = [os.path.join(eighty_pc_path,f) for f in os.listdir(eighty_pc_path) if f.endswith('.nii.gz')]
ninety_pc_files = [os.path.join(ninety_pc_path,f) for f in os.listdir(ninety_pc_path) if f.endswith('.nii.gz')]
LDCT_files = [os.path.join(LDCT_path,f) for f in os.listdir(LDCT_path) if f.endswith('.nii.gz')]
all_files = nc_files + eighty_pc_files + ninety_pc_files + LDCT_files

print(f'Found {len(all_files)} files')
sys.stdout.flush()

#shuffle the list of filepaths
random.shuffle(all_files)

for index,f in enumerate(all_files):
    print(f'Processing {f}...\n')
    sys.stdout.flush()
    image = nib.load(f).get_fdata()
    #crop air regions in all three dimensions. air region defined as ratio of slice < -800 HU is over 0.9
    #then, crop the region without all zeros out via indexing

    #first axis
    #find the first and last non-zero indices
    first_nonzero = 0
    for i in range(image.shape[0]):
        slice = image[i]
        ratio = np.sum(slice<-800)/slice.size
        if ratio < 0.9:
            first_nonzero = i
            break
    highest_nonzero = image.shape[0]
    for i in range(first_nonzero,image.shape[0]):
        slice = image[i]
        ratio = np.sum(slice<-800)/slice.size
        if ratio < 0.9:
            highest_nonzero = i
    image = image[first_nonzero:highest_nonzero]

    #second axis
    first_nonzero = 0
    for i in range(image.shape[1]):
        slice = image[:,i]
        ratio = np.sum(slice<-800)/slice.size
        if ratio < 0.9:
            first_nonzero = i
            break
    highest_nonzero = image.shape[1]
    for i in range(first_nonzero,image.shape[1]):
        slice = image[:,i]
        ratio = np.sum(slice<-800)/slice.size
        if ratio < 0.9:
            highest_nonzero = i
    image = image[:,first_nonzero:highest_nonzero]

    #third axis
    first_nonzero = 0
    for i in range(image.shape[2]):
        slice = image[:,:,i]
        ratio = np.sum(slice<-800)/slice.size
        if ratio < 0.9:
            first_nonzero = i
            break
    highest_nonzero = image.shape[2]
    for i in range(first_nonzero,image.shape[2]):
        slice = image[:,:,i]
        ratio = np.sum(slice<-800)/slice.size
        if ratio < 0.9:
            highest_nonzero = i
    image = image[:,:,first_nonzero:highest_nonzero]



    #clip to [-300,300] HU
    image[image<-200] = -200
    image[image>200] = 200
    image = image.astype(np.float32)
    aff = nib.load(f).affine
    save_path = os.path.join(central_folder,'images',f'QUALITY{index}.nii.gz')
    nii_image = nib.Nifti1Image(image, affine=aff)
    nib.save(nii_image, save_path)

    metadata.append({'study_id':index,'original_filepath':f})

metadata_df = pd.DataFrame(metadata)
metadata_df.to_csv(os.path.join(central_folder,'metadata.csv'),index=False)
