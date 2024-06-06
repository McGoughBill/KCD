#go through nifti images in location A, and calculate the spatial size of each of the largest cancers (label 2)
# will need to seperate each connected component, and calculate the size of each of them
# the size is calculated in mm^3
# the output is a csv file with the following columns: image_name, size

import os
import numpy as np
import nibabel as nib
import pandas as pd
from skimage import measure

#path to the nifti images
path = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/raw_data/coreg_ncct_v2/labels'
src_path = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/raw_data/coreg_ncct_v2/'
large_path = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/raw_data/large_coreg_ncct_v2/'
small_path = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/raw_data/small_coreg_ncct_v2/'


non_con_path = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/raw_data/sldct_vs_ldct_test/nc'
high_ldct_path = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/raw_data/sldct_vs_ldct_test/high_sldct'
low_ldct_path = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/raw_data/sldct_vs_ldct_test/low_sldct'


#copy images from the following folders to a small_path+folder if the size of the cancer is less than 20mm^3
# and copy to a large_path+folder if the size of the cancer is greater than 20mm^3
folders = ['images','labels','noised_ims_0.9','noised_ims_0.8','noised_ims_0.7','noised_ims_0.6']

for folder in folders:
    assert os.path.exists(os.path.join(src_path, folder)), 'Folder does not exist: ' + os.path.join(path, folder)
    if not os.path.exists(os.path.join(large_path, folder)):
        os.makedirs(os.path.join(large_path, folder))
    if not os.path.exists(os.path.join(small_path, folder)):
        os.makedirs(os.path.join(small_path, folder))

for dir in [non_con_path, high_ldct_path, low_ldct_path]:
    for folder in folders:
        assert os.path.exists(dir), 'Folder does not exist: ' + os.path.join(dir)
        if not os.path.exists(os.path.join(dir, folder)):
            os.makedirs(os.path.join(dir, folder))




#list of all the nifti images
images = [f for f in os.listdir(path) if f.endswith('.nii.gz')]

#list to store the size of each cancer
sizes = []
small_boundary = (4/3) * np.pi * (20**3)

for image in images:
    #load the image
    img = nib.load(os.path.join(path, image))
    img_data = img.get_fdata()

    #get the largest connected component
    labels = measure.label(img_data == 2)
    props = measure.regionprops(labels)
    props.sort(key=lambda x: x.area, reverse=True)
    largest_cc = labels == props[0].label

    #calculate the size of the largest connected component
    size = np.sum(largest_cc)
    voxel_size = np.prod(img.header.get_zooms())
    size_mm3 = size * voxel_size
    is_small = size_mm3 < small_boundary

    if is_small:
        for folder in folders:
            os.system('cp ' + os.path.join(src_path, folder, image) + ' ' + os.path.join(small_path, folder, image))
    else:
        for folder in folders:
            os.system('cp ' + os.path.join(src_path, folder, image) + ' ' + os.path.join(large_path, folder, image))

    print('Image: ', image)
    print('Size: ', size_mm3)
    print('Small: ', is_small)
    print()

    #store the size
    sizes.append({'image_name': image, 'size': size_mm3,'is_small': is_small})

#save the sizes to a csv file
df = pd.DataFrame(sizes)
#sort the dataframe rows by the size of the cancer
df = df.sort_values(by='size')
#assign alternate A and B to the is_small column to a new column is is_small is True
df['Group'] = df['is_small'].cumsum() % 3
df.to_csv(os.path.join(src_path, 'cancer_sizes.csv'), index=False)

# for each image, if is_small is true assign '0' in Group to non_con_path, '1' to high_ldct_path, and '2' to low_ldct_path,
# and copy the image to all corresponding folders

for index, row in df.iterrows():
    image = row['image_name']
    is_small = row['is_small']
    group = row['Group']
    if is_small:
        if group == 0:
            for folder in folders:
                os.system('cp ' + os.path.join(src_path, folder, image) + ' ' + os.path.join(non_con_path, folder, image))
        elif group == 1:
            for folder in folders:
                os.system('cp ' + os.path.join(src_path, folder, image) + ' ' + os.path.join(high_ldct_path, folder, image))
        else:
            for folder in folders:
                os.system('cp ' + os.path.join(src_path, folder, image) + ' ' + os.path.join(low_ldct_path, folder, image))



