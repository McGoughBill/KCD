import numpy as np
import pandas as pd
import os
import scipy.ndimage as ndimage
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')

label_loc = '/path/to/kcd_data/Cambridge University Dropbox/William McGough/AIvsRad/Bill/FD/internaltest_metadata.csv'
labels = pd.read_csv(label_loc)
print(labels)
import nibabel as nib
seg_loc = '/path/to/kcd_data/raw_data/coreg_v4/labels'
kits_seg_loc = '/path/to/kcd_data/raw_data/test_set_v2/labels'
results = []
for key in labels.OriginalID.values:
    #extract studyID from this
    studyID = labels[labels['OriginalID'] == key].StudyID.values[0]
    # find case name fr0m labels
    entry = {'case':key}
    # if case.lower does not begin with kits or rcc, skip
    if not key.lower().startswith('kits') and not key.lower().startswith('rcc'):
        continue

    # if case begins with kits in seg_loc
    try:
        if key.lower().startswith('kits'):
            case = key.lower().replace('kits-','KiTS-')
            seg_n = nib.load(os.path.join(kits_seg_loc, case))
        # if case begins with rcc in seg_loc
        elif key.lower().startswith('rcc'):
            case = key.lower().replace('rcc_','Rcc_')
            seg_n = nib.load(os.path.join(seg_loc, case))
        else:
            continue
    except:
        print('Error with case:',key)
        continue
    combined = seg_n.get_fdata().astype(np.uint8)
    vox_vol = np.prod(seg_n.header.get_zooms())
    entry['size'] = np.sum(combined == 2) * vox_vol

    #resize combined to 1x1x1 vozel size using torch.nn.functional.interpolate and existing voxel size
    combined = torch.tensor(combined).unsqueeze(0).unsqueeze(0).float()
    target_shape = (int(combined.shape[2] *seg_n.header['pixdim'][1]),int(combined.shape[3] *seg_n.header['pixdim'][2]),int(combined.shape[4] *seg_n.header['pixdim'][3]))
    combined = F.interpolate(combined, size=target_shape, mode='nearest').squeeze().int().numpy()
    print(combined.shape)
    # print(np.unique(combined,return_counts=True))
    #find the largest cancer region and extract its segmentation
    labels_indices, n = ndimage.label(combined.copy() == 2)
    sizes = ndimage.sum(combined.copy()==2, labels_indices, range(n+1))
    max_label = np.argmax(sizes)
    # print(labels_indices.shape, n, sizes,max_label)
    largest_cancer = np.zeros(combined.shape, dtype=np.uint8)
    largest_cancer[labels_indices != max_label] = 0
    largest_cancer[labels_indices == max_label] = 1
    # extract volume, surface area, sphericity, largest diameter, and exophytic fraction
    volume = np.sum(largest_cancer)
    # print(volume)



    contour = ndimage.binary_dilation(largest_cancer.copy()>0, iterations=1).astype(np.uint8) - largest_cancer.copy()

    # # find max slice for contour
    # max_slice = np.argmax(np.sum(largest_cancer.copy(),axis=(0,1)))
    # fig = plt.figure()
    # # print 2d slice of contour, dilated contour, and kidney
    # plt.subplot(131)
    # plt.imshow(contour[:,:,max_slice])
    # plt.subplot(132)
    # plt.imshow(largest_cancer[:,:,max_slice])
    # plt.subplot(133)
    # plt.imshow(combined[:,:,max_slice])


    surface_area = np.sum(contour)
    # scale surface area to mm^2 and then to cm^2
    sphericity = (np.pi**(1/3))*((6*volume)**(2/3))/surface_area

    # find the exophytic fraction
    overlap_of_contour = np.sum(contour[combined >0]) #find overlap of dilated contour with kidney
    entry['exo_frac'] = (surface_area-overlap_of_contour)/surface_area
    print(entry)
    print()
    # plt.show(block=True)
    results.append(entry)

# combine results into dataframe
df = pd.DataFrame(results)

#save to same loc using name 'internal_exo_size.csv'
df.to_csv('/path/to/kcd_data/Cambridge University Dropbox/William McGough/AIvsRad/Bill/FD/internal_exo_size.csv')
print(df)
df['exo_b'] = df['exo_frac']>=0.5
df['meso_b'] =(df['exo_frac']<0.5) & (df['exo_frac']>0.2)
df['endo_b'] = df['exo_frac']<=0.2
df['large_b'] = df['size']>(4/3)*np.pi*(15**3)
df['small_b'] = df['size']<(4/3)*np.pi*(10**3)

# find numbers of exo,meso, and endo in small and large using groupby
df.groupby(['exo_b','large_b']).size()
df.groupby(['meso_b','large_b']).size()
print(df.groupby(['large_b','endo_b']).size())
print(df.groupby(['large_b','meso_b']).size())
print(df.groupby(['large_b','exo_b']).size())

