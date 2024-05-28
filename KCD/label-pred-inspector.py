import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
plt.switch_backend('TkAgg')

def get_nifti_data(path):
    return nib.load(path).get_fdata()


npy_infpath ='/Users/mcgoug01/Downloads/tcia_cect_segs/new_segs/'
npy_cectpath = '/Users/mcgoug01/Downloads/tcia_cect_segs/cect/'
npy_ncctpath = '/Users/mcgoug01/Downloads/tcia_cect_segs/images/'
# nii_impath = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/raw_data/all_add/images/'
# nii_labpath = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/raw_data/all_add/manual_labels/'

# save_impath = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/raw_data/all_add/new_images/'
save_labpath = '/Users/mcgoug01/Downloads/tcia_cect_segs/new_segs/'
npy_files = [file for file in os.listdir(npy_ncctpath) if file.endswith('.npy')]

for file in npy_files:
    # print(file)
    # if not os.path.exists(nii_labpath+file):
    #     continue

    # im = np.flip(np.rot90(get_nifti_data(nii_impath+file), k=3, axes=(0, 1)), axis=1)
    # canc = np.flip(np.rot90(get_nifti_data(nii_labpath+file)==1, k=3, axes=(0, 1)), axis=1)
    pred_fp = npy_infpath+file
    # kid = np.flip(np.flip(np.load(pred_fp).swapaxes(0,2), axis=2),axis=0)
    kid = np.load(pred_fp)
    cect = np.load(npy_cectpath+file)[0]
    ncct = np.load(npy_ncctpath+file)[0]

    slice = np.argmax(np.sum(kid, axis=(1,2)))
    #2x2 grid plt
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(cect[slice])
    plt.title('CECT')
    plt.subplot(2,2,2)
    plt.imshow(ncct[slice])
    plt.title('NCCT')
    plt.subplot(2,2,3)
    plt.imshow(cect[slice])
    plt.imshow(kid[slice], alpha=0.5)
    plt.title('Prediction CECT')
    plt.subplot(2,2,4)
    plt.imshow(ncct[slice])
    plt.imshow(kid[slice], alpha=0.5)
    plt.title('Prediction NCCT')
    plt.show(block=True)

    #put canc and kid 3 segs into one. wherever canc==1, put 2. canc will overlap with kid
    # so we will have 0 for background, 1 for kidney, 2 for cancer
    # canc[canc==1] = 2
    # seg = np.zeros_like(canc)
    # seg[kid==1] = 1
    # seg[canc==2] = 2

    # nib_im = nib.Nifti1Image(im.astype(np.float32), aff)
    # nib_seg = nib.Nifti1Image(kid.astype(np.uint8), aff)

    #correct header dtypes and as_closest_canonical
    # nib_im.header.set_data_dtype(np.float32)
    # nib_seg.header.set_data_dtype(np.uint8)
    # nib_im = nib.as_closest_canonical(nib_im)
    # nib_seg = nib.as_closest_canonical(nib_seg)


    # nib.save(nib_im, save_impath+file)
    # nib.save(nib_seg, save_labpath+file)

    # np.save(save_labpath+file, kid)