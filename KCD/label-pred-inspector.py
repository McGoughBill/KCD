import nibabel as nib
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

def get_nifti_data(path):
    img = nib.load(path)
    return img.get_fdata()


nii_labpath = '/Users/mcgoug01/Downloads/untitled folder/inferences/'
nii_impath = '/Users/mcgoug01/Downloads/untitled folder/images/'
nii_files = os.listdir(nii_impath)
for file in nii_files:
    print(file)
    if not os.path.exists(nii_labpath+file):
        continue

    im = np.rot90(get_nifti_data(nii_impath+file), k=1, axes=(0, 1))
    pred = np.rot90(get_nifti_data(nii_labpath+file), k=1, axes=(0, 1))
    pred=np.flip(pred, axis=-1)
    index = pred.sum(axis=(0, 1)).argmax()

    fig = plt.figure(figsize=(10,5))
    fig.suptitle(file[:-4])
    plt.subplot(1,2,1)
    plt.title('Unseen CT Scan')
    plt.imshow(im[:,:,index],cmap='gray',vmin=-200,vmax=200)
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.title('Finding the Cancer')
    plt.imshow(im[:,:,index],cmap='gray',vmin=-200,vmax=200)
    plt.imshow(pred[:,:,index],cmap='gray',alpha=0.5)
    plt.axis('off')
    plt.show(block=True)
