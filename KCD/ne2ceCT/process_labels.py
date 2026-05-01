import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel
from scipy.ndimage import binary_erosion, binary_dilation, binary_fill_holes, label, gaussian_filter


# want to clean up the labels generated through coregistration.
def remove_small_components(mask, min_size=100):
    # Label the connected components in the mask
    labeled_mask, num_labels = label(mask)
    # Iterate through each label
    for i in range(num_labels + 1):
        # If the size of the connected component is less than min_size, remove it
        if np.sum(labeled_mask == i) < min_size:
            mask[labeled_mask == i] = 0
    return mask
#to do this, we want to first erode the labels 5 times. Then, we want to clear any sections where the image
# is less than 0. Then we want to fill in any holes in the labels. Then we want to dilate the labels 3 times.

# image path and label path
images_fp = '/path/to/kcd_data/Downloads/process_coregv2/coregistered_images'
labels_fp = '/path/to/kcd_data/Downloads/process_coregv2/initial_labels'

save_labels_fp = '/path/to/kcd_data/Downloads/process_coregv2/cleaned_labels'
save_images_fp = '/path/to/kcd_data/Downloads/process_coregv2/cropped_images'
#all files are nifti
image_files = [file for file in os.listdir(images_fp) if file.endswith('.nii.gz')]

# load images and labels
for image_file in image_files:
    print(f'Processing {image_file}')
    # load image
    image_arr = nibabel.load(os.path.join(images_fp, image_file)).get_fdata()
    # load label
    og_label_arr = nibabel.load(os.path.join(labels_fp, image_file)).get_fdata()
    aff = nibabel.load(os.path.join(labels_fp, image_file)).affine

    label_arr = og_label_arr.copy()
    # clear any sections where the image is less than 0
    #pad image to same size as label
    pad = (label_arr.shape[-1] - image_arr.shape[-1])
    image_arr = np.pad(image_arr, ((0,0),(0,0),(0,pad)), mode='constant')
    pad = (label_arr.shape[0] - image_arr.shape[0])
    image_arr = np.pad(image_arr, ((0,pad),(0,0),(0,0)), mode='constant')
    #pad label to same size as image
    pad = (image_arr.shape[0] - label_arr.shape[0])
    label_arr = np.pad(label_arr, ((0,pad),(0,0),(0,0)), mode='constant')
    pad = (image_arr.shape[-1] - label_arr.shape[-1])
    label_arr = np.pad(label_arr, ((0,0),(0,0),(0,pad)), mode='constant')

    label_arr[image_arr < -50] = 0
    #remove small components
    new_label_arr = remove_small_components((label_arr>0).astype(np.uint8), min_size=1000)
    new_label_arr[(og_label_arr==2) & (new_label_arr==1)] = 2
    new_label_arr[(og_label_arr==3) & (new_label_arr==1)] = 3
    label_arr = new_label_arr
    print(f'Unique values in label: {np.unique(label_arr)}')
    new_label_arr = np.zeros_like(label_arr)

    #fill in any holes in the labels - for each class
    for i in range(1,4):
        if i == 1:
            new_label_arr[binary_fill_holes(label_arr>0)] = i
        else:
            new_label_arr[binary_fill_holes(label_arr==i)] = i

    label_arr = new_label_arr

    if image_arr.shape[0] != 512:

        # crop axially to the foreground, i.e. when atleast one pixel in an axial slice is above zero
        # get the indices of the axial slices that have atleast 100 pixel above zero
        axial_slices = np.sum(image_arr > 0, axis=(0, 1)) > 100
        # get the min and max indices of the axial slices
        min_axial_slice, max_axial_slice = np.where(axial_slices)[0][[0, -1]]
        # crop the image and label
        image_arr = image_arr[:, :, min_axial_slice:max_axial_slice + 1]
        label_arr = label_arr[:, :, min_axial_slice:max_axial_slice + 1]
    if image_arr.shape[1] != 512:
        # crop coronally to the foreground, i.e. when atleast one pixel in a coronal slice is above zero
        # get the indices of the coronal slices that have atleast 100 pixel above zero
        coronal_slices = np.sum(image_arr > 0, axis=(0, 2)) > 100
        # get the min and max indices of the coronal slices
        min_coronal_slice, max_coronal_slice = np.where(coronal_slices)[0][[0, -1]]
        # crop the image and label
        image_arr = image_arr[:, min_coronal_slice:max_coronal_slice + 1, :]
        label_arr = label_arr[:, min_coronal_slice:max_coronal_slice + 1, :]
    if image_arr.shape[2] != 512:
        # crop sagitally to the foreground, i.e. when atleast one pixel in a sagittal slice is above zero
        # get the indices of the sagittal slices that have atleast 100 pixel above zero
        sagittal_slices = np.sum(image_arr > 0, axis=(1, 2)) > 100
        # get the min and max indices of the sagittal slices
        min_sagittal_slice, max_sagittal_slice = np.where(sagittal_slices)[0][[0, -1]]
        # crop the image and label
        image_arr = image_arr[min_sagittal_slice:max_sagittal_slice + 1, :, :]
        label_arr = label_arr[min_sagittal_slice:max_sagittal_slice + 1, :, :]

    assert image_arr.shape == label_arr.shape, 'Image and label shapes are not the same'

    print(f'Shape of image: {image_arr.shape}')
    print(f'Unique values in label: {np.unique(label_arr)}')
    print('Saving image and label')

    # save the label
    nibabel.save(nibabel.Nifti1Image(label_arr.astype(np.uint8), aff), os.path.join(save_labels_fp, image_file),dtype=np.uint8)
    # save the image
    nibabel.save(nibabel.Nifti1Image(image_arr, aff), os.path.join(save_images_fp, image_file))