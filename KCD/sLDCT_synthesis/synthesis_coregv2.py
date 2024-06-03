from KCD.sLDCT_synthesis.jack_additive import calculate_noise
import os
import numpy as np
import nibabel as nib

def get_nii_image(image_path):
    '''
    :param image_path: str
    :return hu_array: arr
    '''
    image = nib.load(image_path).get_fdata()
    header = nib.load(image_path).header
    return image, header

def simulate_noise(image, dose_fraction=0.75, exposure_time=0.5, current = 400):
    '''
    :param image_path: str
    :param dose_fraction: float
    :param current: float
    :return: numpy array_like
    '''
    noise = np.zeros_like(image)
    for i in range(image.shape[0]):
        noise[i] = calculate_noise(np.squeeze(image[i]), current= current*exposure_time, dose_fraction=dose_fraction)
    return image + noise

def save_nii_image(image, header, save_path):
    '''
    :param image: numpy array_like
    :param header: nibabel.nifti1.Nifti1Header
    :param save_path: str
    :return: None
    '''
    nii_image = nib.Nifti1Image(image, None, header)
    nib.save(nii_image, save_path)

def main(raw_data_path, output_data_path):
    '''
    :param raw_data_path: str
    :return: None
    '''

    assert os.path.exists(raw_data_path), f'Path not found: {raw_data_path}'

    if not os.path.exists(output_data_path):
        os.makedirs(output_data_path)

    for file in [f for f in os.listdir(raw_data_path) if f.endswith('.nii.gz')]:
        image_path = os.path.join(raw_data_path, file)
        image, header = get_nii_image(image_path)
        noised_image = simulate_noise(image)
        save_nii_image(noised_image, header, os.path.join(output_data_path, file))

if __name__ == '__main__':
    raw_data_path = '/Users/mcgoug01/Downloads/process_coregv2/images'
    output_data_path = '/Users/mcgoug01/Downloads/process_coregv2/noised_images'
    main(raw_data_path, output_data_path)