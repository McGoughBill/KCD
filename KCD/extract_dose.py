# open mask image '.nii.gz' file, find all axial slices that contain voxel > 0 (axial is third dimension)
# and then use this to index dicom files in another sub directory. From these indexed dicom files, create a new
# nifti file, and find the average dose across these slices.

import os
import numpy as np
import nibabel as nib
import pydicom

mask_dir = '/Users/mcgoug01/Downloads/test_results/mask_flip'
dicom_dir = '/Users/mcgoug01/Downloads/test_results/dicoms'
output_dir = '/Users/mcgoug01/Downloads/test_results/output'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

results = []
exposure_time = []
xray_tube_current = []

for patient_file in [file for file in os.listdir(mask_dir) if file.endswith('.nii.gz')]:
    patient = patient_file.split('.')[0]
    print(patient)
    mask_og = nib.load(os.path.join(mask_dir, patient_file)).get_fdata()

    # find all axial slices that contain voxel
    # slices refer to the third dimension
    mask = np.where(mask_og > 0)
    mask = np.unique(mask[2])
    mask = mask*-1 + mask_og.shape[2]


    dcm_dir_p = os.path.join(dicom_dir, patient)
    dicom_dir_p = os.path.join(dcm_dir_p, [file for file in os.listdir(dcm_dir_p) if file !='.DS_Store'][0])
    dicom_files = [os.path.join(dicom_dir_p, file) for file in os.listdir(dicom_dir_p) if file.endswith('.dcm')]
    dicom_files = [pydicom.dcmread(file) for file in dicom_files]
    dicom_files = [file for file in dicom_files if file.InstanceNumber in mask]

    dicom_files = sorted(dicom_files, key=lambda x: x.InstanceNumber)

    # for each file, find slice dose by multiplying Exposure Time and X-Ray Tube Current. average this across all slices
    # to get the average dose

    try :
        average_exposure_time = np.mean([file.ExposureTime for file in dicom_files])/1000
    except AttributeError:
        average_exposure_time = np.nan

    try:
        average_xray_tube_current = np.mean([file.XRayTubeCurrent for file in dicom_files])
    except AttributeError:
        average_xray_tube_current = np.nan

    results.append({'patient': patient, 'average_exposure_time': average_exposure_time, 'average_xray_tube_current': average_xray_tube_current})

    print('Average Exposure Time: ', average_exposure_time)
    print('Average X-Ray Tube Current: ', average_xray_tube_current)

    #create new nifti file
    data = np.stack([file.pixel_array for file in dicom_files], axis=2)
    affine = np.eye(4)
    nifti = nib.Nifti1Image(data, affine)
    nib.save(nifti, os.path.join(output_dir, patient + '.nii.gz'))

print(results)
import pandas
df = pandas.DataFrame(results)

# where any of the tube currents are nan, fill in values by:
# 1. finding the exposure time for the same patient
# 2. finding the all other patients with an exposure time within 10% of this value
# 3. finding the average xray tube current for these patients
# 4. filling in the nan values with this average

for patient in df['patient']:
    if np.isnan(df.loc[df['patient'] == patient, 'average_xray_tube_current'].values[0]):
        exposure_time = df.loc[df['patient'] == patient, 'average_exposure_time'].values[0]
        patients = df.loc[(df['average_exposure_time'] > exposure_time*0.9) & (df['average_exposure_time'] < exposure_time*1.1)]
        xray_tube_current = np.mean(patients['average_xray_tube_current'])
        df.loc[df['patient'] == patient, 'average_xray_tube_current'] = xray_tube_current

# do the same for exposure time

for patient in df['patient']:
    if np.isnan(df.loc[df['patient'] == patient, 'average_exposure_time'].values[0]):
        xray_tube_current = df.loc[df['patient'] == patient, 'average_xray_tube_current'].values[0]
        patients = df.loc[(df['average_xray_tube_current'] > xray_tube_current*0.9) & (df['average_xray_tube_current'] < xray_tube_current*1.1)]
        exposure_time = np.mean(patients['average_exposure_time'])
        df.loc[df['patient'] == patient, 'average_exposure_time'] = exposure_time

df['dose'] = df['average_exposure_time'] * df['average_xray_tube_current']
df.to_csv(os.path.join(output_dir, 'results.csv'))


