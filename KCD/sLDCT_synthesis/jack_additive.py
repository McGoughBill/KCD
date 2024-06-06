# Jack Dixon 24/01/2024
# Method HEAVILY influenced by my supervisor's scripts for the same purpose, William McGough. Find his GitHub here:
# https://github.com/mcgoughlin
'''
Reference for noise insertion technique:
Yu, Lifeng PhD; Shiung, Maria BA; Jondal, Dayna BA; McCollough, Cynthia H. PhD.
Development and Validation of a Practical Lower-Dose-Simulation Tool for Optimizing Computed Tomography Scan Protocols.
Journal of Computer Assisted Tomography: July/August 2012 - Volume 36 - Issue 4 - p 477-487
doi: 10.1097/RCT.0b013e318258e891
'''
import matplotlib.pyplot as plt
import numpy as np
import pydicom.pixel_data_handlers.util
import pydicom as dicom
import tomosipo as ts
from ts_algorithms import fbp
import torch
import warnings
from gecatsim.pyfiles.CommonTools import *
import dicom2nifti as d2n
import nibabel as nib
warnings.filterwarnings("ignore")


def from_HU_to_normal(image):
    '''
    :param image: numpy array_like
    :return: numpy array_like
    '''
    return np.maximum((image+1000)/3000,0)


def from_normal_to_HU(image):
    '''
    :param image: numpy array_like
    :return: numpy array_like
    '''
    return np.maximum((image*3000)-1000,-1000)


# Define a function for the estimate of incident photon number from 'Air Scan' data:
def XCIST_incident_photon_number(current):
    '''
    :param current: float
    :return: float
    '''
    # Calculates N_0, The below loads from the pickled fit values:
    # m, c = np.load(r'C:\Users\jackb\PycharmProjects\masters_project\XCIST_Air_Gap\N_0a_fit_from_XCIST.npy')[0:2]
    # These are just the values from the above for portability:
    m, c = 1.60178571, -0.01785714
    return m*current + c

# Define a function to calculate the noise using equation (11) in the paper:
def calculate_noise(image, current=400, dose_fraction=0.75, sigma_e=np.sqrt(2), num_detectors=1276):
    '''
    :param image: numpy array_like
    :param current: float
    :param dose_fraction: float <= 1
    :param sigma_e: float
    :param num_detectors: int
    :return noisy_image: ndarray (512,512,1)
    '''
    # Find calibration quanta:
    N_e = sigma_e**2
    N_0a = XCIST_incident_photon_number(current)
    # Normalise image values:
    image = from_HU_to_normal(image)
    shape = np.squeeze(image).shape[::-1]
    # Obtain sinogram of image using tomosipo:
    vg = ts.volume(shape=(1, *shape), size=(5, 300, 300))
    pg = ts.cone(angles=360, shape=(1, num_detectors), size=(1, num_detectors), src_det_dist=1080)
    A = ts.operator(vg, pg)
    expP = A(image.T)
    # Establish an array of variables x~N(0,1) to be used in the calculation:
    x = np.random.normal(0, 1, expP.shape)
    # Equation (11) in the paper:
    term_1 = ((1 - dose_fraction) / dose_fraction) * expP / N_0a
    term_2 = ((1 + dose_fraction) / dose_fraction) * (N_e * expP) / N_0a
    added_noise = np.sqrt(term_1 * (1 + term_2)) * x
    # Here is where the bowtie filter would be applied, if we had the data.
    expP_noisy = expP + added_noise
    # Reconstruct the image from the noisy sinogram:
    noisy_image = from_normal_to_HU(fbp(A, torch.Tensor(expP_noisy)).detach().cpu().numpy())[0].T
    noisy_image = noisy_image

    return noisy_image