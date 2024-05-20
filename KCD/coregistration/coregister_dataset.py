import SimpleITK as sitk
import os

import nibabel as nib
import numpy as np
from utils import start_plot, end_plot, plot_values, display_images, display_images_with_alpha, update_multires_iterations
import matplotlib.pyplot as plt

# linear coreg with https://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/60_Registration_Introduction.html


def linear_coreg(moving,fixed,initial_transform,
                 moving_mask=True,fixed_mask=True):
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetMetricSamplingStrategy(registration_method.NONE)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=0.3, numberOfIterations=1000,
                                                      convergenceMinimumValue=1e-7, convergenceWindowSize=500,
                                                      estimateLearningRate = registration_method.EachIteration)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[3, 2, 1,])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 1])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    if moving_mask :
        # mask image using intensity thresholding
        mmask = sitk.IntensityWindowing(moving, windowMinimum=-200, windowMaximum=300)
        registration_method.SetMetricMovingMask( mmask>-50)
    if fixed_mask:
        fmask = sitk.IntensityWindowing(fixed, windowMinimum=-200, windowMaximum=300)
        registration_method.SetMetricFixedMask( fmask>-50)

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    # Connect all of the observers so that we can perform plotting during registration.
    registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
    registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
    registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations)
    # registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))

    final_transform = registration_method.Execute(sitk.Cast(fixed, sitk.sitkFloat32),
                                                  sitk.Cast(moving, sitk.sitkFloat32))

    return final_transform, registration_method

def nonlinear_coreg(moving,fixed,
                    moving_mask=True,fixed_mask=True, learning_rate=0.1):
    registration_method = sitk.ImageRegistrationMethod()

    # Determine the number of BSpline control points using the physical spacing we want for the control grid.
    grid_physical_spacing = [20.,20.,20.]  # A control point every 20mm
    image_physical_size = [size * spacing for size, spacing in zip(fixed_image.GetSize(), fixed_image.GetSpacing())]
    mesh_size = [int(image_size / grid_spacing + 0.5) \
                 for image_size, grid_spacing in zip(image_physical_size, grid_physical_spacing)]

    initial_transform = sitk.BSplineTransformInitializer(image1=fixed_image,
                                                         transformDomainMeshSize=mesh_size, order=3)
    registration_method.SetOptimizerScalesFromPhysicalShift()


    registration_method.SetInitialTransform(initial_transform)

    registration_method.SetMetricAsMeanSquares()
    # Settings for metric sampling, usage of a mask is optional. When given a mask the sample points will be
    # generated inside that region. Also, this implicitly speeds things up as the mask is smaller than the
    # whole image.
    registration_method.SetMetricSamplingStrategy(registration_method.NONE)

    if moving_mask :
        # mask image using intensity thresholding
        mmask = sitk.IntensityWindowing(moving, windowMinimum=-200, windowMaximum=300)
        registration_method.SetMetricMovingMask( mmask>-50)
    if fixed_mask:
        fmask = sitk.IntensityWindowing(fixed, windowMinimum=-200, windowMaximum=300)
        registration_method.SetMetricFixedMask( fmask>-50)


    # Multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[1])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
    registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
    registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations)
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))

    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(learningRate=0.3, numberOfIterations=20,
                                                      convergenceMinimumValue=1e-7, convergenceWindowSize=5,
                                                      estimateLearningRate = registration_method.EachIteration)

    final_transform = registration_method.Execute(fixed_image, moving)

    return final_transform, registration_method


if __name__ == '__main__':
    os.environ['OV_DATA_BASE'] = "/bask/projects/p/phwq4930-renal-canc/data/seg_data/"
    import sys
    is_kits = bool(str(sys.argv[1]) == 'True')
    small_first = bool(str(sys.argv[2]) == 'True')
    unseen = bool(str(sys.argv[3]) == 'True')

    if is_kits:
        if unseen:
            ncct_dir = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/raw_data/kits_ncct/unseen'
        else:
            ncct_dir = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/raw_data/kits_ncct/images'
        cect_dir = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/raw_data/kits23/images'
        learning_rate = 1
    else:
        ncct_dir = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/raw_data/add_ncct/images'
        cect_dir = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/raw_data/all_add/images'
        learning_rate = 1

    save_dir = ncct_dir+'_registered'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # ncct images have the following name structure: KiTS-XXXXX.nii.gz
    # cect images have the following name structure: case_XXXXX.nii.gz
    # where XXXXX is the case number - matching case numbers correspond to the same patient

    if is_kits:
        images = [f for f in os.listdir(ncct_dir) if f.endswith('.nii.gz')]

        # order ncct images by file size low to high - time efficient to register smaller images first
        # file size should be the max size between it and the corresponding cect image
        images.sort(key=lambda x: max(os.path.getsize(os.path.join(ncct_dir, x)),
                                           os.path.getsize(os.path.join(cect_dir, 'case_'+x.split('-')[1]))), reverse=(not small_first))
        padding = -1024
    else:
        # identify the matching ncct and cect images
        images = [f for f in os.listdir(ncct_dir) if f.endswith('.nii.gz') and f.split('_')[1] in
                    [f.split('_')[1] for f in os.listdir(cect_dir) if f.endswith('.nii.gz')]]

        # order ncct images by file size low to high - time efficient to register smaller images first
        # file size should be the max size between it and the corresponding cect image
        images.sort(key=lambda x: max(os.path.getsize(os.path.join(ncct_dir, x)),
                                           os.path.getsize(os.path.join(cect_dir, 'Rcc_'+x.split('_')[1]))), reverse=(not small_first))
        padding = -1024

    for image in images:

        nc_fp = os.path.join(ncct_dir, image)
        if is_kits:
            ce_casename = 'case_'+image.split('-')[1]
        else:
            ce_casename = 'Rcc_'+image.split('_')[1]
        print('ce case name:', ce_casename+'.')
        print(os.path.exists(os.path.join(cect_dir, ce_casename)))
        ce_fp = os.path.join(cect_dir, ce_casename)
        print('file sizes:', os.path.getsize(nc_fp)/1e6, os.path.getsize(ce_fp)/1e6)

        save_fp = os.path.join(save_dir, image)
        if os.path.exists(save_fp):
            print(f'{save_fp} already exists, skipping')
            continue

        if not is_kits:
            nib_im = nib.load(ce_fp)
            nib.as_closest_canonical(nib_im)
            im, affine = nib_im.get_fdata(), nib_im.affine
            nib_im = nib.Nifti1Image(im, affine)
            nib_im.header.set_data_dtype(im.dtype)
            nib.save(nib_im, ce_fp)


        # Read the images
        fixed_image = sitk.ReadImage(ce_fp, sitk.sitkFloat32, imageIO="NiftiImageIO")
        moving_image = sitk.ReadImage(nc_fp, sitk.sitkFloat32,imageIO="NiftiImageIO")


        initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                              moving_image,
                                                              sitk.Euler3DTransform(),
                                                              sitk.CenteredTransformInitializerFilter.GEOMETRY)

        linear_transform, reg_method = linear_coreg(moving_image,fixed_image,initial_transform)
        lin_reg_met = reg_method.GetMetricValue()
        print('Final metric value: {0}'.format(lin_reg_met))
        print('Optimizer\'s stopping condition, {0}'.format(reg_method.GetOptimizerStopConditionDescription()))

        linear_img = sitk.Resample(moving_image, fixed_image, linear_transform, sitk.sitkLinear, padding,
                                         moving_image.GetPixelID())

        # apply nonlinear coregistration
        nonlinear_transform, reg_method = nonlinear_coreg(linear_img,fixed_image,learning_rate=learning_rate)

        print('Final metric value: {0}'.format(reg_method.GetMetricValue()))
        print('Optimizer\'s stopping condition, {0}'.format(reg_method.GetOptimizerStopConditionDescription()))

        nonlinear_img = sitk.Resample(linear_img, fixed_image, nonlinear_transform, sitk.sitkLinear, padding,
                                         linear_img.GetPixelID())

        # if optimisation metric is greater in the nonlinear coregistration, save the linear coregistered image
        if reg_method.GetMetricValue() > lin_reg_met:
            print('Nonlinear coregistration did not improve the optimisation metric, saving linear coregistered image')
            sitk.WriteImage(linear_img, save_fp)
        else:
            print('Nonlinear coregistration improved the optimisation metric, saving nonlinear coregistered image')
            # save the moving image
            sitk.WriteImage(nonlinear_img, save_fp)
        print(f'Saved the registered image to {save_fp}')