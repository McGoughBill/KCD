import slice_generating_utils as generator
import numpy as np
#
# data_name = 'lowdose_test'
# path = "/path/to/kcd_data/data/seg_data/raw_data/{}/images".format(data_name)
# save_dir = "/path/to/kcd_data/data/seg_data/"
# segpath = "/path/to/kcd_data/data/seg_data/predictions_nii/{}/s1_[4 4 4]mm".format(data_name)
#
# overlap_mm = 40 ## this dictates the minimum distance apart between each slice! not the overlap.
# patch2d = 224 ## in plane slice spacing
# save_limit_percase_perlabel = 50 ## maximum number of saved shifted window slices per kidney
# voxel_spacings = [1.5] ## isotropic voxel size in mm
# thresholds_r_mm = [0] ## threshold used to calculate cancer label
# depth_z = 60 ## size of slice depth in axial dimension in mm - if 1mm, then 2D
# bbox_boundary_mm = 40 ## dilation of segmentation label
# boundary_z= 20 ## axial spacing between slice sampling
# patch_size = np.array([patch2d]*3)
# patch_size[0]=depth_z
#
# for thresh in thresholds_r_mm:
#     for spacing in voxel_spacings:
#         target_spacing = np.array([spacing]*3)
#         overlap = overlap_mm/(patch_size*spacing)
#         # overlap_mm = spacing*patch2d/10
#         print("Voxel Spacing {}mm".format(spacing))
#         generator.create_unlabelled_dataset(path,save_dir,segpath,target_spacing,overlap,
#                        patch_size,thresh,spacing,save_limit_percase_perlabel,
#                        bbox_boundary_mm,data_name=data_name,boundary_z=boundary_z,
#                        depth_z=depth_z,is_merlin=True)


data_name = 'fulldose_test'
path = "/path/to/kcd_data/data/seg_data/raw_data/{}/images".format(data_name)
save_dir = "/path/to/kcd_data/data/seg_data/"
segpath = "/path/to/kcd_data/data/seg_data/predictions_nii/fulldose_test/s1_[4 4 4]mm"

overlap_mm = 40 ## this dictates the minimum distance apart between each slice! not the overlap.
patch2d = 224 ## in plane slice spacing
save_limit_percase_perlabel = 50 ## maximum number of saved shifted window slices per kidney
voxel_spacings = [1.5] ## isotropic voxel size in mm
thresholds_r_mm = [0] ## threshold used to calculate cancer label
depth_z = 60 ## size of slice depth in axial dimension in mm - if 1mm, then 2D
bbox_boundary_mm = 40 ## dilation of segmentation label
boundary_z= 20 ## axial spacing between slice sampling
patch_size = np.array([patch2d]*3)
patch_size[0]=depth_z

for thresh in thresholds_r_mm:
    for spacing in voxel_spacings:
        target_spacing = np.array([spacing]*3)
        overlap = overlap_mm/(patch_size*spacing)
        # overlap_mm = spacing*patch2d/10
        print("Voxel Spacing {}mm".format(spacing))
        generator.create_unlabelled_dataset(path,save_dir,segpath,target_spacing,overlap,
                       patch_size,thresh,spacing,save_limit_percase_perlabel,
                       bbox_boundary_mm,data_name=data_name,boundary_z=boundary_z,
                       depth_z=depth_z,is_merlin=True)