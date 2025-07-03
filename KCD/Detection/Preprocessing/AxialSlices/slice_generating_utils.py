import sys

import array_manipulation_utils as am
import numpy as np
import os
from skimage.measure import regionprops
import scipy.ndimage as spim
import torch.nn as nn
import torch
from scipy import stats
import file_utils as fu
import gc

def extract_sliding_windows(arr, patch_size=(32,32,32),overlap=0.1,boundary_z=1,axes=None):
    axial,lr,ud = axes

    # in case the volume is smaller than the patch size we pad it
    # and save the input size to crop again before returning
    volume = torch.unsqueeze(torch.Tensor(arr),dim=0).type(torch.float32)
    shape_in = np.array(volume.shape)
    #  possible padding of too small volumes
    pad = [0, patch_size[0] - shape_in[3], 0, patch_size[2] - shape_in[2],
           0, patch_size[1] - shape_in[1]]
    pad = np.maximum(pad, 0).tolist()

    volume = nn.functional.pad(volume, pad).type(torch.float32)
    shape = volume.shape[1:]

    zxy_shape = np.array([shape[axes[0]], shape[axes[1]], shape[axes[2]]])
    #  get all top left coordinates of patches
    zxy_list = _get_zxy_list(zxy_shape,patch_size=patch_size,overlap=overlap,boundary_z=boundary_z)
    # introduce batch size
    # some people say that introducing a batch size at inference time makes it faster
    # I couldn't see that so far
    n_full_batches = len(zxy_list)
    zxy_batched = [zxy_list[i : (i + 1) ]
                   for i in range(n_full_batches)]

    
    if n_full_batches < len(zxy_list):
        zxy_batched.append(zxy_list[n_full_batches:])

    sliding_windows = torch.stack([volume[:,
                                pack[0][np.argmax(axes==0)]:pack[0][np.argmax(axes==0)]+patch_size[np.argmax(axes==0)],
                                pack[0][np.argmax(axes==1)]:pack[0][np.argmax(axes==1)]+patch_size[np.argmax(axes==1)],
                                pack[0][np.argmax(axes==2)]:pack[0][np.argmax(axes==2)]+patch_size[np.argmax(axes==2)]]
                                   for pack in zxy_batched], dim=0).type(torch.float32)

    return sliding_windows

def _get_zxy_list(shape,patch_size = np.array([32,32,32]),overlap=0.1,boundary_z=1):
    
    nz, nx, ny = shape

    inplane = stats.mode(overlap,keepdims=False)[0]
    overlap_arr = np.where(overlap==inplane,overlap*patch_size,boundary_z)
    n_patches = np.ceil((np.array([nz, nx, ny]) - patch_size) / 
                        overlap_arr).astype(int)
    

    if patch_size[-1] ==1:
        n_patches[-1] = ny
        
    for i in range(len(n_patches)):
        if n_patches[i] ==0:
            n_patches[i] =1
        
    # upper left corners of all patches
    z_list = np.linspace(0, nz - patch_size[0], n_patches[0]).astype(int).tolist()
    x_list = np.linspace(0, nx - patch_size[1], n_patches[1]).astype(int).tolist()

    y_list = np.linspace(0, ny - patch_size[2], n_patches[2]).astype(int).tolist()
    xyz_list = []
    for z in z_list:
        for x in x_list:
            for y in y_list:
                xyz_list.append((z, x, y))
                # print(z, x, y)
        # assert(1==2)
    return xyz_list

def get_masses(binary_arr,vol_thresh,intensity_image = None):
    return [[mass,mass.centroid] for mass in regionprops(spim.label(binary_arr)[0],intensity_image = intensity_image) if mass.area>vol_thresh]

def is_sole_kidney_central(kidney_centroids, im,inf, inplane_spac,axes,
                           test1_length = 25, test2_length = 10):
    sole_kidney = kidney_centroids[0][1]
    axial,lr_index,ud_index = axes
    
    # Find centre of bone-attenuating tissue - lr is left-to-right on axial, ud is up-down
    try:
        lr_bone, ud_bone = np.array(regionprops((im>250).astype(int))[0].centroid)[lr_index,ud_index]
    except IndexError:
        # if no bone-attenuating tissue is found, assume bone attenuation centre is in centre of the image.
        lr_bone = im.shape[lr_index] // 2
        ud_bone = im.shape[ud_index] // 2
    
    # test 1 - does the centre of the single kidney line up with spine within 25mm? if so - central kidney
    if abs(sole_kidney[lr_index] - lr_bone)*inplane_spac < test1_length:
        return True, ud_bone, lr_bone
    else:
        # test 2 - kidney is also central if wraps around spine.
        # does some portion of the kidney wrap around the spine?
        
        # test 2 distance is 10mm
        _test_extent_inpixels = int((test2_length/2)*inplane_spac)
        
        # create test label - where the pixels within +-10mm of centre of bone attenuating tissue are zeroed out
        _central_test = inf
        _central_test[:,
                      int(lr_bone-_test_extent_inpixels):int(lr_bone+_test_extent_inpixels),
                      :] = 0
        
        # wrapping is true if one or more objects from test label appear either side of the centre of bone attenuating tissue.
        # if wrapping is true, then the kidney is central.
        _test_centroids = [ centroid[1]> lr_bone for _, centroid in get_masses(_central_test,0)]
        if (False in _test_centroids) and (True in _test_centroids):
            return True, ud_bone, lr_bone
        else:
            return False, ud_bone, lr_bone
    
def get_shifted_windows(reshaped_im,reshaped_seg,
                        overlap=None,patch_size=None,axes=None,boundary_z=1):
    

    sw_reshape_im = extract_sliding_windows(reshaped_im, overlap=overlap, patch_size=patch_size,axes=axes,boundary_z=boundary_z)
    sw_reshape_seg = extract_sliding_windows(reshaped_seg, overlap=overlap, patch_size=patch_size,axes=axes,boundary_z=boundary_z)  
    
    
    return sw_reshape_im, sw_reshape_seg

def get_centralised_windows(reshaped_im,reshaped_seg,centroid,axes=None,patch_size=None,
                            boundary_z = 1):
    axial_plane,lr,ud = axes
    shape_in = reshaped_im.shape
    xc,yc,zc = centroid
    pad = [patch_size[2] - shape_in[0],
           0, patch_size[1] - shape_in[1], 0, patch_size[0] - shape_in[2],0][::-1]
    pad = np.maximum(pad, 0).tolist()
    reshaped_im = nn.functional.pad(torch.Tensor(reshaped_im), pad).type(torch.float32).numpy()
    reshaped_seg = nn.functional.pad(torch.Tensor(reshaped_seg), pad).type(torch.float32).numpy()

    shape = reshaped_im.shape


    top_left_x = max(0,int(xc-(patch_size[1]/2)))
    top_left_y = max(0,int(yc-(patch_size[2]/2)))

    bottom_right_y = min(shape[ud],top_left_y+patch_size[2])
    bottom_right_x = min(shape[lr],top_left_x+patch_size[1])
    
    if bottom_right_x == shape[lr]: top_left_x = bottom_right_x-patch_size[1]
    if bottom_right_y == shape[ud]: top_left_y = bottom_right_y-patch_size[2]

    zxy_shape = np.array([shape[axes[0]], shape[axes[1]], shape[axes[2]]])
    
    z_list = np.arange(0,shape[axial_plane]-patch_size[0],boundary_z)
    central_window_im = np.zeros((max(1,len(z_list)),patch_size[1],patch_size[2], patch_size[0]),dtype=np.float32)
    central_window_seg = np.zeros((max(1,len(z_list)),patch_size[1],patch_size[2],patch_size[0]),dtype=np.uint16)

    if len(z_list) == 0:
        z_list = [0]

    print('reshape im shape:',reshaped_im.shape)

    for i,z in enumerate(z_list):
        central_window_im[i, :, :, :] = reshaped_im[top_left_x:bottom_right_x, top_left_y:bottom_right_y,
                                        z:z + patch_size[0]]
        central_window_seg[i, :, :, :] = reshaped_seg[top_left_x:bottom_right_x, top_left_y:bottom_right_y,
                                         z:z + patch_size[0]]


    return central_window_im, central_window_seg

def get_kid_str(kidney_data,reshaped_im,reshaped_seg,spacing,axes,labelled=True):
    axial,lr_index,ud = axes
    
    if len(kidney_data)==1:
        central_kidney_flag, ud_bone, lr_bone = is_sole_kidney_central(kidney_data,reshaped_im,
                                                                       reshaped_seg,np.mean(spacing[1:]),axes)
        if central_kidney_flag:
            kid_str = ['central']
        elif kidney_data[0][1][lr_index] - lr_bone > 0:
            kid_str = ['left']
        else:
            kid_str = ['right']
            
        print("sole kidney is {}".format(kid_str))
    elif len(kidney_data) == 0: kid_str=['_failure']
    elif len(kidney_data)>2:
        if labelled:
            kid_str = ['_failure']
        else:
            kid_str = np.arange(1,len(kidney_data)+1).astype(str).tolist()
    else:
        first_kidney = kidney_data[0]
        second_kidney = kidney_data[1]
    
        if first_kidney[1][lr_index] < second_kidney[1][lr_index]:
            kid_str = ['right','left']
        else:
            kid_str = ['left','right']
            
    return kid_str


def save_windows_segmentation_labelled(windowed_im,windowed_seg,canc_thresh,
                 kid_thresh,target_spacing,slice_save_path, seg_save_path,voxel_size_mm,
                 thresh_r_mm,case_name,kidney_side='random',
                 shuffle=True,save_limit=100,centralised=False,depth_z=1,
                 has_seg_label = True,boundary_z=1,kidthresh=20,dilate=40,patch_dims=(1,224,224)):
    
    if centralised: string = 'centralised'
    else: string = 'shifted'
    
    sw_maligs,seg_maligs,sw_normals,seg_normals,sw_none,seg_none= am.filter_shifted_windows_seg(windowed_im,windowed_seg,canc_thresh,kid_thresh,target_spacing[0],shuffle=shuffle,has_seg_label=True)
    sw_subtypes = [(sw_maligs,seg_maligs),(sw_normals,seg_normals),(sw_none,seg_none)]
    names = ["tumour","kidney","none"]
    print("Creating {} tumour, and {} kidney {} windows.".format(min(len(sw_maligs),save_limit),
                                                                      min(len(sw_normals),save_limit),
                                                                      string))

    for subname, (label_group,label_group_seg) in zip(names,sw_subtypes):
        sw_fold = fu.filename_structure_labelled(slice_save_path,subname,voxel_size_mm,thresh_r_mm,kidthresh,
                                     depth_z,boundary_z,dilate)
        seg_fold = fu.filename_structure_labelled(seg_save_path,subname,voxel_size_mm,thresh_r_mm,kidthresh,
                                     depth_z,boundary_z,dilate)
        for i,sw in enumerate(label_group):
            if i >= save_limit: break
            sw = np.squeeze(sw)
            seg = np.squeeze(label_group_seg[i])
            np.save(os.path.join(sw_fold,"{}_{}_{}_index{}".format(case_name[:-7].replace('-','_'),kidney_side,string,i)),sw)
            np.save(os.path.join(seg_fold,"{}_{}_{}_index{}".format(case_name[:-7].replace('-','_'),kidney_side,string,i)),seg)

def save_windows_labelled(windowed_im, windowed_seg, canc_thresh,
                          kid_thresh, target_spacing, save_path, voxel_size_mm,
                          thresh_r_mm, case_name, kidney_side='random',
                          shuffle=True, save_limit=100, centralised=False, depth_z=1,
                          has_seg_label=True, boundary_z=1, kidthresh=20, dilate=40, patch_dims=(1, 224, 224)):
    if centralised:
        string = 'centralised'
    else:
        string = 'shifted'

    sw_maligs, sw_normals, sw_none = am.filter_shifted_windows(windowed_im, windowed_seg, canc_thresh, kid_thresh,
                                                               target_spacing[0], shuffle=shuffle, has_seg_label=True)
    sw_subtypes = [sw_maligs, sw_normals, sw_none]
    names = ["tumour", "kidney", "none"]
    print("Creating {} tumour, and {} kidney {} windows.".format(min(len(sw_maligs), save_limit),
                                                                 min(len(sw_normals), save_limit),
                                                                 string))

    for subname, label_group in zip(names, sw_subtypes):
        fold = fu.filename_structure_labelled(save_path, subname, voxel_size_mm, thresh_r_mm, kidthresh,
                                              depth_z, boundary_z, dilate)
        for i, sw in enumerate(label_group):
            if i >= save_limit: break
            sw = np.squeeze(sw)
            np.save(
                os.path.join(fold, "{}_{}_{}_index{}".format(case_name[:-7].replace('-', '_'), kidney_side, string, i)),
                sw)


def save_windows_unlabelled(windowed_im,windowed_seg,fg_thresh,target_spacing,
                            save_path,voxel_size_mm,
                             thresh_r_mm,case_name,kidney_side='random',
                             shuffle=True,save_limit=100,centralised=False,depth_z=1,
                             has_seg_label = True,boundary_z=1,kidthresh=20,dilate=40,patch_dims=(1,224,224)):
    
    if centralised: string = 'centralised'
    else: string = 'shifted'
    
    _,sw_normals,sw_none= am.filter_shifted_windows(windowed_im,windowed_seg,1e6,fg_thresh,target_spacing[0],shuffle=shuffle,has_seg_label=False)
    if min(len(sw_normals),save_limit) == 0:
        # do not use filtering if there are no foreground windows
        sw_subtypes = [windowed_im]
        sw_normals = windowed_im

    sw_subtypes = [sw_normals]
    names = ["foreground"]

    print("Creating {} foreground {} windows.".format(min(len(sw_normals),save_limit),
                                                                      string))

    for subname, label_group in zip(names,sw_subtypes):
        fold = fu.filename_structure_unlabelled(save_path,subname,voxel_size_mm,thresh_r_mm,
                                     depth_z,boundary_z,dilate)
        for i,sw in enumerate(label_group):
            if i >= save_limit: break
            sw = np.squeeze(sw)
            np.save(os.path.join(fold,"{}_{}_{}_index{}".format(case_name[:-7].replace('-','_'),kidney_side,string,i)),sw)
            

def create_labelled_dataset(im_path,save_dir,seg_path,target_spacing,overlap,
                   patch_dims,cancer_thresh_rmm,
                   voxel_size_mm, save_limit,bbox_boundary_mm=50,
                   data_name='coreg_ncct',boundary_z=1,depth_z=1,
                   kidney_thresh_rmm=20,is_merlin=False):
    
    save_path = fu.create_save_path_structure(im_path,data_name=data_name,save_dir=save_dir)
    int_list = [file for file in os.listdir(im_path)][::-1]
    
    bbox_boundary = int(np.round(bbox_boundary_mm/target_spacing[0]))
    vol_thresh_vox = 1000 / float(np.prod(target_spacing)) # ignore 'kidney' segmentations with a vol less than 100mm cubed
    kernel = spim.generate_binary_structure(3, 2).astype(np.uint8)
    canc_thresh = ((cancer_thresh_rmm/target_spacing[0])**2) * 3.1416
    kid_thresh = ((kidney_thresh_rmm/target_spacing[0])**2) * 3.1416
    
    for get_int in int_list:
        if get_int=='KiTS-00151.nii.gz': continue #skip this case - strange label artefact
        if get_int=='.DS_Store': continue
        ct,seg = am.get(get_int,im_path,seg_path)
        (ct_im,_),(seg_im,spacing) = am.nifti_2_correctarr(ct), am.nifti_2_correctarr(seg)
        ct_im = ct_im.astype(np.float32)
        patch_size= patch_dims.copy()
        spacing_axes = am.find_orientation(spacing,is_axes=False)
        if spacing_axes == (0,0,0):continue
        z_spac,inplane_spac = spacing[spacing_axes[0]], spacing[spacing_axes[1]]
        correct_spacing = np.array((z_spac,inplane_spac,inplane_spac))
        axes = am.find_orientation(ct_im.shape,is_axes=True,im=ct_im)
        if axes == (0,0,0):continue
        axial,lr,ud = axes
        axes = np.array(axes)


        
        reshaped_im = am.rescale_array(ct_im, correct_spacing, axes, target_spacing = target_spacing).astype(np.float32)
        reshaped_seg = am.rescale_array(seg_im, correct_spacing, axes, target_spacing = target_spacing, is_seg = True).astype(np.float32)

        if is_merlin:
            # use torch interpolation to rescale z-axis to half resolution
            scale_factor = (1,1,0.5)
            reshaped_im = nn.functional.interpolate(torch.unsqueeze(torch.unsqueeze(torch.Tensor(reshaped_im),dim=0),dim=0).type(torch.float32),
                                                    mode = 'trilinear', scale_factor=scale_factor).numpy()[0,0].astype(np.float32)
            reshaped_seg = nn.functional.interpolate(torch.unsqueeze(torch.unsqueeze(torch.Tensor(reshaped_seg),dim=0),dim=0).type(torch.float32),
                                                    mode = 'nearest', scale_factor=scale_factor).numpy()[0,0].astype(np.uint16)

        kidney_data = np.asarray([[mass,centroid] for mass, centroid in get_masses(reshaped_seg>0,vol_thresh_vox)],dtype=object)        

        kid_str = get_kid_str(kidney_data,reshaped_im,reshaped_seg,spacing,axes,labelled=True)
        if kid_str == ['_failure']: continue
                
        for kidney_datum, name in zip(kidney_data,kid_str):
            print("\nGenerating from {} {}-side.".format(get_int,name))
            #  this data is for training only - testing will ignore generic 
            coords = np.array(kidney_datum[0].bbox)

            centroid = kidney_datum[1]
            
            ### final mask should be generated from a 3 dilation of the masked segmentation.
            first_mask = np.zeros_like(reshaped_seg)
            first_mask[coords[0]:coords[3],
                       coords[1]:coords[4],
                       coords[2]:coords[5]] = np.ones((coords[3]-coords[0],
                                                       coords[4]-coords[1],
                                                       coords[5]-coords[2]))
            first_mask = first_mask*(reshaped_seg>0).astype(np.uint8)
            final_mask = spim.binary_dilation(first_mask, kernel, iterations=bbox_boundary).astype(np.uint16)

            seg = first_mask*reshaped_seg
            ct = (final_mask*reshaped_im).astype(np.float32)
            # get rid of zero'd background that confounds training - make bg -200HU, the min possible val in image.
            ct += np.where(final_mask==0,-200,0).astype(np.float32)
            print(' ct shape:',ct.shape,seg.shape)
            
            sw_im, sw_seg = get_shifted_windows(ct,seg,overlap=overlap,patch_size=patch_size,axes=axes,boundary_z=boundary_z)
            #convert to numpy
            sw_im, sw_seg = sw_im.numpy(), sw_seg.numpy()
            print('sw_im shape:',sw_im.shape,sw_seg.shape)
            save_windows_labelled(sw_im,sw_seg,canc_thresh,kid_thresh,
                         target_spacing,save_path,
                         voxel_size_mm,cancer_thresh_rmm,get_int,
                         shuffle=True,save_limit=save_limit,centralised=False,
                         depth_z=depth_z,boundary_z=boundary_z,
                         kidthresh=kidney_thresh_rmm,dilate=bbox_boundary_mm,patch_dims=patch_dims)

            cent_im, cent_seg = get_centralised_windows(ct,seg,centroid,patch_size=patch_size,axes=axes,boundary_z=boundary_z)
            print('cent_im shape:',cent_im.shape,cent_seg.shape)
            save_windows_labelled(cent_im,cent_seg,canc_thresh,kid_thresh,
                         target_spacing,save_path,
                         voxel_size_mm,cancer_thresh_rmm,get_int,kidney_side=name,
                         shuffle=False,save_limit=1e4,centralised=True,
                         depth_z=depth_z,boundary_z=boundary_z,
                         kidthresh=kidney_thresh_rmm,dilate=bbox_boundary_mm,patch_dims=patch_dims)
            sys.stdout.flush()
            del ct, seg, first_mask, final_mask, cent_seg, cent_im, sw_seg, sw_im
            gc.collect()
            torch.cuda.empty_cache()


        del reshaped_im, reshaped_seg
        gc.collect()
        torch.cuda.empty_cache()
        print()
                        
            
def create_unlabelled_dataset(im_path,save_dir,seg_path,target_spacing,overlap,
                   patch_dims,foreground_thresh,
                   voxel_size_mm, save_limit,bbox_boundary_mm=50,
                   data_name='coreg_ncct',boundary_z=1,depth_z=1,
                   kidney_thresh_rmm=20,is_merlin=False):
    save_path = fu.create_save_path_structure(im_path,data_name=data_name,save_dir=save_dir)
    int_list = [file for file in os.listdir(im_path)]

    bbox_boundary = int(np.round(bbox_boundary_mm/target_spacing[0]))
    vol_thresh_vox = 1000 / float(np.prod(target_spacing)) # ignore 'kidney' segmentations with a vol less than 100mm cubed
    kernel = spim.generate_binary_structure(3, 2).astype(np.uint8)
    fg_thresh = ((foreground_thresh/target_spacing[0])**2) * 3.1416
    for get_int in int_list:
        if get_int=='.DS_Store': continue
        ct,seg = am.get(get_int,im_path,seg_path)
        (ct_im,_),(seg_im,spacing) = am.nifti_2_correctarr(ct), am.nifti_2_correctarr(seg)
        patch_size= patch_dims.copy()
        spacing_axes = am.find_orientation(spacing,is_axes=False)
        if spacing_axes == (0,0,0):
            # check if all spacings are identical; if so, use shape of data to generate axes
            if len(set(spacing))==1:
                pass
            else:
                print('spacing axes not found for:',get_int)
                continue
        z_spac,inplane_spac = spacing[spacing_axes[0]], spacing[spacing_axes[1]]
        correct_spacing = np.array((z_spac,inplane_spac,inplane_spac))
        axes = np.array([2,1,0])

        
        reshaped_im = am.rescale_array(ct_im,correct_spacing,axes,target_spacing = target_spacing).astype(np.float32)
        reshaped_seg = am.rescale_array(seg_im,correct_spacing,axes,target_spacing = target_spacing,is_seg=True)

        if is_merlin:
            # use torch interpolation to rescale z-axis to half resolution
            scale_factor = (1,1,0.5)
            reshaped_im = nn.functional.interpolate(torch.unsqueeze(torch.unsqueeze(torch.Tensor(reshaped_im),dim=0),dim=0).type(torch.float32),
                                                    mode = 'trilinear', scale_factor=scale_factor).numpy()[0,0].astype(np.float32)
            reshaped_seg = nn.functional.interpolate(torch.unsqueeze(torch.unsqueeze(torch.Tensor(reshaped_seg),dim=0),dim=0).type(torch.float32),
                                                    mode = 'nearest', scale_factor=scale_factor).numpy()[0,0].astype(np.uint16)



        kidney_data = np.asarray([[mass,centroid] for mass, centroid in get_masses(reshaped_seg>0,vol_thresh_vox)],dtype=object)
        kid_str = get_kid_str(kidney_data,reshaped_im,reshaped_seg,spacing,axes,labelled=False)
        sys.stdout.flush()
        if kid_str == ['_failure']:continue

        print('Number of positive voxels in segmentation:',np.sum(reshaped_seg>0))

                
        for kidney_datum, name in zip(kidney_data,kid_str):
            print("\nGenerating from {} {}-side.".format(get_int,name))
            #  this data is for training only - testing will ignore generic 
            coords = np.array(kidney_datum[0].bbox)

            centroid = kidney_datum[1]
            ### final mask should be generated from a 3 dilation of the masked segmentation.
            first_mask = np.zeros_like(reshaped_seg).astype(np.uint8)
            first_mask[coords[0]:coords[3],
                       coords[1]:coords[4],
                       coords[2]:coords[5]] = np.ones((coords[3]-coords[0],
                                                       coords[4]-coords[1],
                                                       coords[5]-coords[2]))

            first_mask = first_mask*(reshaped_seg>0).astype(np.uint8)
            final_mask = spim.binary_dilation(first_mask, kernel, iterations=bbox_boundary).astype(np.uint16)
            
            seg = first_mask*reshaped_seg
            print('number of positive voxels in {} segmentation:'.format(name),np.sum(seg>0))
            ct = (final_mask*reshaped_im).astype(np.float32)
            # get rid of zero'd background that confounds training - make bg -200HU, the min possible val in image.
            ct += np.where(final_mask==0,-200,0).astype(np.float32)

            print('Unique values in ct:',np.unique(ct,return_counts=True))

            cent_im, cent_seg = get_centralised_windows(ct,seg,centroid,patch_size=patch_size,axes=axes,boundary_z=boundary_z)
            print('cent_im shape:',cent_im.shape,cent_seg.shape)

            print('Unique values in centralised ct:',np.unique(cent_im,return_counts=True))
            
            save_windows_unlabelled(cent_im,cent_seg,fg_thresh,
                         target_spacing,save_path,
                         voxel_size_mm,foreground_thresh,get_int,kidney_side=name,
                         shuffle=False,save_limit=1e4,centralised=True,
                         depth_z=depth_z,boundary_z=boundary_z,
                         kidthresh=kidney_thresh_rmm,dilate=bbox_boundary_mm,patch_dims=patch_dims)
            del ct, seg, first_mask, final_mask, cent_seg, cent_im
            gc.collect()
            torch.cuda.empty_cache()

        del reshaped_im, reshaped_seg
        gc.collect()
        torch.cuda.empty_cache()


def create_segmentation_labelled_dataset(im_path, save_dir, seg_path, target_spacing, overlap,
                            patch_dims, cancer_thresh_rmm,
                            voxel_size_mm, save_limit, bbox_boundary_mm=50,
                            data_name='coreg_ncct', boundary_z=1, depth_z=1,
                            kidney_thresh_rmm=20):
    seg_save_path,slice_save_path = fu.create_save_path_structure_seg(im_path, data_name=data_name, save_dir=save_dir)
    int_list = [file for file in os.listdir(im_path)]

    bbox_boundary = int(np.round(bbox_boundary_mm / target_spacing[0]))
    vol_thresh_vox = 1000 / float(
        np.prod(target_spacing))  # ignore 'kidney' segmentations with a vol less than 100mm cubed
    kernel = spim.generate_binary_structure(3, 2).astype(np.uint8)
    canc_thresh = ((cancer_thresh_rmm / target_spacing[0]) ** 2) * 3.1416
    kid_thresh = ((kidney_thresh_rmm / target_spacing[0]) ** 2) * 3.1416

    for get_int in int_list:
        if get_int == 'KiTS-00151.nii.gz': continue  # skip this case - strange label artefact
        if get_int == '.DS_Store': continue
        ct, seg = am.get(get_int, im_path, seg_path)
        ct_im, seg_im = am.nifti_2_correctarr(ct), am.nifti_2_correctarr(seg)
        patch_size = patch_dims.copy()
        spacing = am.get_spacing(ct)
        spacing_axes = am.find_orientation(spacing, is_axes=False)
        if spacing_axes == (0, 0, 0): continue
        z_spac, inplane_spac = spacing[spacing_axes[0]], spacing[spacing_axes[1]]
        correct_spacing = np.array((z_spac, inplane_spac, inplane_spac))
        axes = am.find_orientation(ct_im.shape, is_axes=True, im=ct_im)
        if axes == (0, 0, 0): continue
        axial, lr, ud = axes
        axes = np.array(axes)

        reshaped_im = am.rescale_array(ct_im, correct_spacing, axes, target_spacing=target_spacing)
        reshaped_seg = am.rescale_array(seg_im, correct_spacing, axes, target_spacing=target_spacing, is_seg=True)

        kidney_data = np.asarray([[mass, centroid] for mass, centroid in get_masses(reshaped_seg > 0, vol_thresh_vox)],
                                 dtype=object)

        kid_str = get_kid_str(kidney_data, reshaped_im, reshaped_seg, spacing, axes, labelled=True)
        if kid_str == ['_failure']: continue

        for kidney_datum, name in zip(kidney_data, kid_str):
            print("\nGenerating from {} {}-side.".format(get_int, name))
            #  this data is for training only - testing will ignore generic
            coords = np.array(kidney_datum[0].bbox)

            centroid = kidney_datum[1]

            ### final mask should be generated from a 3 dilation of the masked segmentation.
            first_mask = np.zeros_like(reshaped_seg)
            first_mask[coords[0]:coords[3],
            coords[1]:coords[4],
            coords[2]:coords[5]] = np.ones((coords[3] - coords[0],
                                            coords[4] - coords[1],
                                            coords[5] - coords[2]))
            first_mask = first_mask * (reshaped_seg > 0).astype(np.uint8)
            final_mask = spim.binary_dilation(first_mask, kernel, iterations=bbox_boundary)

            seg = first_mask * reshaped_seg
            ct = final_mask * reshaped_im
            # get rid of zero'd background that confounds training - make bg -200HU, the min possible val in image.
            ct += np.where(final_mask == 0, -200, 0)

            sw_im, sw_seg = get_shifted_windows(ct, seg, overlap=overlap, patch_size=patch_size, axes=axes,
                                                boundary_z=boundary_z)


            if (sw_seg.shape[-3:] == tuple(patch_dims)) and (sw_im.shape[-3:] == tuple(patch_dims)):
                save_windows_segmentation_labelled(sw_im, sw_seg, canc_thresh, kid_thresh,
                                      target_spacing, slice_save_path,seg_save_path,
                                      voxel_size_mm, cancer_thresh_rmm, get_int,
                                      shuffle=True, save_limit=save_limit, centralised=False,
                                      depth_z=depth_z, boundary_z=boundary_z,
                                      kidthresh=kidney_thresh_rmm, dilate=bbox_boundary_mm, patch_dims=patch_dims)

            cent_im, cent_seg = get_centralised_windows(ct, seg, centroid, patch_size=patch_size, axes=axes,
                                                        boundary_z=boundary_z)

            if (cent_im.shape[-3:] == tuple(patch_dims)) and (cent_seg.shape[-3:] == tuple(patch_dims)):
                save_windows_segmentation_labelled(cent_im, cent_seg, canc_thresh, kid_thresh,
                                      target_spacing, slice_save_path, seg_save_path,
                                      voxel_size_mm, cancer_thresh_rmm, get_int, kidney_side=name,
                                      shuffle=False, save_limit=1e4, centralised=True,
                                      depth_z=depth_z, boundary_z=boundary_z,
                                      kidthresh=kidney_thresh_rmm, dilate=bbox_boundary_mm, patch_dims=patch_dims)


            del ct, seg, first_mask, final_mask, cent_seg, cent_im,sw_seg, sw_im
            gc.collect()
            torch.cuda.empty_cache()

        del reshaped_im, reshaped_seg
        gc.collect()
        torch.cuda.empty_cache()
