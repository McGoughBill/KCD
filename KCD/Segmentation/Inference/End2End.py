import os
os.environ['OV_DATA_BASE'] = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/'
from KCD.Segmentation.Inference.endtoend import EnsembleSeg, masking
import numpy as np
from skimage.measure import label
import nibabel as nib

def detection(home, pred_path, op_point, name):
    assert os.path.exists(home), 'Path does not exist'
    assert os.path.exists(pred_path), 'Path does not exist'
    assert op_point >= 0 and op_point <= 1, 'Op point must be between 0 and 1'

    predictions = [x for x in os.listdir(pred_path) if 'nii' in x]
    save_path = os.path.join(home, 'detections', name)

    pos_csv = os.path.join(save_path, 'positive.csv')
    neg_csv = os.path.join(save_path, 'negative.csv')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #if pos or neg csv already exists, delete it
    if os.path.exists(pos_csv):
        os.remove(pos_csv)
    if os.path.exists(neg_csv):
        os.remove(neg_csv)

    #open pos and neg csv files
    pos_csv = open(pos_csv, 'w')
    neg_csv = open(neg_csv, 'w')


    for pred_fn in predictions:
        print(pred_fn)
        pred_path = os.path.join(pred_path, pred_fn)
        pred_nib = nib.load(pred_path)
        pred = pred_nib.get_fdata()
        voxvol = np.prod(pred_nib.header.get_zooms())

        pred_foreground = (pred > op_point).astype(int)
        pred_components = label(np.squeeze(pred_foreground))

        # if any component is greater than 500mm^3, append to pos_csv the name of the file,
        # the number of components, and the volume of the components. Otherwise, append to neg_csv
        component_volumes = [ (pred_components == i).sum() /voxvol for i in range(1, np.max(pred_components) + 1)]

        if any([x > 500 for x in component_volumes]):
            pos_csv.write(pred_fn + ',' + str(max(component_volumes)) + '\n')
        else:
            neg_csv.write(pred_fn + ',' + str(0) + '\n')

    pos_csv.close()
    neg_csv.close()



def EndToEnd(dataset, stage1_path, stage_2path, op_points = [0.01,0.91,0.99],
             op_point_names = ['high_sens','high_acc','high_spec'],
             save_path='None'):

    # high sens - based on small cancer detection
    # high acc and high spec - based on all cancer detection

    s1 = EnsembleSeg.Ensemble_Seg(dataset,name='s1',
                            seg_fp=stage1_path,
                            spacing=np.array([4] * 3),
                            do_prep=True, do_infer=True,
                            is_cect=False, cont=False,overlap=0.5,
                            batch_size=4, patch_size=64)

    dataset_path = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', dataset, 'images')
    masked_dataset_path = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'masked_' + dataset, 'images')
    pred_path = os.path.join(os.environ['OV_DATA_BASE'], 'predictions_nii', dataset, 's1_[4 4 4]mm')

    masking.mask_dataset(dataset_path, pred_path, masked_dataset_path)

    s2 = EnsembleSeg.Ensemble_Seg('masked_' + dataset,name='s2',
                            seg_fp=stage_2path,
                            spacing=np.array([2] * 3),
                            do_prep=True, do_infer=True,
                            is_cect=False, cont=True,overlap=0.5,
                            batch_size=4, patch_size=64)

    cont_pred_path = os.path.join(os.environ['OV_DATA_BASE'], 'predictions_nii', 'masked_' + dataset, 's2_[2 2 2]mm_cont')

    for i, op_point in enumerate(op_points):
        detection(os.environ['OV_DATA_BASE'], cont_pred_path, op_point, op_point_names[i])


if __name__ == '__main__':
    dataset = 'coreg_ncct'
    s1_path = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/trained_models/coreg_v2_noised/4mm_kidneybinary_customPP/6_finetune_trained'
    s2_path = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/trained_models/masked_coreg_v2_noised/2mm_cancerbinary_customPP/top2_fromablation/6_finetune_l2-0.0_cce1.0_cos1.0_2e-05lr_0.999997lg_16bs_2000ep_4gpus_efr_dfr100_seg'

    EndToEnd(dataset, s1_path, s2_path)