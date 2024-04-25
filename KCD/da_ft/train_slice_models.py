from KCD.Detection.Training import train_utils as tu
import KCD.Detection.Evaluation.eval_scripts as eval_
from KCD.da_ft.finetune_models import LinearClassifier
from sklearn.model_selection import StratifiedKFold as kfold_strat
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import numpy as np
import warnings
import pandas as pd

#set seed
torch.manual_seed(0)
np.random.seed(0)

def finetune_da_model(path_to_pretrained_model,home = '/Users/mcgoug01/Downloads/Data/',
                         dataname='coreg_ncct', type='normal',
                         splits:list=[0],folds=5,params:dict=None):
    # Suppress warnings
    warnings.filterwarnings("ignore") #makes dgl stop complaining!

    # Initialization
    dev = tu.initialize_device()

    if params==None:params = tu.init_domainadaptation(type=type)
    else:tu.check_params(params,tu.init_domainadaptation(type=type))
    model_type = 'PatchModel'

    save_dir = tu.init_training_home(home, dataname)
    print(save_dir)

    slicedataset, test_slicedataset = tu.get_slice_data_DA(home, dataname, params['voxel_size'], params['cancthresh_r_mm'],params['kidthresh_r_mm'],params['depth_z'],params['boundary_z'],params['dilated'],device=dev,
                                                        min_att=params['min_att'],max_att=params['max_att'],mean_att=params['mean_att'],att_stddev=params['att_stddev'])
    cases, is_ncct = tu.get_cases(slicedataset)

    # More Initialization
    loss_fnc = nn.CrossEntropyLoss().to(dev)

    for split in splits:
        cv_results = []
        split_path = os.path.join(save_dir,'split_{}'.format(split))
        if not os.path.exists(split_path):
            os.mkdir(split_path)

        split_fp = os.path.join(split_path,'split.npy')
        five_fold_strat = kfold_strat(n_splits=folds, shuffle=True)

        if os.path.exists(split_fp): fold_split = np.load(split_fp,allow_pickle=True)
        else:
            fold_split = np.array([(fold,tr_index,ts_index) for fold,(tr_index, ts_index) in enumerate(five_fold_strat.split(cases,is_ncct))],dtype=object)
            np.save(os.path.join(split_path,split_fp),fold_split,allow_pickle=True)

        for fold,train_index, test_index in fold_split:
            fold_path = os.path.join(split_path,'fold_{}'.format(fold))
            slice_path = os.path.join(fold_path,model_type)

            if not os.path.exists(fold_path):os.mkdir(fold_path)
            if not os.path.exists(slice_path):os.mkdir(slice_path)

            model = LinearClassifier(3)
            model.feature_extractor.load_state_dict(torch.load(path_to_pretrained_model))
            model = model.to(dev)

            # turn all model.feature_extractor parameters to non-trainable
            for param in model.feature_extractor.parameters():
                param.requires_grad = False

            opt = torch.optim.Adam(model.fc.parameters(),lr=params['lr'])

            dl,test_dl = tu.generate_dataloaders(slicedataset,test_slicedataset,cases[train_index],params['batch_size'])

            if not os.path.exists(os.path.join(slice_path,'model')):
                os.mkdir(os.path.join(slice_path,'model'))
                os.mkdir(os.path.join(slice_path,'csv'))
            model_name = 'finetune_DA_{}_{}_{}_{}_{}_{}'.format(type,model_type,params['model_size'],params['epochs'],params['epochs'],params['lr'])

            model = tu.train_model(dl,dev,params['epochs'],loss_fnc,opt,model,model_name,slice_path,params['lr'])



            torch.save(model,os.path.join(slice_path,'model',model_name))

            model.eval()
            model_res,test_df = eval_.eval_cnn(model,test_dl,dev=dev)
            cv_results.append(test_df)
            model_res.to_csv(os.path.join(slice_path,'csv',model_name+'.csv'))

        CV_results = pd.concat(cv_results, axis=0, ignore_index=True)
        model_ROC = eval_.ROC_func(CV_results['Top-{}'.format(params['pred_window'])],CV_results['label'],max_pred=params['pred_window'],intervals=1000)
        np.save(os.path.join(split_path, '{}_ROC_'.format(model_type)+model_name), model_ROC)

        fig = plt.figure(figsize=(8, 6))
        tu.plot_roc('{}'.format(model_type),model_name,model_ROC)
        plt.legend()
        plt.savefig(os.path.join(split_path, '{}_ROC_'.format(model_type)+model_name+'.png'))
        plt.show()
        plt.close()

def finetune_SSL_model(home = '/Users/mcgoug01/Downloads/Data/',
                         dataname='coreg_ncct',type='normal',
                         splits:list=[0],folds=5,params:dict=None):
    # Suppress warnings
    warnings.filterwarnings("ignore") #makes dgl stop complaining!

    # Initialization
    dev = tu.initialize_device()

    if params==None:params = tu.init_domainadaptation(type=type)
    else:tu.check_params(params,tu.init_domainadaptation(type=type))
    model_type = 'PatchModel'

    save_dir = tu.init_training_home(home, dataname)
    print(save_dir)

    slicedataset, test_slicedataset = tu.get_slice_data_DA(home, dataname, params['voxel_size'], params['cancthresh_r_mm'],params['kidthresh_r_mm'],params['depth_z'],params['boundary_z'],params['dilated'],device=dev,
                                                        min_att=params['min_att'],max_att=params['max_att'],mean_att=params['mean_att'],att_stddev=params['att_stddev'])
    cases, is_ncct = tu.get_cases(slicedataset)

    # More Initialization
    loss_fnc = nn.CrossEntropyLoss().to(dev)

    for split in splits:
        cv_results = []
        split_path = os.path.join(save_dir,'split_{}'.format(split))
        if not os.path.exists(split_path):
            os.mkdir(split_path)

        split_fp = os.path.join(split_path,'split.npy')
        five_fold_strat = kfold_strat(n_splits=folds, shuffle=True)

        if os.path.exists(split_fp): fold_split = np.load(split_fp,allow_pickle=True)
        else:
            fold_split = np.array([(fold,tr_index,ts_index) for fold,(tr_index, ts_index) in enumerate(five_fold_strat.split(cases,is_ncct))],dtype=object)
            np.save(os.path.join(split_path,split_fp),fold_split,allow_pickle=True)

        for fold,train_index, test_index in fold_split:
            fold_path = os.path.join(split_path,'fold_{}'.format(fold))
            slice_path = os.path.join(fold_path,model_type)

            if not os.path.exists(fold_path):os.mkdir(fold_path)
            if not os.path.exists(slice_path):os.mkdir(slice_path)

            model = LinearClassifier(3).to(dev)

            # turn all model.feature_extractor parameters to non-trainable
            for param in model.feature_extractor.parameters():
                param.requires_grad = False

            if not os.path.exists(os.path.join(slice_path,'model')):
                os.mkdir(os.path.join(slice_path,'model'))
                os.mkdir(os.path.join(slice_path,'csv'))

            opt = torch.optim.Adam(model.fc.parameters(),lr=params['lr'])

            dl,test_dl = tu.generate_dataloaders(slicedataset,test_slicedataset,cases[train_index],params['batch_size'])
            model_name = 'finetune_SSL_{}_{}_{}_{}_{}_{}'.format(type,model_type,params['model_size'],params['epochs'],params['epochs'],params['lr'])
            model = tu.train_model(dl,dev,params['epochs'],loss_fnc,opt,model,model_name,slice_path,params['lr'])
            torch.save(model,os.path.join(slice_path,'model',model_name))

            model.eval()
            model_res,test_df = eval_.eval_cnn(model,test_dl,dev=dev)
            cv_results.append(test_df)
            model_res.to_csv(os.path.join(slice_path,'csv',model_name+'.csv'))

        CV_results = pd.concat(cv_results, axis=0, ignore_index=True)
        model_ROC = eval_.ROC_func(CV_results['Top-{}'.format(params['pred_window'])],CV_results['label'],max_pred=params['pred_window'],intervals=1000)
        np.save(os.path.join(split_path, '{}_ROC_'.format(model_type)+model_name), model_ROC)

        fig = plt.figure(figsize=(8, 6))
        tu.plot_roc('{}'.format(model_type),model_name,model_ROC)
        plt.legend()
        plt.savefig(os.path.join(split_path, '{}_ROC_'.format(model_type)+model_name+'.png'))
        plt.show()
        plt.close()

if __name__ == '__main__':
    import sys

    is_da = sys.argv[1].lower() == 'true'
    type = str(sys.argv[2])

    dataset = 'coreg_ncct'
    path_to_pretrained_model = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/ne2ceCT/resnet_tciaandcoltea_1_cancer/l1-0.2_l2-1.0_var0.01_cov0.2_cce0.0_cos1.0_clip0.0_2e-05lr_0.99998lg_80bs_5000ep_4gpus/network_weights'
    home = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/3DSlice_data'
    if is_da:
        finetune_da_model(path_to_pretrained_model,home=home,dataname=dataset,splits=[0,1,2],
                          type=type)
    else:
        finetune_SSL_model(home=home,dataname=dataset,splits=[0,1,2],
                          type=type)

