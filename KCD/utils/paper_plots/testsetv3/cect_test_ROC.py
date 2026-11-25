import os
import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


small = False

meta_folder = '/path/to/kcd_data/Cambridge University Dropbox/William McGough/AIvsRad/AI/predictions_v3/cect_test_ROCv3'
sub_folders = [os.path.join(meta_folder, f,'predictions.csv') for f in os.listdir(meta_folder) if os.path.isdir(os.path.join(meta_folder, f))]
names_sub_folders = [float(f.split('/')[-2]) for f in sub_folders]
seg_loc = '/path/to/kcd_data/raw_data/test_set_cect/labels'
save_loc = '/path/to/kcd_data/Cambridge University Dropbox/William McGough/nat_paper'
save_name = 'cect_test_ROC_small={}'.format( small)
save_fp = os.path.join(save_loc,save_name)
# sort sub_folders by names_sub_folders
sub_folders = [x for _,x in sorted(zip(names_sub_folders,sub_folders))]
names_sub_folders = sorted(names_sub_folders)


truth_csv_fp = '/path/to/kcd_data/Cambridge University Dropbox/William McGough/AIvsRad/Bill/FD/internaltest_metadata_cect.csv'
iztok_predfulldose_csv_fp = '/path/to/kcd_data/Cambridge University Dropbox/William McGough/AIvsRad/iztokFD.csv'
hania_predfulldose_csv_fp = '/path/to/kcd_data/Cambridge University Dropbox/William McGough/AIvsRad/haniaFD.csv'
cathal_predfulldose_csv_fp = '/path/to/kcd_data/Cambridge University Dropbox/William McGough/AIvsRad/cathalFD.csv'

folders = [iztok_predfulldose_csv_fp, hania_predfulldose_csv_fp,cathal_predfulldose_csv_fp] + sub_folders
names = ['Iztok','Hania','Cathal'] + names_sub_folders
humans = ['Iztok','Hania','Cathal']

high_acc_lowdose_fp = '/path/to/kcd_data/Cambridge University Dropbox/William McGough/AIvsRad/AI/predictions_v3/lowdose_test_ROCv3/0.550/predictions.csv'
low_dose_predictions = pd.read_csv(high_acc_lowdose_fp)
high_acc_fulldose_fp = '/path/to/kcd_data/Cambridge University Dropbox/William McGough/AIvsRad/AI/predictions_v3/fulldose_test_ROCv3/0.550/predictions.csv'
full_dose_predictions = pd.read_csv(high_acc_fulldose_fp)

# using full_dose_predictions, make a key of StudyID to OriginalID - dictionary
ID_key = full_dose_predictions[['StudyID','OriginalID']].copy()

ID_key['OriginalID'] = ID_key['OriginalID'].str.replace('RCC_','Rcc_')
ID_key['OriginalID'] = ID_key['OriginalID'].str.replace('KiTS-','case_')
ID_key = ID_key.drop_duplicates()
ID_key = ID_key.set_index('StudyID')
ID_key = ID_key.to_dict()['OriginalID']
print(ID_key)

# for low_dose and full_dose predictions, remove StudyID column, rename OriginalID if starts with 'KiTS-' to 'case_',
low_dose_predictions = low_dose_predictions.drop(columns=['StudyID'])
low_dose_predictions['OriginalID'] = low_dose_predictions['OriginalID'].apply(lambda x: 'case_' + x.split('-')[-1] if x.startswith('KiTS-') else x)
full_dose_predictions = full_dose_predictions.drop(columns=['StudyID'])
full_dose_predictions['OriginalID'] = full_dose_predictions['OriginalID'].apply(lambda x: 'case_' + x.split('-')[-1] if x.startswith('KiTS-') else x)

# for low_dose and full_dose predictions,rename OriginalID if starts with RCC to Rcc
low_dose_predictions['OriginalID'] = low_dose_predictions['OriginalID'].apply(lambda x: 'Rcc_' + x.split('_')[-1] if x.startswith('RCC') else x)
full_dose_predictions['OriginalID'] = full_dose_predictions['OriginalID'].apply(lambda x: 'Rcc_' + x.split('_')[-1] if x.startswith('RCC') else x)

#in both low_dose and full_dose predictions, remove all rows where OriginalID's first two characters are numeric
low_dose_predictions = low_dose_predictions[~low_dose_predictions['OriginalID'].str[:2].str.isnumeric()]
full_dose_predictions = full_dose_predictions[~full_dose_predictions['OriginalID'].str[:2].str.isnumeric()]

#sort both by OriginalID
low_dose_predictions = low_dose_predictions.sort_values(by='OriginalID', ascending=True)
full_dose_predictions = full_dose_predictions.sort_values(by='OriginalID', ascending=True)
tr = pd.read_csv(truth_csv_fp)
sizes_dictlist = []
plt.switch_backend('TkAgg')

fig = plt.figure(figsize=(6,6))


for high_acc_conditioning  in [False, True]:
    ai_results = [[100, 0]]
    large_cases = []
    pile = []
    for ogid  in tr['OriginalID']:
        # if case.lower does not begin with kits or rcc, skip
        if not ogid.lower().startswith('kits') and not ogid.lower().startswith('rcc'):
            continue

        # if case begins with kits in seg_loc
        if ogid.lower().startswith('case'):
            ogid = ogid.lower().replace('kits-','KiTS-')
            seg_n = nib.load(os.path.join(seg_loc,ogid))
            seg = (seg_n.get_fdata()==2).astype(np.uint8)
            seg = np.swapaxes(seg,0,2)
            vox_vol = np.prod(seg_n.header.get_zooms())
            size = np.sum(seg) * vox_vol
            if np.sum(seg)*vox_vol > 33510.32:
                # append to large cases
                large_cases.append(ogid)
            else:
                pile.append(np.sum(seg)*vox_vol)
        # if case begins with rcc in seg_loc
        elif ogid.lower().startswith('rcc'):
            ogid = ogid.lower().replace('rcc_','Rcc_')
            seg_n = nib.load(os.path.join(seg_loc,ogid))
            seg = (seg_n.get_fdata()==2).astype(np.uint8)
            vox_vol = np.prod(seg_n.header.get_zooms())
            size = np.sum(seg) * vox_vol
            if np.sum(seg)*vox_vol > 33510.32:
                # append to large cases
                large_cases.append(ogid)
            else:
                pile.append(np.sum(seg)*vox_vol)
        else:
            size = 0

        sizes_dictlist.append({'originalid':ogid,'size':size})

    print('mean small size is',np.mean(pile))

    size_df = pd.DataFrame(sizes_dictlist)
    size_df.to_csv('/path/to/kcd_data/Cambridge University Dropbox/William McGough/AIvsRad/Bill/internaltest_sizes.csv',index=False)

    for name,pred_csv_fp in zip(names,folders):

        # pred and columns have 'RightCancer', 'LeftCancer' columns - trust has answers, pred has predictions
        #calc sens and spec
        pred = pd.read_csv(pred_csv_fp)
        truth = pd.read_csv(truth_csv_fp)
        truth = truth.sort_values(by='OriginalID', ascending=True)

        #sort by OriginalID and ID
        if name in humans:
            # create an ID column for humans using ID_key
            pred['ID'] = pred['StudyID'].apply(lambda x: ID_key[x] if x in ID_key else x)
            pred = pred[pred['ID'].isin(truth['OriginalID'])]

        #sort by OriginalID and ID
        pred = pred.sort_values(by='ID', ascending=True)

        #remove all rows in pred and truth where the StudyID is in large_cases
        if small:
            pred = pred[~pred['ID'].isin(large_cases)]
            truth = truth[~truth['OriginalID'].isin(large_cases)]
            low_dose_predictions = low_dose_predictions[~low_dose_predictions['OriginalID'].isin(large_cases)]
            full_dose_predictions = full_dose_predictions[~full_dose_predictions['OriginalID'].isin(large_cases)]


        # get the truth values
        truth_right = truth['RightCancer'].values
        truth_left = truth['LeftCancer'].values

        # get the predictions
        pred_right = pred['RightCancer'].values
        pred_left = pred['LeftCancer'].values

        # get the ncct predictions
        ncct_right = low_dose_predictions['RightCancer'].values | full_dose_predictions['RightCancer'].values
        ncct_left = low_dose_predictions['LeftCancer'].values | full_dose_predictions['LeftCancer'].values

        # get the number of patients
        num_patients = len(truth_right)

        # get the number of true positives, true negatives, false positives, false negatives
        if high_acc_conditioning:
            tp_right = np.sum(np.logical_and(np.logical_and(truth_right, pred_right), ncct_right))
            tn_right = np.sum(np.logical_and(np.logical_and(np.logical_not(truth_right), np.logical_not(pred_right)), ncct_right))
            fp_right = np.sum(np.logical_and(np.logical_and(np.logical_not(truth_right), pred_right), ncct_right))
            fn_right = np.sum(np.logical_and(np.logical_and(truth_right, np.logical_not(pred_right)), ncct_right))

            tp_left = np.sum(np.logical_and(np.logical_and(truth_left, pred_left), ncct_left))
            tn_left = np.sum(np.logical_and(np.logical_and(np.logical_not(truth_left), np.logical_not(pred_left)), ncct_left))
            fp_left = np.sum(np.logical_and(np.logical_and(np.logical_not(truth_left), pred_left), ncct_left))
            fn_left = np.sum(np.logical_and(np.logical_and(truth_left, np.logical_not(pred_left)), ncct_left))
        else:
            tp_right = np.sum(np.logical_and(truth_right, pred_right))
            tn_right = np.sum(np.logical_and(np.logical_not(truth_right), np.logical_not(pred_right)))
            fp_right = np.sum(np.logical_and(np.logical_not(truth_right), pred_right))
            fn_right = np.sum(np.logical_and(truth_right, np.logical_not(pred_right)))

            tp_left = np.sum(np.logical_and(truth_left, pred_left))
            tn_left = np.sum(np.logical_and(np.logical_not(truth_left), np.logical_not(pred_left)))
            fp_left = np.sum(np.logical_and(np.logical_not(truth_left), pred_left))
            fn_left = np.sum(np.logical_and(truth_left, np.logical_not(pred_left)))

        tp = tp_right + tp_left
        tn = tn_right + tn_left
        fp = fp_right + fp_left
        fn = fn_right + fn_left

        sens = 100*tp/(tp+fn)
        spec = 100*tn/(tn+fp)
        print(f'{name}:')
        #print sens and spec to 2dp
        print(f'Sensitivity: {sens:.2f}%')
        print(f'Specificity: {spec:.2f}%')
        print()

        if name in humans:
            if name == 'Iztok':
                iztok = [sens,spec]
                print('Iztok',sens,spec)
            elif name == 'Hania':
                hania = [sens,spec]
            elif name == 'Cathal':
                cathal = [sens,spec]
        else:
            if name == 0.0089:
                high_sens = [sens,spec]
            elif name == 0.150:
                high_acc = [sens,spec]
            elif name == 0.365:
                high_spec = [sens,spec]
            elif name == 0.3:
                non_inferior = [sens,spec]
            ai_results.append([sens,spec])

    ai_results.append([0,100])


    ai_results = np.array(ai_results)
    # sort by ascending first column and descending second column - merge sort by second column first
    ai_results = ai_results[ai_results[:,1].argsort(kind='mergesort')[::-1]]
    ai_results = ai_results[ai_results[:,0].argsort(kind='mergesort')]


    #calculate AUC for ai_results
    auc = np.abs(np.trapz(ai_results[:,0],ai_results[:,1])/10000)
    print(f'AUC: {auc:.3f}')
    print(ai_results)

    #make marker star
    marker = '*'

    #plot ROC curve, and plot the point for Iztok and Cathal
    # also, plot the operating points for the AI models (high sensitivity, high accuracy, high specificity
    if high_acc_conditioning:
        label = 'Conditioned CECT (AUC = {:.3f})'.format(auc)
        linestyle = '-.'
    else:
        label = 'CECT (AUC = {:.3f})'.format(auc)
        linestyle = 'solid'
    plt.plot(ai_results[:,1],ai_results[:,0],label=label,c='k', linewidth=2, linestyle=linestyle)

plt.xlabel('Specificity (%)',fontdict={'fontsize':26})
plt.ylabel('Sensitivity (%)',fontdict={'fontsize':26})
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.legend(fontsize=16)
plt.xlim([0,100.5])
plt.ylim([0,100.5])
plt.tight_layout()

print(high_sens)
print(high_acc)
print(high_spec)
print(non_inferior)
plt.savefig(save_fp + '.png')
plt.show()

