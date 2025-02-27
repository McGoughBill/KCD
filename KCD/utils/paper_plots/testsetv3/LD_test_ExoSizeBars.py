import os
import pandas as pd
import numpy as np
import matplotlib.ticker as plticker
import matplotlib.patches as mpatches

meta_folder = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/AI/predictions_v3/lowdose_test_ROCv3'
sub_folders = [os.path.join(meta_folder, f,'predictions.csv') for f in os.listdir(meta_folder) if os.path.isdir(os.path.join(meta_folder, f))]
names = [float(f.split('/')[-2]) for f in sub_folders]
save_loc = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/nat_paper'
# sort sub_folders by names_sub_folders
folders = [x for _,x in sorted(zip(names,sub_folders))]
names_sub_folders = sorted(names)

truth_csv_fp = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/Bill/LD/internaltest_metadata.csv'
exo_fp = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/Bill/FD/internal_exo_size.csv'
truth = pd.read_csv(truth_csv_fp)
exo_etc = pd.read_csv(exo_fp)

# merge the exo_etc with the truth based on OriginalID in truth and case in exo_etc
truth = truth.merge(exo_etc, left_on='OriginalID', right_on='case', how='outer')
# where size or exo_frac is nan, set to 0
truth['size'] = truth['size'].fillna(0)
truth['exo_frac'] = truth['exo_frac'].fillna(0)
#print number of large cases (over 14137) and small cases (under 14137)


truth['large_exo'] = (truth[truth['size']>14137]['exo_frac']>0.5).astype(int)
truth['large_endo'] = (truth[truth['size']>14137]['exo_frac']<=0.5).astype(int)
truth['small_exo'] = (truth[(truth['size']<=14137)&(truth['size']>0)]['exo_frac']>0.5).astype(int)
truth['small_endo'] = (truth[(truth['size']<=14137)&(truth['size']>0)]['exo_frac']<=0.5).astype(int)

print(truth[truth['large_exo']==1]['OriginalID'])
# fill nans with 0
truth['large_exo'] = truth['large_exo'].fillna(0)
truth['large_endo'] = truth['large_endo'].fillna(0)
truth['small_exo'] = truth['small_exo'].fillna(0)
truth['small_endo'] = truth['small_endo'].fillna(0)

#print number of large exo, large endo, small exo, small endo
print(truth['large_exo'].value_counts())
print(truth['large_endo'].value_counts())
print(truth['small_exo'].value_counts())
print(truth['small_endo'].value_counts())

print()


#sortby StudyID
truth = truth.sort_values(by='StudyID')
# count the number of each type of lesion
# print(truth['large_exo'].value_counts())
# print(truth['large_endo'].value_counts())
# print(truth['small_exo'].value_counts())
# print(truth['small_endo'].value_counts())
# print(truth.head(10))
# assert False

#plot a 2x2 fig with 4 bar charts
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
fig, axs = plt.subplots(2,2,figsize=(10,10))

for i,column in enumerate(['large_exo','small_exo','large_endo','small_endo']):
    print(column)
    column_truth = truth[truth[column] == 1]
    column_truth= column_truth[['OriginalID','RightCancer','LeftCancer']]
    column_truth = column_truth.sort_values(by='OriginalID')
    results = []
    for name,pred_csv_fp in zip(names_sub_folders,folders):
        if name == 0.120:
            op = 'High Sensitivity'
        elif name == 0.550:
            op = 'High Accuracy'
        elif name == 0.994:
            op = 'High Specificity'
        else:
            continue
        print(name)

        # pred and columns have 'RightCancer', 'LeftCancer' columns - trust has answers, pred has predictions
        #calc sens and spec
        pred = pd.read_csv(pred_csv_fp)
        pred = pred.sort_values(by='OriginalID')

        #create column pred according to where OriginalID in pred is in column_truth
        column_pred = pred[pred['OriginalID'].isin(column_truth['OriginalID'])]

        # get the truth values
        truth_right = column_truth['RightCancer'].values
        truth_left = column_truth['LeftCancer'].values

        # get the predictions
        pred_right = column_pred['RightCancer'].values
        pred_left = column_pred['LeftCancer'].values

        # get the number of patients
        num_patients = len(truth_right)

        # get the number of true positives, true negatives, false positives, false negatives
        tp_right = np.sum(np.logical_and(truth_right, pred_right))
        tn_right = np.sum(np.logical_and(np.logical_not(truth_right), np.logical_not(pred_right)))
        fp_right = np.sum(np.logical_and(np.logical_not(truth_right), pred_right))
        fn_right = np.sum(np.logical_and(truth_right, np.logical_not(pred_right)))

        tp_left = np.sum(np.logical_and(truth_left, pred_left))
        tn_left = np.sum(np.logical_and(np.logical_not(truth_left), np.logical_not(pred_left)))
        fp_left = np.sum(np.logical_and(np.logical_not(truth_left), pred_left))
        fn_left = np.sum(np.logical_and(truth_left, np.logical_not(pred_left)))

        #print dataframe rows for all false negatives
        fn_right_indices = np.where(np.logical_and(truth_right, np.logical_not(pred_right)))[0]
        fn_left_indices = np.where(np.logical_and(truth_left, np.logical_not(pred_left)))[0]


        tp = tp_right + tp_left
        tn = tn_right + tn_left
        fp = fp_right + fp_left
        fn = fn_right + fn_left

        print(tp,tn,fp,fn)

        sens = 100*tp/(tp+fn)
        spec = 100*tn/(tn+fp)
        print(f'{name}:')
        #print sens and spec to 2dp
        print(f'Sensitivity: {sens:.2f}%')
        print(f'Specificity: {spec:.2f}%')
        print()
        entry = {'name':name,'sens':sens,'spec':spec,'op':op}

        results.append(entry)

    #plot bar chart using results with sensitivity and specificity next to each other and spacing between each operating point
    df = pd.DataFrame(results)
    df = df.sort_values(by='name')
    ax = axs[i//2,i%2]
    x_index = np.arange(len(df['name']))
    width = 0.35
    ax.bar(x_index,df['sens'],width,label='Sensitivity',color='lightgreen')
    ax.bar(x_index+width,df['spec'],width,label='Specificity',color='darkgreen')
    ax.set_xticks(x_index + width / 2)
    ax.set_xticklabels(df['op'])
    ax.spines[['right', 'top']].set_visible(False)

    ax.set_ylim([0,100])
    # set font to helvetica
    ax.set_xlabel('Operating Point',fontsize=20,fontname='Helvetica')
    ax.set_ylabel('Percentage',fontsize=20,fontname='Helvetica')
    ax.set_aspect("auto")

#all subplots have the same legend - make this appear once, outside of the subplots
fig.legend(['Sensitivity','Specificity'],loc='upper right',fontsize=16,framealpha=0.9)
plt.savefig(os.path.join(save_loc,'exo_endo_sens_spec.png'))
plt.show()

high_acc_pred_fp = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/AI/predictions_v3/lowdose_test_ROCv3/0.550/predictions.csv'
high_acc_pred = pd.read_csv(high_acc_pred_fp)
high_acc_pred = high_acc_pred.sort_values(by='OriginalID')
truth = truth.sort_values(by='OriginalID')

# use this to get the signficance of size and exo_frac on the predictions for cancer case
# fill nans with 0
print()
truth = truth[truth['size']>0]
#filter high_acc_pred to only include cases where size is greater than 0
high_acc_pred = high_acc_pred[high_acc_pred['OriginalID'].isin(truth['OriginalID'])]
print(high_acc_pred.head(5))
print(truth.head(5))

high_acc_pred['true_pos_right'] = np.logical_and(high_acc_pred['RightCancer']==truth['RightCancer'].values,truth['RightCancer'].values)
high_acc_pred['true_pos_left'] = np.logical_and(high_acc_pred['LeftCancer']==truth['LeftCancer'].values,truth['LeftCancer'].values)
high_acc_pred['false_neg_right'] = np.logical_and(high_acc_pred['RightCancer']!=truth['RightCancer'].values,truth['RightCancer'].values)
high_acc_pred['false_neg_left'] = np.logical_and(high_acc_pred['LeftCancer']!=truth['LeftCancer'].values,truth['LeftCancer'].values)
# remove all rows where size is 0
print(len(high_acc_pred),len(truth))
high_acc_pred = high_acc_pred[(truth['size']>0).values]

#calculate significance of size and exo_frac on true_pos
size_true_pos = truth[(high_acc_pred['true_pos_right']).values]['size'].values.tolist() + truth[(high_acc_pred['true_pos_left']).values]['size'].values.tolist()
size_false_neg = truth[(high_acc_pred['false_neg_right']).values]['size'].values.tolist() + truth[(high_acc_pred['false_neg_left']).values]['size'].values.tolist()
exo_true_pos = truth[(high_acc_pred['true_pos_right']).values]['exo_frac'].values.tolist() + truth[(high_acc_pred['true_pos_left']).values]['exo_frac'].values.tolist()
exo_false_neg = truth[(high_acc_pred['false_neg_right']).values]['exo_frac'].values.tolist() + truth[(high_acc_pred['false_neg_left']).values]['exo_frac'].values.tolist()

X1 = np.array(size_true_pos + size_false_neg)
diameter_X1 = np.array([((3*x)/(4*np.pi))**(1/3) for x in X1])
mean_diam,std_diam = np.mean(diameter_X1),np.std(diameter_X1)
# normalise X1
X2 = np.array(exo_true_pos + exo_false_neg)
mean_exo,std_exo = np.mean(X2),np.std(X2)
X = np.column_stack((X1,X2))
Y = np.array([1]*len(size_true_pos) + [0]*len(size_false_neg))

#make cmap from hex colour values
import matplotlib.colors as mcolors
cmap = mcolors.ListedColormap(['#332288', '#88CCEE'])

fig = plt.figure(figsize=(6,6))
# plot size vs exo_frac with detection as colour
plt.scatter(diameter_X1*2,X2,c=Y,cmap=cmap,s=40)
plt.xlabel('Diameter / mm',fontsize=24,fontname='Helvetica')
plt.ylabel('Exophytic Fraction',fontsize=24,fontname='Helvetica')
# set splines to be invisible like ax.spines[['right', 'top']].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)


#plot legend to show detected / not detected
detected = mpatches.Patch(color='#332288',label='Not Detected')
not_detected = mpatches.Patch(color='#88CCEE',label='Detected')
plt.legend(handles=[detected,not_detected],loc = 'lower right',fontsize=16,framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(save_loc,'size_vs_exo_frac.png'))

plt.show()

