import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
plt.switch_backend('TkAgg')

home = '/Users/mcgoug01/Dropbox/AIvsRad/Bill/QA'
hania_response = pd.read_csv(home + '/hania_qa.csv')
hania_response['StudyID'] = hania_response['StudyID'].astype(str).str[7:].astype(int)
hania_response['What type of image is this?'] = hania_response['What type of image is this?'].astype(str)
hania_response['Quality? (1-5)'] = hania_response['Quality? (1-5)'].astype(float)
labels = pd.read_csv(home + '/metadata.csv')
#change labels 'study_id' to 'StudyID'
labels = labels.rename(columns={'study_id': 'StudyID'})
labels['StudyID'] = labels['StudyID'].astype(int)

# Merge the two dataframes
df = pd.merge(hania_response, labels, on='StudyID', how='outer')
df = df.rename(columns={'original_filepath': 'label',
                        'What type of image is this?':'pred',
                        'Quality? (1-5)':'quality'}).astype(str)
# change label to 0 if contains '/nc/' in string, 1 if contains 'pc_sldct' in string,
# and 2 else
# 0 for full dose nc, 1 for synthetic low dose, 2 for real low dose
df['label_class'] = df['label'].apply(lambda x: 0 if '/nc/' in x else (1 if 'pc_sldct' in x else 2))
df['pred_class'] = df['pred'].apply(lambda x: 1 if 'synthetic' in x else (2 if 'real' in x else 0))

# confusion matrix
cm = confusion_matrix(df['label_class'], df['pred_class'])

#create 3x3 confusion matrix that excludes the 80pc_sldct and then the 90pc_sldct
df_80 = df[~df['label'].str.contains('90pc_sldct')]
df_90 = df[~df['label'].str.contains('80pc_sldct')]

cm_80 = confusion_matrix(df_80['label_class'], df_80['pred_class'])
cm_90 = confusion_matrix(df_90['label_class'], df_90['pred_class'])

# print the sensitivity and specificity of the [1,1] position in each 3x3 confusion matrix
print('Confusion Matrix for 80pc_sldct')
print('Sensitivity:', cm_80[1,1]/(cm_80[1,1]+cm_80[1,0]+cm_80[1,2]))
print(cm_80[1,1], cm_80[1,0], cm_80[1,2])
print('Confusion Matrix for 90pc_sldct')
print('Sensitivity:', cm_90[1,1]/(cm_90[1,1]+cm_90[1,0]+cm_90[1,2]))
print(cm_90[1,1], cm_90[1,0], cm_90[1,2])

fig, ax = plt.subplots(1 , 2   , figsize=(12, 5))
sns.heatmap(cm_80, annot=True, ax=ax[0], fmt='d')
ax[0].set_title('Confusion Matrix for 80pc_sldct')
sns.heatmap(cm_90, annot=True, ax=ax[1], fmt='d')
ax[1].set_title('Confusion Matrix for 90pc_sldct')
#add axis titles and labels
for i in range(2):
    ax[i].set_xlabel('Predicted')
    ax[i].set_ylabel('True')
    ax[i].set_xticklabels(['Full Dose', 'Synthetic Low Dose', 'Real Low Dose'])
    ax[i].set_yticklabels(['Full Dose', 'Synthetic Low Dose', 'Real Low Dose'])

# plot whisker plot of quality scores for each of: full dose, 80pc_sldct, 90pc_sldct, real low dose
df['label_class_q'] = df['label'].apply(lambda x: 'full' if '/nc/' in x else ('80' if '80pc_sldct' in x else ('90' if '90pc_sldct' in x else 'real low')))

#print df label_class_q and quality and mean and average quality scores for each of the 4 types of images
df['quality'] = df['quality'].astype(float)
print(df[['label_class_q', 'quality']])
print(df.groupby('label_class_q')['quality'].mean())
print(df.groupby('label_class_q')['quality'].std())
fig, ax = plt.subplots(1, 1, figsize=(12, 5))
sns.boxplot(x='label_class_q', y='quality', data=df)
ax.set_xticklabels(['Full Dose', '80% Synthetic Low Dose','90% Synthetic Low Dose', 'Real Low Dose'])
ax.set_xlabel('Type of Image')
ax.set_ylabel('Quality Score')
plt.show()



plt.show()






