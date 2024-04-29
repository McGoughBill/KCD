import pandas as pd
import numpy as np
import os
os.environ['OV_DATA_BASE'] ='/bask/projects/p/phwq4930-renal-canc/data/seg_data'

home = os.path.join(os.environ['OV_DATA_BASE'], '3DSlice_data','training_info','coreg_ncct')
splits = [subdir for subdir in os.listdir(home) if os.path.isdir(subdir)]
combined_res = []
for subdir in splits:
    split = os.path.join(home, subdir)
    folds = [subdir for subdir in os.listdir(split) if os.path.isdir(subdir)]
    for fold in folds:
        fold_path = os.path.join(split, fold,'PatchModel','csv')
        files = [f for f in os.listdir(fold_path) if f.endswith('.csv')]
        names = [f.split('_')[1:3] for f in files]
        for file,(modeltype,preprocessing) in zip(files,names):
            res = pd.read_csv(os.path.join(fold_path,file),index_col=0,header=0)
            AUC = res[(res['dataset_loc']=='test') & (res['Voting Size']==1)]['AUC'].values[0]
            combined_res.append({'modeltype':modeltype,
                                 'preprocessing':preprocessing,
                                 'AUC':AUC})

# group and average where modeltype and preprocessing are the same
df = pd.DataFrame(combined_res)
print(df.head())
df = df.groupby(['modeltype','preprocessing']).mean().reset_index()
df.to_csv(os.path.join(home,'combined_results.csv'),index=False)




