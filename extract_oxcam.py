# read oxcam_df and use ARTIDs to guide the copying of files from source to destination
# source dir contains sub_dir with ARTID as name. Copy the whole sub_dir to destination

import os
import pandas as pd
import shutil

# read oxcam_df
oxcam_df = pd.read_excel('/rds/project/rds-sDuALntK11g/oxcam.xlsx')
#adjust ARTID column to string, and also convert ARTXXXX to ARTIST_XXX format
oxcam_df['ARTID'] = oxcam_df['ARTID'].astype(str)
oxcam_df['OXCAMID'] = oxcam_df['OXCAMID'].astype(str)
oxcam_df['ARTID'] = oxcam_df['ARTID'].apply(lambda x: 'ARTIST' + x[4:])
print(oxcam_df.head())

# source and destination directories
source_dir = '/rds/project/rds-sDuALntK11g/IMAGING_nifti'
dest_dir = '/rds/project/rds-sDuALntK11g/oxcam_nifti'

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# copy files from source to destination
for index, row in oxcam_df.iterrows():
    source_subdir = os.path.join(source_dir, row['ARTID'])
    dest_subdir = os.path.join(dest_dir, 'OXCAM'+row['OXCAMID'])
    if os.path.exists(source_subdir):
        shutil.copytree(source_subdir, dest_subdir)
    else:
        print('Source subdir does not exist:', source_subdir)