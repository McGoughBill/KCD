import pandas as pd
import os

save_home = '/bask/projects/p/phwq4930-renal-canc/data/TCIA/CT/'
collections = [folder for folder in os.listdir(save_home) if os.path.isdir(save_home + '/' + folder)]
pp_dir = '/bask/projects/p/phwq4930-renal-canc/data/TCIA/tcia_PP'
used = [file for file in os.listdir(pp_dir) if file.endswith('.npy') and not file.endswith('_noised.npy')]

# tutorial on how to use the NBIA API: https://github.com/kirbyju/TCIA_Notebooks/blob/main/TCIA_REST_API_Downloads.ipynb

# subject_ID - patientID - person
#series - date for a specific person
#study - specific scan taken on a specific study (on a specific date for a specific patient)

#I think we only want to download one study per patient. Maybe we can allow multiple studies, if they are
#seperated by more than 10 years.

count_dict_list = []

total_patient_count = 0
study_count = 0
for col in collections:
    count = 0
    patients = [folder for folder in os.listdir(save_home + '/' + col) if os.path.isdir(save_home + '/' + col + '/' + folder)]
    for pat in patients:
        #count number of folders in each patient folder - add to study count
        studies = [folder for folder in os.listdir(save_home + '/' + col + '/' + pat) if os.path.isdir(save_home + '/' + col + '/' + pat + '/' + folder)]
        study_count += len(studies)
    total_patient_count += len(patients)
    #check if each patients is in the used list, if it is, print the patient ID and col

    for patient in patients:
        for pat_used in used:
            if pat_used.startswith(patient):
                count +=1
                print(patient,col,count)
                break
    count_dict_list.append({'collection':col,'count':count})

total_count = 0
for item in count_dict_list:
    if item['count'] > 0:
        total_count += item['count']
        print(item)
print(total_count)
print(total_patient_count)
print(study_count)

