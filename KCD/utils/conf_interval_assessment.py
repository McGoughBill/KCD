import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from skimage.measure import label


home = '/Users/mcgoug01/Downloads/conf_intv'
preds = os.path.join(home, 'preds')
labels = os.path.join(home, 'labels')

confidences = [0.5]

results = []

files = [f for f in os.listdir(preds) if f.endswith('.nii.gz')]
for conf in confidences:
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for file in files:
        print(file)

        label_nib = nib.load(os.path.join(labels, file.replace('_0.','_0.7.')))
        pred_nib = nib.load(os.path.join(preds, file))
        seg = label_nib.get_fdata()
        pred = pred_nib.get_fdata()
        voxvol = np.prod(label_nib.header.get_zooms())

        label_foreground = (seg ==2).astype(int)
        pred_foreground = (pred > conf).astype(int)
        pred_components = label(np.squeeze(pred_foreground))
        label_components = label(np.squeeze(label_foreground))
        # find the centroids of each prediction
        pred_pos_centroids = []
        for i in range(1, np.max(pred_components) + 1):
            pred_component = pred_components == i
            if pred_component.sum() < (500 / voxvol):
                continue
            pred_pos_centroid = np.round(np.mean(np.where(pred_component), axis=1)).astype(int)
            pred_pos_centroids.append(pred_pos_centroid)
        pred_pos_centroids = np.array(pred_pos_centroids)

        # now we have a set of positive detections - evaluate these using the label
        # as ground truth.

        # loop through all label sections - if it doesn't have a prediction in it, it is a false negative
        for i in range(1, np.max(label_components) + 1):
            label_component = label_components == i
            label_pos_centroid = np.round(np.mean(np.where(label_component), axis=1)).astype(int)
            if pred_foreground[label_pos_centroid[0], label_pos_centroid[1], label_pos_centroid[2]] < 1:
                fn += 1

        for pred_centroid in pred_pos_centroids:
            if seg[pred_centroid[0], pred_centroid[1], pred_centroid[2]] ==2:
                tp += 1
            else:
                fp += 1

        ### THIS SECTION COUNTS NUMBER OF TRUE NEGATIVES

        kidney_binary = (seg > 0).astype(int)
        # get rid of any connected components in kidney_binary that contains a positive prediction
        kidney_components = label(np.squeeze(kidney_binary))
        for pred_pos_centroid in pred_pos_centroids:
            kidney_component = kidney_components == kidney_components[pred_pos_centroid[0], pred_pos_centroid[1], pred_pos_centroid[2]]
            kidney_binary[kidney_component] = 0

        # get rid of any connected components in kidney_binary that contains an actual cancer

        for i in range(1, np.max(label_components) + 1):
            label_component = label_components == i
            label_pos_centroid = np.round(np.mean(np.where(label_component), axis=1)).astype(int)
            kidney_component = kidney_components == kidney_components[label_pos_centroid[0], label_pos_centroid[1], label_pos_centroid[2]]
            kidney_binary[kidney_component] = 0

        # now we have a set of true negatives in kidney binary - any remaining connected components in kidney binary

        kidney_components = label(np.squeeze(kidney_binary))
        tn+= np.max(kidney_components)
        print(f'Confidence: {conf}, Sensitivity: {tp/(tp+fn+1e-6)}, Specificity: {tn/(tn+fp+1e-6)}')
        print()


    results.append([conf, tp, fp, fn, tn])