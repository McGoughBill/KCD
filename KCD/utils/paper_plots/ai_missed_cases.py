import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Define the path to your folder containing the PNG files
homepath = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/nat_paper/AImissed_masses'  # Replace with your actual path
save_loc = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/nat_paper'

# Get all PNG files from 'homepath' and sort them
png_files = [f for f in os.listdir(homepath) if f.endswith('.png')]
png_files.sort()  # Ensure consistent order

# Since we are interested in the 4 cases the AI missed, select only those files
# If they are the first four files, you can slice the list
png_files = png_files[:4]

# Alternatively, if the missed cases have specific filenames, specify them directly
# png_files = ['missed_case1.png', 'missed_case2.png', 'missed_case3.png', 'missed_case4.png']

# Desired image size (width, height)
desired_size = (512, 512)  # Adjust as needed

# Determine the appropriate resampling filter based on Pillow version
try:
    resample_filter = Image.Resampling.LANCZOS  # For Pillow 9.1.0 and above
except AttributeError:
    resample_filter = Image.LANCZOS  # For older versions of Pillow

# Create a 2x2 subplot arrangement
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()  # Flatten to 1D array for easy iteration

# Labels for subplots
labels = ['(a)', '(b)', '(c)', '(d)']

# Loop over the images and plot them
for idx, ax in enumerate(axes):
    if idx < len(png_files):
        img_path = os.path.join(homepath, png_files[idx])
        img = Image.open(img_path)
        img_resized = img.resize(desired_size, resample=resample_filter)
        img_array = np.array(img_resized)
        ax.imshow(img_array, cmap='gray')
        ax.axis('off')
        ax.set_title(labels[idx], fontsize=16)
    else:
        ax.axis('off')  # Hide any unused subplots

# Add an overall figure caption
fig.suptitle('Axial CT slices showing 4 renal lesions missed by the AI during YKST', fontsize=20)
plt.savefig(save_loc + '/4a_aimissed_lesions.png')
# Adjust layout to make space for the caption
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Display th
