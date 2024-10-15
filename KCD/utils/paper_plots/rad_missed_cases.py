import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Define the path to your folder containing the PNG files
homepath = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/nat_paper/radiologist_missed_lesions'  # Replace with your actual path
save_loc = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/nat_paper'
# Get all PNG files from 'homepath' and sort them
png_files = [f for f in os.listdir(homepath) if f.endswith('.png')]
png_files.sort()  # Ensure consistent order

# Desired image size (width, height)
desired_size = (512, 512)  # Adjust as needed

# Determine the appropriate resampling filter based on Pillow version
try:
    resample_filter = Image.Resampling.LANCZOS  # For Pillow 9.1.0 and above
except AttributeError:
    resample_filter = Image.LANCZOS  # For older versions of Pillow

# Create a 2x3 subplot arrangement
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()  # Flatten to 1D array for easy iteration

# Labels for subplots
labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

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
fig.suptitle('Axial CT slices showing 6 renal lesions missed by radiologists,\nbut detected by our AI pipeline, during YKST.\nVisualised with a CT window of -200HU, 200HU.', fontsize=20)

# Adjust layout to make space for the caption
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save the figure as a png
save_path = os.path.join(save_loc, '4b_radmissed_lesions.png')
plt.savefig(save_path)
# Display the plot
plt.show()
