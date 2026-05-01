import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Load the data
fp = '/path/to/kcd_data/Cambridge University Dropbox/William McGough/YKST/ykst_mass_size.csv'
df = pd.read_csv(fp)

# Discard 'id' and 'Notes', and set data types
df = df.drop(columns=['id', 'Notes'])
df['type'] = df['type'].astype(str)
df['diameter'] = df['diameter'].astype(int)
df['Detected'] = df['Detected'].astype(int)

# Define bin edges for bins like 0-5, 5-10, 10-15, etc.
bin_width = 5
max_diameter = 80  # Set this to the maximum diameter you want to display
bin_edges = np.arange(0, max_diameter + bin_width, bin_width)

# Use pandas.cut to bin 'diameter' into the defined bins
df['diameter_bin'] = pd.cut(df['diameter'], bins=bin_edges, right=False, labels=bin_edges[:-1])

# Convert 'diameter_bin' to numeric type for plotting
df['diameter_bin'] = df['diameter_bin'].astype(float)

# Group and pivot the data
grouped = df.groupby(['diameter_bin', 'type', 'Detected']).size().reset_index(name='count')
pivot_table = grouped.pivot_table(index='diameter_bin', columns=['type', 'Detected'], values='count', fill_value=0)
pivot_table = pivot_table.sort_index(axis=1, level=[0,1])

# Get unique 'type' values and assign colors
types = df['type'].unique()
print(dir(plt.cm))
colors = plt.cm.OrRd(np.linspace(0, 1, len(types)))
type_colors = dict(zip(types, colors))

# Create the plot
fig, ax = plt.subplots(figsize=(8,5))
ax.spines[['right', 'top']].set_visible(False)


diameter_bins = pivot_table.index.values
bar_width = bin_width
bottom = np.zeros(len(diameter_bins))

# Define hatching patterns for 'Detected' status
detected_hatches = {1: '/', 0: ''}

for (type_value, detected_value), counts in pivot_table.items():
    color = type_colors[type_value]
    hatch = detected_hatches[detected_value]
    ax.bar(diameter_bins, counts.values, bar_width, bottom=bottom,
           align='edge', color=color, hatch=hatch, edgecolor='black')
    bottom += counts.values

ax.set_xlabel('Diameter / mm',fontdict={'fontsize': 26})
ax.set_ylabel('Count',fontdict={'fontsize': 26})

# Set x-axis limits and ticks
ax.set_xlim(0, max_diameter)
ax.set_xticks(np.arange(0, max_diameter + bin_width, bin_width))
ax.set_xticklabels(np.arange(0, max_diameter + bin_width, bin_width))

# Create custom legend handles for 'type'
type_handles = [Patch(facecolor=type_colors[type_value], edgecolor='black') for type_value in types]
type_labels = list(types)

# Create custom legend handles for 'Detected' status
hatches = ['/', '']  # Detected == 1: '/', Detected == 0: ''
detected_status = ['Detected', 'Not Detected']
detected_handles = [Patch(facecolor='white', edgecolor='black', hatch=hatch) for hatch in hatches]
detected_labels = detected_status

# Combine both legends into one
labels = ['Type']+ type_labels + ['Detected'] +detected_labels
handles = [Patch(facecolor='none', edgecolor='none',visible=False)] + type_handles + [Patch(facecolor='none', edgecolor='none',visible='False')] + detected_handles
legend = ax.legend(handles=handles, labels=labels, loc='upper right',fontsize='large')

isheader = False
# Make 'Type' and 'Detected' labels bold to act as headers
for text, label in zip(legend.get_texts(), labels):
    print(label)
    if (label == 'Type'):
        print('a')
        text.set_weight('bold')
        text.set_ha('left')  # Align headers to the left
    elif (label == 'Detected') and not isheader:
        print('b')
        text.set_weight('bold')
        text.set_ha('right')
        isheader = True
    else:
        text.set_ha('left')  # Align labels to the left

#set font size in axis ticks and labels
ax.tick_params(axis='both', which='major', labelsize=15)
for label in ax.get_xticklabels():
    label.set_rotation(45)
    label.set_ha('right')
    label.set_fontsize(15)
plt.tight_layout()

#set all fonts to helvetica and font size to 20 on xtick and ytick and x label and y label
for text in ax.get_xticklabels():
    text.set_fontname('Helvetica')
    text.set_fontsize(20)
for text in ax.get_yticklabels():
    text.set_fontname('Helvetica')
    text.set_fontsize(20)

#set legend to helvetica and font size to 16
for text in legend.get_texts():
    text.set_fontname('Helvetica')
    text.set_fontsize(16)

plt.savefig('/path/to/kcd_data/Cambridge University Dropbox/William McGough/YKST/ykst_mass_size.png',dpi=300)
plt.show()

#make cmap from hex colour values
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
cmap = mcolors.ListedColormap(['#332288', '#88CCEE'])

# if df['Exo/Meso/Endo'] == Exo, set to 2, if Meso, set to 1, if Endo, set to 0
df['exo_bin'] = df['Exo/Meso/Endo'].map({'Exo': 1, 'Meso': 0.5, 'Endo': 0})

# plot diameter vs exo_bin, with color representing 'Detected'
fig, ax = plt.subplots(figsize=(6,6))
scatter = ax.scatter(df['diameter'], df['exo_bin'], c=df['Detected'], cmap=cmap,s=100)
#rename y ticks to Exo, Meso, Endo
ax.set_yticks([0,0.5,1])
ax.set_yticklabels(['Endo','Meso','Exo'])
ax.set_xlabel('Diameter / mm',fontdict={'fontsize': 26,'family':'Helvetica'})
# ax.set_ylabel('Exophyticness',fontdict={'fontsize': 26,'family':'Helvetica'})
ax.legend(*scatter.legend_elements(), title='Detected',loc='lower right')
detected = mpatches.Patch(color='#332288',label='Not Detected')
not_detected = mpatches.Patch(color='#88CCEE',label='Detected')
legend = plt.legend(handles=[detected,not_detected],loc = 'lower right',fontsize=16,framealpha=0.9)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

#set all fonts to helvetica and font size to 20 on xtick and ytick and x label and y label
for text in ax.get_xticklabels():
    text.set_fontname('Helvetica')
    text.set_fontsize(20)
for text in ax.get_yticklabels():
    text.set_fontname('Helvetica')
    text.set_fontsize(20)

#set legend to helvetica and font size to 16
for text in legend.get_texts():
    text.set_fontname('Helvetica')
    text.set_fontsize(16)

plt.tight_layout()
plt.savefig('/path/to/kcd_data/Cambridge University Dropbox/William McGough/YKST/ykst_mass_size2.png',dpi=300)
plt.show()





