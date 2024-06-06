#combine casewise_confidences.csv with results.csv - add a column for dose in casewise_confidences.csv

import pandas
import os

casewise_confidences = pandas.read_csv('/Users/mcgoug01/Downloads/test_results/casewise_confidences.csv')
results = pandas.read_csv('/Users/mcgoug01/Downloads/test_results/output/results.csv')

casewise_confidences['patient'] = casewise_confidences['file'].str.split('.').str[0]

casewise_confidences = casewise_confidences.merge(results, on='patient', how='left')
print(casewise_confidences.columns)
#remove any column with 'Unnamed' and 'patient
casewise_confidences = casewise_confidences.loc[:, ~casewise_confidences.columns.str.contains('^Unnamed')]
# drop where label is 0
casewise_confidences['radius_mm'] = (casewise_confidences['size_mm']*(3/4)/3.14159)**(1/3)
casewise_confidences['pred'] = casewise_confidences['confidence'] > 0.5
casewise_confidences['correct'] = casewise_confidences['pred'] == casewise_confidences['label']
#find sensitivity above and below 20mm radius, and above/below dose of 100
sensitivities = []
for boundary_radius in [0,3,5, 10, 15, 20, 25, 30, 35, 40,60]:
    casewise_confidences[f'above_{boundary_radius}mm'] = casewise_confidences['radius_mm'] < boundary_radius
    sensitivity_radius = casewise_confidences[casewise_confidences[f'above_{boundary_radius}mm'] & (casewise_confidences['label']==1)]['correct'].mean() * 100
    sensitivities.append({'boundary_radius': boundary_radius,'sensitivity_radius': sensitivity_radius})

for boundary_dose in [50,60, 75, 100, 125, 150, 175, 200,300]:
    casewise_confidences[f'above_{boundary_dose}mAs'] = casewise_confidences['dose'] < boundary_dose
    sensitivity_dose = casewise_confidences[casewise_confidences[f'above_{boundary_dose}mAs'] & (casewise_confidences['label']==1)]['correct'].mean() * 100
    sensitivities.append({'boundary_dose': boundary_dose, 'sensitivity_dose': sensitivity_dose})

sensitivities = pandas.DataFrame(sensitivities)
sensitivities.to_csv('/Users/mcgoug01/Downloads/test_results/sensitivities.csv')

specificity = casewise_confidences[casewise_confidences['label']==0]['correct'].mean()
print(specificity)


import matplotlib.pyplot as plt
import seaborn as sns
plt.switch_backend('TkAgg')

correlation_radconf = casewise_confidences[casewise_confidences['label']==1][['radius_mm', 'confidence']].corr()
correlation_doseconf = casewise_confidences[casewise_confidences['label']==1][['dose', 'confidence']].corr()

# plot a two 2d scatter plot of size and and dose vs confidence. confidence on y axis, size and dose on x axis
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.scatterplot(data=casewise_confidences[casewise_confidences['label']==1], y='confidence', x='radius_mm', ax=ax[0])
sns.scatterplot(data=casewise_confidences[casewise_confidences['label']==1], y='confidence', x='dose', ax=ax[1])

#draw a boundary at 20mm radius and 100mAs
ax[0].axvline(20, color='r')
ax[1].axvline(100, color='r')

#put text left of boundary saying 'small' and 'low dose'
ax[0].text(5, 0.5, 'Small', color='r', fontsize=12)
ax[0].text(40, 0.5, 'Large', color='g', fontsize=12)
ax[1].text(35, 0.5, 'Low\nDose', color='r', fontsize=12)
ax[1].text(300, 0.5, 'High\nDose', color='g', fontsize=12)

#put text in top right of each plot with correlation - like a legend
ax[0].text(30, 0.9, f'Correlation: {correlation_radconf.iloc[0,1]:.2f}', fontsize=12)
ax[1].text(200, 0.9, f'Correlation: {correlation_doseconf.iloc[0,1]:.2f}', fontsize=12)
plt.savefig('/Users/mcgoug01/Downloads/Figure_1.png')
plt.show()


plt.close()

#plot two curves of sensitivity vs dose and sensitivity vs size
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.lineplot(data=sensitivities, x='boundary_dose', y='sensitivity_dose', ax=ax[0])
sns.lineplot(data=sensitivities, x='boundary_radius', y='sensitivity_radius', ax=ax[1])

ax[0].set_title('Sensitivity vs Dose')
ax[1].set_title('Sensitivity vs Size')

ax[0].set_xlabel('Dose (mAs)')
ax[1].set_xlabel('Radius of Cancer (mm)')
ax[0].set_ylabel('Sensitivity / %')
ax[1].set_ylabel('Sensitivity / %')

correlation = casewise_confidences[['radius_mm', 'dose']].corr()
ax[0].text(100, 60, f'Correlation between\nradius and dose: {correlation.iloc[0,1]:.2f}', fontsize=12)
plt.savefig('/Users/mcgoug01/Downloads/Figure_2.png')
plt.show()


plt.close()

casewise_confidences.to_csv('/Users/mcgoug01/Downloads/test_results/dose_and_size.csv')