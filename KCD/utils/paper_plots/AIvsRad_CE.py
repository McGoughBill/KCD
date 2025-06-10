import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import scipy.optimize as opt


save_loc = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/nat_paper'
def polynomial_model(x, a, b, c):
    return a * x**0.1 + b * x + c


AI_func = polynomial_model
Rad_func = polynomial_model
# Raw data for cost-effectiveness calculation
intervals = np.array([1, 2, 4, 10, 20])

# AI data
ai_costs = np.array([19.8, 10.9, 6.4, 3.85, 3.2])
ai_lys = np.array([945, 841, 706, 519, 462])
ai_ce = ai_lys / (1e6 * ai_costs)

# Radiologist data
rad_costs = np.array([37.8, 20.0, 11.1, 5.8, 3.9])
rad_lys = np.array([1063, 906, 739, 561, 419])
rad_ce = rad_lys / (1e6 * rad_costs)

# Prepare data for curve fitting (extend to include 0) - for radiologist
prev_max = np.array(intervals).max()
x_fit = np.array(intervals)
y_fit_rad = np.array(rad_ce) / prev_max

# Fit the log model to cost-effectiveness data
opt_params_rad, _ = opt.curve_fit(Rad_func, x_fit, y_fit_rad, maxfev=10000)

# Prepare data for curve fitting (extend to include 0) - for AI
prev_max = np.array(intervals).max()
x_fit = np.array(intervals)
y_fit_ai = np.array(ai_ce) / prev_max

# Fit the log model to cost-effectiveness data
opt_params_ai, _ = opt.curve_fit(AI_func, x_fit, y_fit_ai, maxfev=10000)

# Create a finer grid for plotting the fitted curve
ai_ce_fitted = AI_func(np.linspace(0, 20, 1000), *opt_params_ai)*prev_max
rad_ce_fitted = Rad_func(np.linspace(0, 20, 1000), *opt_params_rad)*prev_max

# Apply the same scaling factor as before
scale_factor = 1000000
ai_ce_scaled = ai_ce * scale_factor
rad_ce_scaled = rad_ce * scale_factor
ai_rad_ce_fitted = ai_ce_fitted * scale_factor
rad_ce_fitted = rad_ce_fitted * scale_factor

# Plotting
fig, ax = plt.subplots(figsize=(8, 5))
# ax.plot(np.linspace(0, 20, 1000), ai_rad_ce_fitted, color='black', linestyle='--',alpha=0.5)
# ax.plot(np.linspace(0, 20, 1000), rad_ce_fitted, color='r', linestyle='--',alpha=0.5)
ax.plot(intervals, ai_ce_scaled, color='k', linestyle='--', label='AI and Radiologist',marker='^', markersize=6)
ax.plot(intervals, rad_ce_scaled, color='r', linestyle='-.', label='Radiologist',marker='^', markersize=6)

# #plot the scatters of raw data
# ax.scatter(intervals, ai_ce_scaled, color='k', marker='x', s=50)
# ax.scatter(intervals, rad_ce_scaled, color='r', marker='o', s=50)



ax.set_xlim(0, 20.5)
ax.set_ylim(0, 160)
ax.set_xlabel('Screening interval / years', fontdict={'fontsize': 26, 'family': 'Helvetica'})
#unit of cost-effectiveness is life-years saved per £1,000,000
ax.set_ylabel('Life-years saved per £M', fontdict={'fontsize': 26, 'family': 'Helvetica'})
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# Format tick labels
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(16)
    tick.label.set_fontname('Helvetica')
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(16)
    tick.label.set_fontname('Helvetica')

# Legend formatting
legend = ax.legend(loc=(0.65, 0.05))
for text in legend.get_texts():
    text.set_fontname('Helvetica')
    text.set_fontsize(16)

# Remove unnecessary spines for a cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
plt.savefig(os.path.join(save_loc, 'AIvsRad_CE.png'), dpi=600)
plt.show()

