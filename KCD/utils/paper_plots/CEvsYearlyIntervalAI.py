import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.optimize as opt
import os

save_loc = '/path/to/kcd_data/Cambridge University Dropbox/William McGough/nat_paper'


# Define model functions for curve fitting
def log_model(x, a, b, c):
    return a * np.log(b * x +1e-9) + c

def polynomial_model(x, a, b, c):
    return a * x**0.1 + b * x + c

# logistic regression model
def sat_exp_model(x, a, b, c):
    return a / (1 + np.exp(-b * (x - c)))


func = polynomial_model


# Data
intervals = [1, 2, 4, 10, 20]
costs = [19.8, 10.9, 6.4, 3.85, 3.2]
life_years_saved = [945, 841, 706, 519, 462]
cost_effectiveness = [lys / (1e6 * cost) for cost, lys in zip(costs, life_years_saved)]

# Prepare data for curve fitting (extend to include 0)
prev_max = np.array(intervals).max()
x_fit = np.array(intervals)
y_fit = np.array(cost_effectiveness) / prev_max

# Fit the log model to cost-effectiveness data
opt_params, _ = opt.curve_fit(func, x_fit, y_fit, maxfev=10000)

# Create a finer grid for plotting the fitted curve
x_new = np.linspace(0, 20, 1000)

# Create the figure and primary axis
fig, ax1 = plt.subplots(figsize=(8, 5))

# Plot cost data on the primary y-axis
ax1.scatter(intervals, costs, color='yellowgreen', marker='x', label='Cost',s=50)
ax1.set_xlim(0, 20.5)
ax1.set_ylim(0, 40)
ax1.set_xlabel('Screening interval / years', fontdict={'fontsize': 26, 'family': 'Helvetica'})
ax1.set_ylabel('Cost / M£', fontdict={'fontsize': 26, 'family': 'Helvetica'})
ax1.tick_params(axis='x', labelsize=16)
# Set primary y-axis tick fonts
for tick in ax1.yaxis.get_major_ticks():
    tick.label.set_fontsize(16)
    tick.label.set_fontname('Helvetica')

# Create secondary y-axis for life-years saved
ax2 = ax1.twinx()
ax2.scatter(intervals, life_years_saved, color='deepskyblue', marker='o', label='Life-years saved',s=50)
ax2.set_ylabel('Life-years Saved', fontdict={'fontsize': 26, 'family': 'Helvetica'})
ax2.set_ylim(0, 1200)
ax2.tick_params(axis='y', labelsize=16)
# Set secondary y-axis tick fonts
for tick in ax2.yaxis.get_major_ticks():
    tick.label.set_fontsize(16)
    tick.label.set_fontname('Helvetica')

# Remove unnecessary spines for a cleaner look
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)

# Force x-axis ticks to be integers
ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# Plot the fitted cost-effectiveness curve and data points on the primary axis
# plt.plot(x_new, func(x_new, *opt_params)*(prev_max*5000000), color='k', linestyle='--', alpha=0.5)
plt.plot(intervals, np.array(cost_effectiveness) * 5000000, color='k', linestyle='--', marker='^', markersize=6, label='Cost-effectiveness')

# Create and format a combined legend
legend = fig.legend(loc=(0.55, 0.77))
for text in legend.get_texts():
    text.set_fontname('Helvetica')
    text.set_fontsize(16)

fig.tight_layout()
plt.savefig(os.path.join(save_loc, 'CEvsYearlyIntervalAI.png'), dpi=600)
plt.show()
