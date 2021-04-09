import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tabulate import tabulate
import seaborn as sns

# Plotting defaults
sns.set_theme(style="darkgrid")
# plt.rcParams.update(plt.rcParamsDefault) # reset matplotlib to defaults
# plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

# Getting back the data:
with open('results.pkl', 'rb') as f:
    results = pickle.load(f)
with open('abstich.pkl', 'rb') as f:
    meas = pickle.load(f)
with open('example.pkl', 'rb') as f:
    example = pickle.load(f)

# Generate time series of daily values
meas_day = pd.DataFrame()
meas_day['final'] = pd.to_numeric(meas['final'])
meas_day['marker_error'] = pd.to_numeric(meas['marker_error'])
meas_day['scale_error'] = pd.to_numeric(meas['scale_error'])

# add midnight time stamps, sort df, interpolate linearly
dd = pd.date_range(meas_day.index[0], meas_day.index[-1], freq='1d', normalize=True)
meas_day = meas_day.append(pd.DataFrame(index=dd)).sort_index().interpolate()

# take midnight values and calculate differences
meas_day = meas_day.groupby(pd.DatetimeIndex(meas_day.index).to_period('1D')).nth(0).to_timestamp()
meas_day['ablation'] = meas_day['final'].diff().shift(-1) * 0.9
meas_day['err_day'] = np.sqrt((meas_day['scale_error'] ** 2).rolling(2).sum()).shift(-1) * 0.9

# table output
print(tabulate(meas_day, tablefmt="simple", floatfmt=("", ".2f", ".2f", ".2f")))

data = meas.convert_dtypes().loc['2020-07-01']
fig, ax = plt.subplots(figsize=(8, 8. / 1.62), constrained_layout=True, sharex=True)
date_form = mdates.ConciseDateFormatter("%m-%d")
ax.xaxis.set_major_formatter(date_form)
sns.lineplot(x=data.index,
             y="final",
             marker="o",
             data=data)
ax.errorbar(x=data.index,
            y="final",
            yerr="scale_error",
            fmt=' ',
            zorder=-1,
            data=data,
            ecolor="gray")  # ecolor=colors,
plt.ylabel("Visible stake length (mm)")
plt.savefig("plt_abstich_singleday.png")

data = meas.convert_dtypes()
data['hour'] = data.index.hour + data.index.minute / 60.
data_g = data.groupby(meas.index.date)
fig, ax = plt.subplots(figsize=(8, 8. / 1.62), constrained_layout=True, sharex=True)
for name, data in data_g:
    sns.lineplot(x='hour',
                 y=(data['final'] - data['final'][0]),
                 data=data,
                 alpha=0.3)
plt.ylabel("Cumulative ablation (mm)")
plt.xlabel("Hour")
plt.savefig("plt_abstich_day-overplot.png")

data = meas.convert_dtypes()
fig, ax = plt.subplots(figsize=(8, 8. / 1.62), constrained_layout=True, sharex=True)
date_form = mdates.ConciseDateFormatter("%m-%d")
ax.xaxis.set_major_formatter(date_form)
sns.lineplot(x=data.index,
             y="final",
             marker="",
             data=data)
plt.ylabel("Visible stake length (mm)")
plt.savefig("plt_abstich_fullperiod.png")

fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True, sharex=True)
date_form = mdates.ConciseDateFormatter("%m-%d")
ax.xaxis.set_major_formatter(date_form)
ax.autoscale(enable=True, axis='x', tight=True)
ax.bar(meas_day.index,
       meas_day['ablation'],
       0.8,
       yerr=meas_day['err_day'])
plt.ylabel("Ablation (mm w.e.)")
plt.savefig("plt_melt.png")

# regression: vertical midpoints as X, mm-per-px as Y
fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True, sharex=True)
ax.set_box_aspect(1)
sns.scatterplot(x=example['x'],
                y=example['y'],
                marker="o")
xi = np.arange(0, 1.1 * max(example['x']))
sns.lineplot(x=xi,
             y=example["sI_scale_fun"](xi),
             alpha=0.5)
plt.text(max(example['x']), max(example['y']) - 0.02, str("R² = {:.2f}".format(example["Rsq"][0])),
         horizontalalignment='right', size='medium', color='black', weight='semibold')
plt.ylabel("mm-per-px (mm/px)")
plt.xlabel("Distance from bottom (px)")
plt.savefig("plt_regExmple.png")
