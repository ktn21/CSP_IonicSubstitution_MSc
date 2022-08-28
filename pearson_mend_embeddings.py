import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

'''
# Plotting the Pearson Correlation (or other similarity metric) against the difference of Mendeleev numbers between
# the two ions within the binary compound MX.

# Plan to include the substitution probabilities versus the Mendeleev difference!
'''

# Load Data
# control = pd.read_csv("control_pred_results.csv") # FIX
sradius = pd.read_csv("shannon_correlation.csv")
mat2vec = pd.read_csv("mat2vec_correlation.csv")
megnet = pd.read_csv("megnet16_correlation.csv")
mod_petti = pd.read_csv("mod_petti_correlation.csv")
crabnet = pd.read_csv("crabnet_correlation.csv")

fig = plt.figure()
ax1 = fig.add_subplot(111)

mat2vec_mean = mat2vec.groupby('mend_diff').mean()
sradius_mean = (sradius.dropna()).groupby('mend_diff').mean()
megnet_mean = megnet.groupby('mend_diff').mean()
mod_petti_mean = mod_petti.groupby('mend_diff').mean()
crabnet_mean = crabnet.groupby('mend_diff').mean()

test = mat2vec_mean.groupby(np.arange(len(mat2vec_mean))//3).mean()

mat2vec_std = mat2vec.groupby('mend_diff').std()
sradius_std = (sradius.dropna()).groupby('mend_diff').std()
megnet_std = megnet.groupby('mend_diff').std()
mod_petti_std = mod_petti.groupby('mend_diff').std()
crabnet_std = crabnet.groupby('mend_diff').std()

# Define every num'th points to be plotted
# Note: the standard deviation is divided by 3 to improve visibility of the plot
num = 2
sradius_ind = sradius_mean.iloc[::num, :].index
ind = mat2vec_mean.iloc[::num, :].index

ax1.errorbar(sradius_ind, sradius_mean.iloc[::num, :]['lambda'],
             yerr=(sradius_std.iloc[::num, :]['lambda'].div(3)), xerr=None, c='mediumseagreen', marker='o', ls="none")
ax1.errorbar(ind, mat2vec_mean.iloc[::num, :]['pearson_corr'],
             yerr=mat2vec_std.iloc[::num, :]['pearson_corr'].div(3), xerr=None, c='palevioletred', marker='v', ls='none')
ax1.errorbar(ind, megnet_mean.iloc[::num, :]['pearson_corr'],
             yerr=megnet_std.iloc[::num, :]['pearson_corr'].div(3), xerr=None, c='cornflowerblue', marker='s', ls='none')
ax1.errorbar(ind, mod_petti_mean.iloc[::num, :]['EMD'],
             yerr=mod_petti_std.iloc[::num, :]['EMD'].div(3), xerr=None, c='sandybrown', marker='d', ls='none')
ax1.errorbar(ind, crabnet_mean.iloc[::num, :]['pearson_corr'],
             yerr=crabnet_std.iloc[::num, :]['pearson_corr'].div(3), xerr=None, c='lightslategrey', marker='*', ls='none')
ax1.legend(['Shannon Radius', 'mat2vec', 'MEGNet', 'Mod Pettifor', 'CrabNet'])
ax1.set_xlabel('Mendeleev Number Difference')
ax1.set_ylabel('Mean Correlation Metric, ' + chr(955))

'''
# Plot without error bars
fig = plt.figure()
ax2 = fig.add_subplot(111)
ax2.scatter(sradius_mean.index, sradius_mean['lambda'], c='mediumseagreen')
ax2.scatter(mat2vec_mean.index, mat2vec_mean['pearson_corr'], c='palevioletred', marker='v')
ax2.scatter(megnet_mean.index, megnet_mean['pearson_corr'], c='cornflowerblue', marker='s')
ax2.scatter(mod_petti_mean.index, mod_petti_mean['EMD'], c='sandybrown', marker='d')
ax2.scatter(crabnet_mean.index, crabnet_mean['pearson_corr'], c='lightslategrey', marker='*')
ax2.legend(['Shannon Radius', 'mat2vec', 'MEGNet', 'Mod Pettifor', 'CrabNet'])
plt.yscale('log')
ax2.set_xlabel('Mendeleev Number Difference')
ax2.set_ylabel('Mean Correlation Metric')
'''