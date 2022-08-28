import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import FixedLocator, FormatStrFormatter
from myCSPfunctions import prediction_correctness, sort_heatmap_correctness

'''
########################## Coordination Number Prediction Accuracy ##########################

Overlaying the prediction results of the 4 experimental embeddings, with a different color specifiying different 
combinations of the embeddings that provide a correct prediction + sizing depending on number of embeddings that 
gave a correct prediction. Prediction results can be obtained from:

BinaryMX_StructurePredictor.py

'''

# Load Data
mat2vec = pd.read_csv("BinaryMX_mat2vecLambda_pred_results.csv")
megnet = pd.read_csv("BinaryMX_megnet16Lambda_pred_results.csv")
mod_petti = pd.read_csv("BinaryMX_mod_pettiLambda_pred_results.csv")
crabnet = pd.read_csv("BinaryMX_crabnet_Lambda_pred_results.csv")
# Create binary pass/fail pivot table from the prediction results
mat2vec_tab = prediction_correctness(mat2vec)
megnet_tab = prediction_correctness(megnet)
mod_petti_tab = prediction_correctness(mod_petti)
crabnet_tab = prediction_correctness(crabnet)
# Sum of maps to indicate number of embeddigns which predicted correct structure
tab = mat2vec_tab + megnet_tab + mod_petti_tab + crabnet_tab
tab_cols, tab_ind = sort_heatmap_correctness(tab)
tab = tab[tab_cols[::-1]]
tab = tab.reindex(tab_ind)
# Here, we use a custom numerical mapping scheme to represent a unique number of a color for all combinations of
# embeddings that are possible. The orering vectors move the rows and columns such that the most correct predictions
# PER total number of predictions is moved to the top left
overlay = mat2vec_tab + megnet_tab.multiply(5) + mod_petti_tab.multiply(8) + crabnet_tab.multiply(15)
overlay = overlay[tab_cols[::-1]]
overlay = overlay.reindex(tab_ind)
# Mapping Scheme and Accompanying Colormap
map_dict = dict({1: 'm2v', 5: 'MEG', 6: 'm2v & MEG', 8: 'mod', 9: 'm2v & mod', 13: 'MEG & mod', 14: 'm2v & MEG & mod',
                 15: 'crab', 16: 'm2v & crab', 20: 'MEG & crab', 21: 'm2v & MEG & crab', 23: 'mod & crab',
                 24: 'm2v & mod & crab', 28: 'MEG & mod & crab', 29: 'Correct All'})
my_colors = ['indianred', 'palevioletred', 'cornflowerblue', 'pink', 'sandybrown', 'gold', 'bisque', 'mediumpurple',
             'lightslategrey', 'thistle', 'aquamarine', 'darkgrey', 'lightblue', 'rosybrown',
             'orchid', 'mediumseagreen']
my_cmap = ListedColormap(my_colors)
bounds = list(map_dict.keys())
bounds = [x + 0.5 for x in bounds]
bounds.insert(0, 0.5)
bounds.insert(0, -0.5)
my_norm = BoundaryNorm(bounds, ncolors=len(my_colors))

# Heatmap Plotting
N = overlay.shape[0]
M = overlay.shape[1]
ylabels = list(overlay.index)
xlabels = list(overlay.columns)

# Data
x, y = np.meshgrid(np.arange(M), np.arange(N))
c = overlay.to_numpy()

fig, ax = plt.subplots(figsize=(10, 14))

# Sizing Factor
R = (tab.to_numpy()+1)/(tab.max().max()+1)/2
R[R == 0.4] = 0.5-0.07
R[R == 0.3] = 0.5-2*0.07
R[R == 0.2] = 0.5-3*0.07
R[R == 0.1] = 0.5-4*0.07
circles = [plt.Circle((j, i), radius=r, linewidth=1, linestyle='-', edgecolor='k') for r, j, i in zip(R.flat, x.flat, y.flat)]
col = PatchCollection(circles, array=c.flatten(), cmap=my_cmap, norm=my_norm, linewidth=0.5, linestyle='-', edgecolor='k')
ax.add_collection(col)

ax.yaxis.set_major_locator(FixedLocator(np.arange(N)[0::2]))
ax.yaxis.set_minor_locator(FixedLocator(np.arange(N)[1::2]))
ax.yaxis.set_minor_formatter(FormatStrFormatter("%d"))
ax.tick_params(which='major', pad=2, axis='y', length=38)

ax.set(xticks=np.arange(M), yticks=np.arange(N)[0::2],
       xticklabels=xlabels, yticklabels=ylabels[0::2])
ax.set_yticks(np.arange(N)[1::2], minor=True)
ax.set_yticklabels(ylabels[1::2], minor=True)
ax.grid(which='major', lw=0.3)
ax.grid(which='minor', axis='y', lw=0.3)

fig.colorbar(col)
colorbar = ax.collections[0].colorbar
colorbar.set_ticks([(b0+b1)/2 for b0, b1 in zip(bounds[:-1], bounds[1:])])
labels = list(map_dict.values())
labels.insert(0, 'Incorrect All')
colorbar.set_ticklabels(labels)

plt.ylim([-1, 131])
plt.xlim([-1, 28])
plt.gca().invert_yaxis()
plt.xticks(rotation=90)
plt.show()

ax.set_title("SMACT Coordination Number Prediction Results")
plt.savefig('Prediction Accuracy CN Pastel Vertical V2.png')

########################## Structure Prediction Accuracy ##########################

# Load Atomic Embeddings
mat2vec_struct = pd.read_csv("BinaryMX_mat2vec_Lambda_struct_pred_results.csv")
megnet_struct = pd.read_csv("BinaryMX_megnet16_Lambda_struct_pred_results.csv")
mod_petti_struct = pd.read_csv("BinaryMX_mod_petti_Lambda_struct_pred_results.csv")
crabnet_struct = pd.read_csv("BinaryMX_CrabNet_Lambda_struct_pred_results.csv")
# Create binary pass/fail pivot table from the prediction results
mat2vec_struct_tab = prediction_correctness(mat2vec_struct)
megnet_struct_tab = prediction_correctness(megnet_struct)
mod_petti_struct_tab = prediction_correctness(mod_petti_struct)
crabnet_struct_tab = prediction_correctness(crabnet_struct)
# Sum of maps to indicate number of embeddigns which predicted correct structure
tab = mat2vec_struct_tab + megnet_struct_tab + mod_petti_struct_tab + crabnet_struct_tab
tab_cols, tab_ind = sort_heatmap_correctness(tab)
tab = tab[tab_cols[::-1]]
tab = tab.reindex(tab_ind)
# Mapping Scheme and Accompanying Colormap previously defined in Coordination Number Prediction Accuracy Section
overlay = mat2vec_struct_tab + megnet_struct_tab.multiply(5) + mod_petti_struct_tab.multiply(8) + crabnet_struct_tab.multiply(15)
overlay = overlay[tab_cols[::-1]]
overlay = overlay.reindex(tab_ind)

my_cmap = ListedColormap(my_colors)
bounds = list(map_dict.keys())
bounds = [x + 0.5 for x in bounds]
bounds.insert(0, 0.5)
bounds.insert(0, -0.5)
my_norm = BoundaryNorm(bounds, ncolors=len(my_colors))

# Heatmap Plotting
N = overlay.shape[0]
M = overlay.shape[1]
ylabels = list(overlay.index)
xlabels = list(overlay.columns)

# Data, c
x, y = np.meshgrid(np.arange(M), np.arange(N))
c = overlay.to_numpy()

fig, ax = plt.subplots(figsize=(10, 14))

# Sizing Factor, R
R = (tab.to_numpy()+1)/(tab.max().max()+1)/2
R[R == 0.4] = 0.5-0.07
R[R == 0.3] = 0.5-2*0.07
R[R == 0.2] = 0.5-3*0.07
R[R == 0.1] = 0.5-4*0.07
circles = [plt.Circle((j, i), radius=r) for r, j, i in zip(R.flat, x.flat, y.flat)]
col = PatchCollection(circles, array=c.flatten(), cmap=my_cmap, norm=my_norm, linewidth=0.5, linestyle='-', edgecolor='k')
ax.add_collection(col)

ax.yaxis.set_major_locator(FixedLocator(np.arange(N)[0::2]))
ax.yaxis.set_minor_locator(FixedLocator(np.arange(N)[1::2]))
ax.yaxis.set_minor_formatter(FormatStrFormatter("%d"))
ax.tick_params(which='major', pad=2, axis='y', length=38)

ax.set(xticks=np.arange(M), yticks=np.arange(N)[0::2],
       xticklabels=xlabels, yticklabels=ylabels[0::2])
ax.set_yticks(np.arange(N)[1::2], minor=True)
ax.set_yticklabels(ylabels[1::2], minor=True)
ax.grid(which='major', lw=0.3)
ax.grid(which='minor', axis='y', lw=0.3)

fig.colorbar(col)
colorbar = ax.collections[0].colorbar
colorbar.set_ticks([(b0+b1)/2 for b0, b1 in zip(bounds[:-1], bounds[1:])])
labels = list(map_dict.values())
labels.insert(0, 'Incorrect All')
colorbar.set_ticklabels(labels)

plt.ylim([-1, 131])
plt.xlim([-1, 28])
plt.gca().invert_yaxis()
plt.xticks(rotation=90)
plt.show()

ax.set_title("SMACT Structure Prediction Results")
plt.savefig('Prediction Accuracy Structure Pastel Vertical V2.png')
