import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from smact.structure_prediction.utilities import parse_spec
from pymatgen.core import Element
from matplotlib.colors import BoundaryNorm, ListedColormap
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder
from myCSPfunctions import prediction_correctness, sort_heatmap_correctness

'''
This script contains the plotting of overlay heatmaps to illustrate the frequency of correct predictions by the 
experimental embeddings for both coordination number and crystal structure. It also includes a linear analysis of the 
ordering of ions resulting from the mapping, and visualization of the correct compositions versus the entire dataset

Prediction results can be obtained from: BinaryMX_StructurePredictor.py

'''

'''
########################## Coordination Number Prediction Accuracy ##########################

Overlaying the prediction results of the 4 experimental embeddings, with a different color specifiying different 
combinations of the embeddings that provide a correct prediction.

'''
# Load Atomic Embeddings
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
# Ordering vectors
tab_cols, tab_ind = sort_heatmap_correctness(tab)
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
# Plot the heatmap
figsize = [10, 14]
plt.figure(figsize=figsize)
ax = sns.heatmap(overlay,
                 cmap=my_cmap,
                 linewidths=0.5,
                 linecolor="k",
                 norm=my_norm)

colorbar = ax.collections[0].colorbar
colorbar.set_ticks([(b0 + b1) / 2 for b0, b1 in zip(bounds[:-1], bounds[1:])])
labels = list(map_dict.values())
labels.insert(0, 'Incorrect All')
colorbar.set_ticklabels(labels)
# Save the figure!
ax.set_title("SMACT Coordination Number Prediction Results")
plt.savefig('Prediction Accuracy CN Vertical.png')

'''
# Cation Linear Analysis for Coordination Numbers

Plotting the Mendelev Number, Atomic Number, and Atomic Radius against the order in which the the cations are plotted 
in the coordination number overlay prediction results. The ordering is based on the number of correct predictions over 
the number of total predictions for materials containing that cation.

'''
mend = [Element(parse_spec(ion)[0]).mendeleev_no for ion in tab_ind]
atomZ = [Element(parse_spec(ion)[0]).Z for ion in tab_ind]
atom_rad = [Element(parse_spec(ion)[0]).atomic_radius_calculated for ion in tab_ind]

cation_char = pd.DataFrame({'Cation': tab_ind, 'Mendeleev No': mend, 'Atomic Number': atomZ, 'Atomic Radius': atom_rad})
tick_freq = 13
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))

ax1.plot(cation_char['Cation'], cation_char['Mendeleev No'], 'o')
ax1.set_xlabel('Cation')
ax1.set_ylabel('Mendeleev Number')
ax1.set_xticks(cation_char['Cation'][::tick_freq])
ax1.set_xticklabels(cation_char['Cation'][::tick_freq], rotation=45)

ax2.plot(cation_char['Cation'], cation_char['Atomic Number'], 'o')
ax2.set_xlabel('Cation')
ax2.set_ylabel('Atomic Number')
ax2.set_xticks(cation_char['Cation'][::tick_freq])
ax2.set_xticklabels(cation_char['Cation'][::tick_freq], rotation=45)

ax3.plot(cation_char['Cation'], cation_char['Atomic Radius'], 'o')
ax3.set_xlabel('Cation')
ax3.set_ylabel('Atomic Radius $\mathrm{\AA}$')
ax3.set_xticks(cation_char['Cation'][::tick_freq])
ax3.set_xticklabels(cation_char['Cation'][::tick_freq], rotation=45)

plt.show()

x = np.arange(0, 131, 1, dtype=int).reshape((-1, 1))
y1 = cation_char['Mendeleev No'].to_numpy()
y2 = cation_char['Atomic Number'].to_numpy()
y3_bool = cation_char['Atomic Radius'].notna()
y3 = cation_char['Atomic Radius'][y3_bool == True].to_numpy()
x3 = x[y3_bool == True]

# Linear Regression
linear_model1 = LinearRegression().fit(x, y1)
linear_model2 = LinearRegression().fit(x, y2)
linear_model3 = LinearRegression().fit(x3, y3)
# Coefficient of Determination
r_sq1 = linear_model1.score(x, y1)
r_sq2 = linear_model2.score(x, y2)
r_sq3 = linear_model3.score(x3, y3)

'''
########################## Structure Prediction Accuracy ##########################
'''

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
# Ordering vectors
tab_cols, tab_ind = sort_heatmap_correctness(tab)

overlay_struct = mat2vec_struct_tab + megnet_struct_tab.multiply(5) + \
                 mod_petti_struct_tab.multiply(8) + crabnet_struct_tab.multiply(15)
overlay_struct = overlay_struct[tab_cols[::-1]]
overlay_struct = overlay_struct.reindex(tab_ind)
# Mapping Scheme and Accompanying Colormap previously defined in Coordination Number Prediction Accuracy Section
figsize = [10, 14]
plt.figure(figsize=figsize)
ax = sns.heatmap(overlay_struct,
                 cmap=my_cmap,
                 linewidths=0.5,
                 linecolor="k",
                 norm=my_norm,
                 )

colorbar = ax.collections[0].colorbar
colorbar.set_ticks([(b0 + b1) / 2 for b0, b1 in zip(bounds[:-1], bounds[1:])])
labels = list(map_dict.values())
labels.insert(0, 'Incorrect All')
colorbar.set_ticklabels(labels)
# Save the figure!
ax.set_title("SMACT Structure Prediction Results")
plt.savefig('Prediction Accuracy Structures Vertical.png')


'''
# Cation Linear Analysis for Coordination Numbers

Plotting the Mendelev Number, Atomic Number, and Atomic Radius against the order in which the the cations are plotted 
in the structure overlay prediction results. The ordering is based on the number of correct predictions over 
the number of total predictions for materials containing that cation.

'''
mend = [Element(parse_spec(ion)[0]).mendeleev_no for ion in tab_ind]
atomZ = [Element(parse_spec(ion)[0]).Z for ion in tab_ind]
atom_rad = [Element(parse_spec(ion)[0]).atomic_radius_calculated for ion in tab_ind]

cation_char = pd.DataFrame({'Cation': tab_ind, 'Mendeleev No': mend, 'Atomic Number': atomZ, 'Atomic Radius': atom_rad})
tick_freq = 13
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
ax1.plot(cation_char['Cation'], cation_char['Mendeleev No'], 'o')
ax1.set_xlabel('Cation')
ax1.set_ylabel('Mendeleev Number')
ax1.set_xticks(cation_char['Cation'][::tick_freq])
ax1.set_xticklabels(cation_char['Cation'][::tick_freq], rotation=45)

ax2.plot(cation_char['Cation'], cation_char['Atomic Number'], 'o')
ax2.set_xlabel('Cation')
ax2.set_ylabel('Atomic Number')
ax2.set_xticks(cation_char['Cation'][::tick_freq])
ax2.set_xticklabels(cation_char['Cation'][::tick_freq], rotation=45)

ax3.plot(cation_char['Cation'], cation_char['Atomic Radius'], 'o')
ax3.set_xlabel('Cation')
ax3.set_ylabel('Atomic Radius $\mathrm{\AA}$')
ax3.set_xticks(cation_char['Cation'][::tick_freq])
ax3.set_xticklabels(cation_char['Cation'][::tick_freq], rotation=45)

plt.show()

x = np.arange(0, 131, 1, dtype=int).reshape((-1, 1))
y1 = cation_char['Mendeleev No'].to_numpy()
y2 = cation_char['Atomic Number'].to_numpy()
y3_bool = cation_char['Atomic Radius'].notna()
y3 = cation_char['Atomic Radius'][y3_bool == True].to_numpy()
x3 = x[y3_bool == True]

linear_model1 = LinearRegression().fit(x, y1)
linear_model2 = LinearRegression().fit(x, y2)
linear_model3 = LinearRegression().fit(x3, y3)

r_sq1 = linear_model1.score(x, y1)
r_sq2 = linear_model2.score(x, y2)
r_sq3 = linear_model3.score(x3, y3)


'''
################# Number of Correctly Predicted Structures vs All Structures in Dataset Plot #################

Scatter Plot: Shows the crystal system and coordination number of the materials in the dataset and color codes 
depending on correct/incorrect predictions

Bar Chart: Shows the number of correctly predicted materials over the total dataset, separated by crystal system

'''

# Differences between coordination number and structure predictions
# Can replace table inside brackets with any struct_tab for any available embedding
correct_structs_all = overlay_struct[mat2vec_struct_tab == 1]
compare_overlay = overlay[overlay_struct == 29]

passing_materials = correct_structs_all.reset_index().melt(id_vars=['M']).dropna()
passing_materials = passing_materials.reset_index()

print(len(passing_materials['X'].unique()))
print(passing_materials['X'].value_counts())

formula = []
for i, row in enumerate(passing_materials.values):
    ele_1 = parse_spec(passing_materials['M'][i])[0]
    ele_2 = parse_spec(passing_materials['X'][i])[0]
    formula.append(ele_1 + ele_2)

passing_materials["Formula"] = formula

# Find the materials in the database
BinaryDB = pd.read_csv('Binary_Dataset_for_RRRule_localenv.csv')
BinaryDB = BinaryDB.sort_values(by=['pretty_formula', 'e_above_hull'])
# Check lowest e_above_hull state only
BinaryDB = BinaryDB.drop_duplicates(subset=['pretty_formula'], keep='first')
BinaryDB = BinaryDB.reset_index(drop=True)
BinaryDB = BinaryDB[BinaryDB['pretty_formula'].isin(crabnet['Pretty Formula'])]
ReducedDB = BinaryDB[BinaryDB['pretty_formula'].isin(passing_materials['Formula'])]

BinaryDB = BinaryDB.assign(InDf2=BinaryDB['pretty_formula'].isin(passing_materials['Formula']).astype(int))
ExcludedDB = BinaryDB[BinaryDB['InDf2'] == 0]

# Scatterplot of coordination number and crystal system
ord_enc = OrdinalEncoder()
enc_df = pd.DataFrame(ord_enc.fit_transform(BinaryDB), columns=list(BinaryDB.columns))
categories = pd.DataFrame([np.array(ord_enc.categories_).transpose()], columns=list(BinaryDB.columns))
categories1 = categories['spacegroup.crystal_system'].apply(pd.Series).values.tolist()
categories1 = [item for sublist in categories1 for item in sublist]
categories2 = categories['coordination_num'].apply(pd.Series).values.tolist()
categories2 = [item for sublist in categories2 for item in sublist]
# Generate the random noise
xnoise, ynoise = np.random.random(len(BinaryDB))/2, np.random.random(len(BinaryDB))/2
# Plot the scatterplot
figsize = [8, 5]
plt.figure(figsize=figsize)
colors = {1: 'sandybrown', 0: 'cornflowerblue'}
plt.scatter(enc_df['spacegroup.crystal_system']+xnoise, enc_df["coordination_num"]+ynoise,
            c=BinaryDB['InDf2'].map(colors), alpha=0.5)
# You can also set xticks and yticks to be your category names:
plt.xticks(np.linspace(0.25, len(enc_df["spacegroup.crystal_system"].unique())-0.75,
                       len(enc_df["spacegroup.crystal_system"].unique())), categories1)
plt.yticks(np.linspace(0.25, len(enc_df["coordination_num"].unique())-0.75,
                       len(enc_df["coordination_num"].unique())), categories2)
plt.xlabel('Crystal Structure')
plt.ylabel('Coordination Number')
plt.grid()
sns.despine(left=True, bottom=True)

# Stacked Bar Chart of Crystal Systems, Correct for All vs Not
plotdata = BinaryDB.groupby(['spacegroup.crystal_system', 'InDf2'])['spacegroup.crystal_system'].\
    count().unstack('InDf2').fillna(0)
ax = plotdata.plot(kind='bar', color=['cornflowerblue', 'sandybrown'], stacked=True, rot=45)
ax.set_xlabel("Crystal Structure")
ax.set_ylabel("Count")
ax.legend(["Incorrect", "Correct"])
