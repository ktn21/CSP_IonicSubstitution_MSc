from AtomicEmbeddings import Atomic_Embeddings
import matplotlib.pyplot as plt
import numpy as np
from ElMD import elmd
from itertools import combinations_with_replacement
import pandas as pd
import seaborn as sns
from pymatgen.core import Element
from scipy.stats import pearsonr

'''
# Plotting and Visualizing the Atomic Embeddings
'''

# Import the Atomic Embeddings
cbfvs = ['mat2vec', 'megnet16', 'mod_petti']
AtomEmbeds = {cbfv: Atomic_Embeddings.from_json(cbfv) for cbfv in cbfvs}

# mat2vec correction
post_103 = ['Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
for el in post_103:
    del AtomEmbeds['mat2vec'].embeddings[el]

# mod_pettifor correction
del AtomEmbeds['mod_petti'].embeddings['D']
del AtomEmbeds['mod_petti'].embeddings['T']
for sub in AtomEmbeds['mod_petti'].embeddings:
    AtomEmbeds['mod_petti'].embeddings[sub] = np.array([int(AtomEmbeds['mod_petti'].embeddings[sub]), int(-1)])

post_103 = ['Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', 'Uue']
for el in post_103:
    del AtomEmbeds['mod_petti'].embeddings[el]

# For the modified pettifor number, use the Earth Mover's Distance as the correlation metric
emd_tab = []
pairs = combinations_with_replacement(AtomEmbeds['mod_petti'].element_list, 2)
for s1, s2 in pairs:
    prob = elmd(s1, s2, metric="mod_petti")
    emd_tab.append((s1, s2, prob))
    if s1 != s2:
        emd_tab.append((s2, s1, prob))

emd_lambda = pd.DataFrame(emd_tab, columns=["Element 1", "Element 2", "EMD"])

mend_1 = [(Element(ele).mendeleev_no, ele) for ele in emd_lambda["Element 1"]]
mend_2 = [(Element(ele).mendeleev_no, ele) for ele in emd_lambda["Element 2"]]

emd_lambda["mend_1"] = mend_1
emd_lambda["mend_2"] = mend_2

emd_lambda = emd_lambda.sort_values(by=['mend_1', 'mend_2'])
emd_pivot = emd_lambda.pivot_table(values="EMD", index="Element 1", columns="Element 2", sort=False)
col_order = list(emd_lambda.sort_values('mend_2')['Element 2'].unique())
emd_pivot = emd_pivot[col_order]

# emd_lambda = emd_lambda.pivot_table(values="EMD", index="mend_1", columns="mend_2")
emd_pivot = -(emd_pivot/(emd_pivot.to_numpy().max())-1)

plt.figure(figsize=[10, 10])
ax = sns.heatmap(emd_pivot,
                 cmap="Blues",
                 square=True,
                 linecolor="k",
                 cbar_kws={"shrink": 0.7},
                 xticklabels=2)
plt.savefig('mod_petti_elmd.png')

mat2vec_pearson = AtomEmbeds['mat2vec'].plot_pearson_correlation()
plt.savefig('mat2vec_pearson.png')
megnet_pearson = AtomEmbeds['megnet16'].plot_pearson_correlation()
plt.savefig('megnet_pearson.png')

# mat2vec_chebychev = AtomEmbeds['mat2vec'].plot_distance_correlation(metric='chebyshev')
# megnet_chebyshev = AtomEmbeds['megnet16'].plot_distance_correlation(metric='chebyshev')

# mat2vec_distance = AtomEmbeds['mat2vec'].plot_tSNE()
# megnet_distance = AtomEmbeds['megnet16'].plot_tSNE()
# mod_petti_distance = AtomEmbeds['mod_petti'].plot_tSNE()


# CrabNet Embeddings
AtomEmbeds = pd.read_csv(r"embeddings_crabnet_mat2vec/OQMD_Bandgap_layer2_Q.csv")
AtomEmbeds = AtomEmbeds.rename({'Unnamed: 0': 'Elements'}, axis=1)

# crabnet correction
post_103 = ['Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
for el in post_103:
    AtomEmbeds = AtomEmbeds[AtomEmbeds['Elements'] != el]

neutral_pairs = combinations_with_replacement(AtomEmbeds["Elements"], 2)

AtomEmbeds = AtomEmbeds.set_index('Elements').T.to_dict('list')
for sub in AtomEmbeds:
    AtomEmbeds[sub] = np.asarray(AtomEmbeds[sub])

# Pivot Table construction taken from the Atomic_Embeddings class
neutral_lambda_tab = []
for ele1, ele2 in neutral_pairs:
    pearson = pearsonr(AtomEmbeds[ele1], AtomEmbeds[ele2])
    neutral_lambda_tab.append((ele1, ele2, pearson[0]))
    if ele1 != ele2:
        neutral_lambda_tab.append((ele2, ele1, pearson[0]))
neutral_lambda_tab = pd.DataFrame(neutral_lambda_tab, columns=["Element 1", "Element 2", "pearson_corr"])
# Order by Mendeleev Number
mend_1 = [(Element(ele).mendeleev_no, ele) for ele in neutral_lambda_tab["Element 1"]]
mend_2 = [(Element(ele).mendeleev_no, ele) for ele in neutral_lambda_tab["Element 2"]]
neutral_lambda_tab["mend_1"] = mend_1
neutral_lambda_tab["mend_2"] = mend_2
# crabnet_pearson = neutral_lambda_tab.pivot_table(values="pearson_corr", index="mend_1", columns="mend_2")

neutral_lambda_tab = neutral_lambda_tab.sort_values(by=['mend_1', 'mend_2'])
crabnet_pearson = neutral_lambda_tab.pivot_table(values="pearson_corr", index="Element 1", columns="Element 2", sort=False)
col_order = list(neutral_lambda_tab.sort_values('mend_2')['Element 2'].unique())
crabnet_pearson = crabnet_pearson[col_order]

plt.figure(figsize=[10, 10])
ax = sns.heatmap(crabnet_pearson,
                 cmap="Blues",
                 square=True,
                 linecolor="k",
                 cbar_kws={"shrink": 0.7},
                 xticklabels=2)
plt.savefig('crabnet_pearson.png')


