import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

'''
Identify zero-coordinated compositions
Plot the lambda and substitution probabilities for available embedding types
Coordination Number Comparison for final datasets of SMACT structure predictor

Data can be obtained from: BinaryMX_StructurePredictor.py
'''


# Load Atomic Embeddings
control = pd.read_csv("BinaryMX_control_Lambda_pred_results.csv")
sradius = pd.read_csv("BinaryMX_shannonLambda_pred_results.csv")
mat2vec = pd.read_csv("BinaryMX_mat2vecLambda_pred_results.csv")
megnet = pd.read_csv("BinaryMX_megnet16Lambda_pred_results.csv")
mod_petti = pd.read_csv("BinaryMX_mod_pettiLambda_pred_results.csv")
crabnet = pd.read_csv("BinaryMX_crabnet_Lambda_pred_results.csv")


'''
# Zero Coordination Number Compositions
'''
mat2vec_zeros = []
for i, row in enumerate(mat2vec.values):
    if mat2vec["CN Prediction"][i] == 0:
        mat2vec_zeros.append([mat2vec["Pretty Formula"][i], mat2vec["Parent formula"][i]])

megnet_zeros = []
for i, row in enumerate(megnet.values):
    if megnet["CN Prediction"][i] == 0:
        megnet_zeros.append([megnet["Pretty Formula"][i], megnet["Parent formula"][i]])

mod_petti_zeros = []
for i, row in enumerate(mod_petti.values):
    if mod_petti["CN Prediction"][i] == 0:
        mod_petti_zeros.append([mod_petti["Pretty Formula"][i], mod_petti["Parent formula"][i]])

crabnet_zeros = []
for i, row in enumerate(crabnet.values):
    if crabnet["CN Prediction"][i] == 0:
        crabnet_zeros.append([crabnet["Pretty Formula"][i], crabnet["Parent formula"][i]])


'''
# Lambda and substitution probability value histograms comparison
'''

# Lambda of Atomic Embeddings
nbins = 100
control_lambda = pd.read_csv("Control Lambda Table from Pymatgen.csv")
control_lambda = control_lambda.drop(columns='0')
sradius_lambda = pd.read_csv("Shannon Radius Lambda Table.csv")
sradius_lambda = sradius_lambda.drop(columns='0')
mat2vec_lambda = pd.read_csv("mat2vec Lambda Table.csv")
mat2vec_lambda = mat2vec_lambda.drop(columns='0')
megnet_lambda = pd.read_csv("megnet Lambda Table.csv")
megnet_lambda = megnet_lambda.drop(columns='0')
mod_petti_lambda = pd.read_csv("mod_petti Lambda Table.csv")
mod_petti_lambda = mod_petti_lambda.drop(columns='0')
crabnet_lambda = pd.read_csv("CrabNet Lambda Table.csv")
crabnet_lambda = crabnet_lambda.drop(columns='0')

control_lambda = control_lambda.stack()
sradius_lambda = sradius_lambda.stack()
mat2vec_lambda = mat2vec_lambda.stack()
megnet_lambda = megnet_lambda.stack()
mod_petti_lambda = mod_petti_lambda.stack()
crabnet_lambda = crabnet_lambda.stack()
fig = plt.figure(figsize=[12, 4.5])
ax2 = fig.add_subplot(1, 2, 1)
plt.hist(control_lambda, bins=nbins, histtype=u'step', density=True)
plt.hist(sradius_lambda, bins=nbins, histtype=u'step', density=True)
plt.hist(mat2vec_lambda, bins=nbins, histtype=u'step', density=True)
plt.hist(megnet_lambda, bins=nbins, histtype=u'step', density=True)
plt.hist(mod_petti_lambda, bins=nbins, histtype=u'step', density=True)
plt.hist(crabnet_lambda, bins=nbins, histtype=u'step', density=True)
plt.yscale('log')
plt.xlim([-2, 2])
plt.legend(["Control", "Shannon Radius", "mat2vec", "MEGNet", "Mod Pettifor", "CrabNet"])
plt.xlabel("Lambda (" + chr(955) + ") Values")

# Substitution Probabilities
control_sub = pd.read_csv("BinaryMX_control_Conditional_Substitution_Table.csv")
control_sub = control_sub.drop(columns='0')
sradius_sub = pd.read_csv("BinaryMX_shannon_Conditional_Substitution_Table.csv")
sradius_sub = sradius_sub.drop(columns='0')
mat2vec_sub = pd.read_csv("BinaryMX_mat2vec_Conditional_Substitution_Table.csv")
mat2vec_sub = mat2vec_sub.drop(columns='0')
megnet_sub = pd.read_csv("BinaryMX_megnet16_Conditional_Substitution_Table.csv")
megnet_sub = megnet_sub.drop(columns='0')
mod_petti_sub = pd.read_csv("BinaryMX_mod_petti_Conditional_Substitution_Table.csv")
mod_petti_sub = mod_petti_sub.drop(columns='0')
crabnet_sub = pd.read_csv("BinaryMX_crabnet_Conditional_Substitution_Table.csv")
crabnet_sub = crabnet_sub.drop(columns='0')

control_sub = control_sub.stack()
sradius_sub = sradius_sub.stack()
mat2vec_sub = mat2vec_sub.stack()
megnet_sub = megnet_sub.stack()
mod_petti_sub = mod_petti_sub.stack()
crabnet_sub = crabnet_sub.stack()

nbins = 100
ax1 = fig.add_subplot(1, 2, 2)
'''
plt.hist(control["probability"], bins=nbins, histtype=u'step', density=True)
plt.hist(sradius["probability"], bins=nbins, histtype=u'step', density=True)
plt.hist(mat2vec["probability"], bins=nbins, histtype=u'step', density=True)
plt.hist(megnet["probability"], bins=nbins, histtype=u'step', density=True)
plt.hist(mod_petti["probability"], bins=nbins, histtype=u'step', density=True)
plt.hist(crabnet["probability"], bins=nbins, histtype=u'step', density=True)
'''
plt.hist(control_sub, bins=nbins, histtype=u'step', density=True)
plt.hist(sradius_sub, bins=nbins, histtype=u'step', density=True)
plt.hist(mat2vec_sub, bins=nbins, histtype=u'step', density=True)
plt.hist(megnet_sub, bins=nbins, histtype=u'step', density=True)
plt.hist(mod_petti_sub, bins=nbins, histtype=u'step', density=True)
plt.hist(crabnet_sub, bins=nbins, histtype=u'step', density=True)
plt.yscale('log')
plt.xlim([0, 0.015])
# plt.legend(["Control", "Shannon Radius", "mat2vec", "MEGNet", "Mod Pettifor", "CrabNet"])
plt.xlabel("Substitutional Probability")
plt.show()

'''
# Coordination Number Comparison
'''
coornums_df = pd.DataFrame({"Experimental": mat2vec["CN Experiment"].value_counts(sort=False),
                            "mat2vec": mat2vec["CN Prediction"].value_counts(sort=False),
                            "MEGNet": megnet["CN Prediction"].value_counts(sort=False),
                            "mod_petti": mod_petti["CN Prediction"].value_counts(sort=False),
                            "CrabNet": crabnet["CN Prediction"].value_counts(sort=False)},
                           index=mat2vec["CN Experiment"].unique())

color_dict = {'Experimental': 'mediumseagreen', 'mat2vec': 'palevioletred', 'MEGNet': 'cornflowerblue',
              'mod_petti': 'sandybrown', 'CrabNet': 'lightslategrey'}

ax = coornums_df.plot(kind='bar', color=list(color_dict.values()), rot=0)
ax.set_xlabel('Coordination Number')
ax.set_ylabel('Count')

'''
# Proof that the materials for each coordination number are equal across experimental embeddings
coornums_df = pd.DataFrame({"mat2vec": mat2vec["CN Experiment"].value_counts(sort=False),
                            "MEGNet": megnet["CN Experiment"].value_counts(sort=False),
                            "mod_petti": mod_petti["CN Experiment"].value_counts(sort=False),
                            "CrabNet": crabnet["CN Experiment"].value_counts(sort=False)},
                           index=mat2vec["CN Experiment"].unique())
ax = coornums_df.plot(kind='bar')
ax.set_xlabel('Coordination Number')
ax.set_ylabel('Count')
'''