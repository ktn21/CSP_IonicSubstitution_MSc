import os
import pandas as pd
import json
import pymatgen.analysis.structure_prediction as pymatgen_sp
import matplotlib.pyplot as plt
import seaborn as sns
from pymatgen.core import Element
from smact.structure_prediction.utilities import parse_spec
import smact
import csv


'''

Studying the control embeddings from Hautier ionic substitution algorithm, available on pymatgen

'''


py_sp_dir = os.path.dirname(pymatgen_sp.__file__)
pymatgen_lambda = os.path.join(py_sp_dir, "data", "lambda.json")
with open(pymatgen_lambda, 'r') as f:
    lambda_dat = json.load(f)

pymatgen_lambda = os.path.join(py_sp_dir, "data", "pair_correlation.json")
with open(pymatgen_lambda, 'r') as f:
    corr_dat = json.load(f)

elem_dict = smact.element_dictionary()

# Get rid of 'D1+' values to reflect pymatgen
# implementation
lambda_dat = [x for x in lambda_dat if 'D1+' not in x]
corr_dat = [x for x in corr_dat if 'D1+' not in x]

# Convert lambda table to pandas DataFrame
lambda_dat = [tuple(x) for x in lambda_dat]
lambda_df = pd.DataFrame(lambda_dat)

corr_dat = [tuple(x) for x in corr_dat]
corr_df = pd.DataFrame(corr_dat)

shannon_df = pd.read_csv('shannon_correlation.csv')
shannon_df = shannon_df.dropna()

in_shannon = []
for i, row in enumerate(lambda_df.values):
    if lambda_df[0][i] in shannon_df['ion_1'].unique() and lambda_df[1][i] in shannon_df['ion_2'].unique() and \
            lambda_df[0][i] in shannon_df['ion_2'].unique() and lambda_df[1][i] in shannon_df['ion_1'].unique():
        in_shannon.append(1)
    else:
        in_shannon.append(0)

lambda_df['In Shannon'] = in_shannon
lambda_df = lambda_df[lambda_df['In Shannon'] == 1]
lambda_df = lambda_df.drop('In Shannon', axis=1)
lambda_df = lambda_df.rename(columns={0: 'ion_1', 1: 'ion_2', 2: 'lambda'})
lambda_df = lambda_df.reset_index(drop=True)

corr_df = corr_df.rename(columns={0: 'ion_1', 1: 'ion_2', 2: 'lambda'})

mend_1 = [(Element(parse_spec(ion)[0]).mendeleev_no, ion) for ion in lambda_df["ion_1"]]
mend_2 = [(Element(parse_spec(ion)[0]).mendeleev_no, ion) for ion in lambda_df["ion_2"]]

lambda_df["mend_1"] = mend_1
lambda_df["mend_2"] = mend_2
df_pivot = lambda_df.pivot_table(values="lambda", index="mend_1", columns="mend_2")

# df_pivot = df_pivot.dropna(thresh=len(df_pivot) - len(df_pivot)/2, axis=1)
# df_pivot = df_pivot.dropna(thresh=len(df_pivot) - len(df_pivot)/2, axis=0)
# mend_1 = [(Element(parse_spec(ion)[0]).mendeleev_no, ion) for ion in corr_df["ion_1"]]
# mend_2 = [(Element(parse_spec(ion)[0]).mendeleev_no, ion) for ion in corr_df["ion_2"]]

# corr_df["mend_1"] = mend_1
# corr_df["mend_2"] = mend_2
# corr_pivot = corr_df.pivot_table(values="lambda", index="mend_1", columns="mend_2")

# lambda_pivot = lambda_df.pivot(index=0, columns=1, values=2)

# lambda_df.to_csv("Control Lambda Table from Pymatgen.csv")
figsize = [10, 10]
plt.figure(figsize=figsize)
ax1 = sns.heatmap(df_pivot,
                  cmap="Spectral",
                  square=True,
                  linecolor="k",
                  cbar_kws={"shrink": 0.7},
                  vmin=-2, vmax=4)

with open('Control Lambda Data from Pymatgen.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(lambda_dat)
