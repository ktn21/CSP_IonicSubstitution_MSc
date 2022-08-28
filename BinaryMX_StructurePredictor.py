import json
from datetime import datetime
from operator import itemgetter
from pymatgen.ext.matproj import MPRester

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis.local_env import CrystalNN
import pymatgen.core.structure as mg
from pymatgen.core import Element as Pymatgen_ele

from smact.structure_prediction.database import StructureDB
from smact.structure_prediction.mutation import CationMutator
from smact.structure_prediction.prediction import StructurePredictor
from smact.structure_prediction.structure import SmactStructure
from smact.structure_prediction.utilities import parse_spec
from smact.structure_prediction.probability_models import RadiusModel

from itertools import combinations_with_replacement
from AtomicEmbeddings import Atomic_Embeddings
from myCSPfunctions import emd_corr_df


'''
################## SMACT STRUCTURE PREDICTOR ##################
This is the script used to predict the coordination numbers and structures of binary materials with formula MX.

Be sure to have run BinaryMX_MaterialGenerator.py and Experimental_BinaryMX_Database.py in order to have a functioning 
database + trial dataset to predict structures with.

For embeddings_choice, choose between the following:
- 'control'
- 'shannon'
- 'mat2vec'
- 'megnet16'
- 'mod_petti'
- 'crabnet'
- 'skipatom'
- 'magpie_sc'
- 'matscholar'

For test_type, choose one of the following:
- 'coornum': Predict coordination numbers
- 'structure': Predict structures
- 'topthree': See whether the true structure is amnong the top three most probable structures

'''


# Choice of Atomic Embeddings
# Choose between control, shannon, mat2vec, mod_petti, megnet
embeddings_choice = "crabnet"

# Coordination Number (coornum), Structure (structure), Top 3 Structures (topthree)
test_type = "structure"

'''
End User Definition
'''

RadiusModel = RadiusModel()
coornum_calc = CrystalNN()
mpr = MPRester("s7nGaCXnx6g4iuPoFT4")


def my_scaler(min_scale_num,max_scale_num,var):
    return (max_scale_num - min_scale_num) * ( (var - min(var)) / (max(var) - min(var)) ) + min_scale_num


# noinspection PyShadowingNames
def gen_model_lambda(species_list, model):
    pairs = combinations_with_replacement(species_list, 2)

    lambda_tab = []
    if model.casefold() == "shannon":
        for s1, s2 in pairs:
            prob = RadiusModel.sub_prob(s1, s2)
            lambda_tab.append((s1, s2, prob))
            if s1 != s2:
                lambda_tab.append((s2, s1, prob))

        # Plotting the shannon radius lambda table
        df = pd.DataFrame(lambda_tab, columns=["ion_1", "ion_2", "lambda"])

        mend_1 = [(Pymatgen_ele(parse_spec(ion)[0]).mendeleev_no, ion) for ion in df["ion_1"]]
        mend_2 = [(Pymatgen_ele(parse_spec(ion)[0]).mendeleev_no, ion) for ion in df["ion_2"]]
        df["mend_1"] = mend_1
        df["mend_2"] = mend_2

        # Save Correlation
        mend_diff = []
        for i, r in enumerate(df.values):
            mend_diff.append(abs(df["mend_1"][i][0]-df["mend_2"][i][0]))

        df["mend_diff"] = mend_diff
        df.to_csv(model.casefold() + "_correlation.csv")


        # Plot Shannon Lambda
        df = df.pivot_table(values="lambda", index="mend_1", columns="mend_2")

        # Heatmap of Pearson Correlation
        plt.figure(figsize=[10, 10])
        ax = sns.heatmap(df,
                         cmap="Blues",
                         square=True,
                         linecolor="k",
                         cbar_kws={"shrink": 0.7},
                         vmin=0.9, vmax=1)

        df = pd.DataFrame(lambda_tab)

    elif model.casefold() == "mat2vec":
        AtomEmbeds = Atomic_Embeddings.from_json(model.casefold())
        # mat2vec correction
        post_103 = ['Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
        for el in post_103:
            del AtomEmbeds.embeddings[el]

        neutral_lambda_tab = AtomEmbeds.create_correlation_df()
        # neutral_lambda_tab = AtomEmbeds.create_distance_correlation_df(metric='euclidean')

        # Plot the Pearson Correlation versus the difference in Mendeleev Number
        mend_diff = []
        for i, r in enumerate(neutral_lambda_tab.values):
            mend_diff.append(abs(neutral_lambda_tab["mend_1"][i][0]-neutral_lambda_tab["mend_2"][i][0]))

        neutral_lambda_tab["mend_diff"] = mend_diff
        neutral_lambda_tab.to_csv(model.casefold() + "_correlation.csv")

        for s1, s2 in pairs:
            s1_spec = parse_spec(s1)
            s2_spec = parse_spec(s2)
            prob = neutral_lambda_tab.loc[(neutral_lambda_tab['ele_1'] == s1_spec[0])
                                          & (neutral_lambda_tab['ele_2'] == s2_spec[0]), 'pearson_corr'].item()
            lambda_tab.append((s1, s2, prob))
            if s1 != s2:
                lambda_tab.append((s2, s1, prob))

        df = pd.DataFrame(lambda_tab)

    elif model.casefold() == "magpie_sc" or model.casefold() == "megnet16" or model.casefold() == "skipatom" or \
            model.casefold() == "matscholar":
        AtomEmbeds = Atomic_Embeddings.from_json(model.casefold())

        neutral_lambda_tab = AtomEmbeds.create_correlation_df()
        # neutral_lambda_tab = AtomEmbeds.create_distance_correlation_df(metric='chebyshev')

        # Save Correlation
        mend_diff = []
        for i, r in enumerate(neutral_lambda_tab.values):
            mend_diff.append(abs(neutral_lambda_tab["mend_1"][i][0]-neutral_lambda_tab["mend_2"][i][0]))

        neutral_lambda_tab["mend_diff"] = mend_diff
        neutral_lambda_tab.to_csv(model.casefold() + "_correlation.csv")

        for s1, s2 in pairs:
            s1_spec = parse_spec(s1)
            s2_spec = parse_spec(s2)
            prob = neutral_lambda_tab.loc[(neutral_lambda_tab['ele_1'] == s1_spec[0])
                                          & (neutral_lambda_tab['ele_2'] == s2_spec[0]), 'pearson_corr'].item()
            lambda_tab.append((s1, s2, prob))
            if s1 != s2:
                lambda_tab.append((s2, s1, prob))

        df = pd.DataFrame(lambda_tab)

    elif model.casefold() == "mod_petti":
        AtomEmbeds = Atomic_Embeddings.from_json('mod_petti')

        # mod_petti correction
        del AtomEmbeds.embeddings['D']
        del AtomEmbeds.embeddings['T']
        for sub in AtomEmbeds.embeddings:
            AtomEmbeds.embeddings[sub] = np.array([int(AtomEmbeds.embeddings[sub]), int(-1)])
        post_103 = ['Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', 'Uue']
        for el in post_103:
            del AtomEmbeds.embeddings[el]

        # Earth Mover's Distance Correlation Dataframe
        neutral_lambda_tab = emd_corr_df(AtomEmbeds.element_list)

        # Save Correlation
        mend_diff = []
        for i, r in enumerate(neutral_lambda_tab.values):
            mend_diff.append(abs(neutral_lambda_tab["mend_1"][i][0]-neutral_lambda_tab["mend_2"][i][0]))

        neutral_lambda_tab["mend_diff"] = mend_diff
        neutral_lambda_tab.to_csv(model.casefold() + "_correlation.csv")

        for s1, s2 in pairs:
            s1_spec = parse_spec(s1)
            s2_spec = parse_spec(s2)
            prob = neutral_lambda_tab.loc[(neutral_lambda_tab['ele_1'] == s1_spec[0])
                                          & (neutral_lambda_tab['ele_2'] == s2_spec[0]), 'EMD'].item()
            lambda_tab.append((s1, s2, prob))
            if s1 != s2:
                lambda_tab.append((s2, s1, prob))

        df = pd.DataFrame(lambda_tab)
        df[2] = -(df[2]-1)
    elif model.casefold() == "crabnet":
        AtomEmbeds = pd.read_csv(r"embeddings_crabnet_mat2vec/OQMD_Bandgap_layer2_Q.csv")
        # AtomEmbeds = pd.read_csv(r"embeddings_crabnet_mat2vec/OQMD_Volume_per_atom_layer2_Q.csv")

        AtomEmbeds = AtomEmbeds.rename({'Unnamed: 0': 'Elements'}, axis=1)

        # crabnet correction
        post_103 = ['Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
        for el in post_103:
            AtomEmbeds = AtomEmbeds[AtomEmbeds['Elements'] != el]

        neutral_pairs = combinations_with_replacement(AtomEmbeds["Elements"], 2)

        AtomEmbeds = AtomEmbeds.set_index('Elements').T.to_dict('list')
        for sub in AtomEmbeds:
            AtomEmbeds[sub] = np.asarray(AtomEmbeds[sub])

        neutral_lambda_tab = []
        for ele1, ele2 in neutral_pairs:
            pearson = pearsonr(AtomEmbeds[ele1], AtomEmbeds[ele2])
            neutral_lambda_tab.append((ele1, ele2, pearson[0]))
            if ele1 != ele2:
                neutral_lambda_tab.append((ele2, ele1, pearson[0]))
        neutral_lambda_tab = pd.DataFrame(neutral_lambda_tab, columns=["ele_1", "ele_2", "pearson_corr"])

        mend_1 = [(Pymatgen_ele(ele).mendeleev_no, ele) for ele in neutral_lambda_tab["ele_1"]]
        mend_2 = [(Pymatgen_ele(ele).mendeleev_no, ele) for ele in neutral_lambda_tab["ele_2"]]
        neutral_lambda_tab["mend_1"] = mend_1
        neutral_lambda_tab["mend_2"] = mend_2

        # Save Correlation
        mend_diff = []
        for i, r in enumerate(neutral_lambda_tab.values):
            mend_diff.append(abs(neutral_lambda_tab["mend_1"][i][0]-neutral_lambda_tab["mend_2"][i][0]))

        neutral_lambda_tab["mend_diff"] = mend_diff
        neutral_lambda_tab.to_csv(model.casefold() + "_correlation.csv")

        for s1, s2 in pairs:
            s1_spec = parse_spec(s1)
            s2_spec = parse_spec(s2)
            prob = neutral_lambda_tab.loc[(neutral_lambda_tab['ele_1'] == s1_spec[0])
                                          & (neutral_lambda_tab['ele_2'] == s2_spec[0]), 'pearson_corr'].item()
            lambda_tab.append((s1, s2, prob))
            if s1 != s2:
                lambda_tab.append((s2, s1, prob))

        df = pd.DataFrame(lambda_tab)

    elif model.casefold() == "crabnet_test":
        # Test the CrabNet self-attention correlation
        AtomEmbeds1 = pd.read_csv(r"embeddings_crabnet_mat2vec/OQMD_Bandgap_layer2_Q.csv")
        AtomEmbeds1 = AtomEmbeds1.rename(columns={'Unnamed: 0': 'ele_1'})
        AtomEmbeds1 = AtomEmbeds1.set_index('ele_1')
        AtomEmbeds2 = pd.read_csv(r"embeddings_crabnet_mat2vec/OQMD_Bandgap_layer2_K.csv")
        AtomEmbeds2 = AtomEmbeds2.rename(columns={'Unnamed: 0': 'ele_2'})
        AtomEmbeds2 = AtomEmbeds2.set_index('ele_2')

        neutral_lambda_tab = AtomEmbeds1.dot(AtomEmbeds2.transpose())

        neutral_lambda_tab = neutral_lambda_tab.stack().reset_index(name='lambda')
        neutral_lambda_tab["lambda"] = neutral_lambda_tab["lambda"] / (neutral_lambda_tab["lambda"].to_numpy().max())

        for s1, s2 in pairs:
            s1_spec = parse_spec(s1)
            s2_spec = parse_spec(s2)
            prob = neutral_lambda_tab.loc[(neutral_lambda_tab['ele_1'] == s1_spec[0])
                                          & (neutral_lambda_tab['ele_2'] == s2_spec[0]), 'lambda'].item()
            lambda_tab.append((s1, s2, prob))
            if s1 != s2:
                lambda_tab.append((s2, s1, prob))

        df = pd.DataFrame(lambda_tab)

    else:
        raise Exception("Invalid atomic embeddings")
    return df.pivot(index=0, columns=1, values=2)


# Compositions
comps = pd.read_csv("BinaryMX_ForSMACT.csv")
comps.head()

# SMACT Database
DB = StructureDB("BinaryMX.db")

# Cation Mutator
# Get list of all cations
symbols = pd.concat([comps["M"], comps["X"]], ignore_index=True, axis=0).drop_duplicates()
symbols = symbols.reset_index(drop=True)
symbols_list = list(symbols)

if embeddings_choice.casefold() == "control":
    CM = CationMutator.from_json()
else:
    lambda_df = gen_model_lambda(symbols_list, embeddings_choice)
    # lambda_df = -(lambda_df-1)
    lambda_df = lambda_df.dropna(axis=0, how="all")
    lambda_df = lambda_df.dropna(axis=1, how="all")
    csv_filename = embeddings_choice + " Lambda Table.csv"
    lambda_df.to_csv(csv_filename)
    lambda_pivot = lambda_df.stack().to_frame().set_index(0, append=True).index.tolist()

    lambda_json = json.dumps(lambda_pivot)
    json_filename = embeddings_choice + "_lambda.json"
    jsonFile = open(json_filename, "w")
    jsonFile.write(lambda_json)
    jsonFile.close()
    CM = CationMutator.from_json(json_filename)

SP = StructurePredictor(CM, DB, "Experimental")

# Corrections
cond_df = CM.complete_cond_probs()
fullpath = "./BinaryMX_" + embeddings_choice + "_Conditional_Substitution_Table.csv"
cond_df.to_csv(fullpath)
sub_df = CM.complete_sub_probs()
fullpath = "./BinaryMX_" + embeddings_choice + "_Substitution_Table.csv"
sub_df.to_csv(fullpath)

species = list(cond_df.columns)
comps_copy = comps[['M', 'X']]
df_copy_bool = comps_copy.isin(species)
x = comps_copy[df_copy_bool].fillna(0)
x = x[x.M != 0]
x = x[x.X != 0]
x = x.reset_index(drop=True)
# x.to_csv("./Garnet_Comps_Corrected_Pym.csv", index=False)

inner_merged = pd.merge(x, comps)
# inner_merged = inner_merged.drop_duplicates(subset=['Pretty Formula'], keep='first')
inner_merged = inner_merged[inner_merged['M'].str.contains("\+")]
inner_merged = inner_merged.reset_index(drop=True)
print(inner_merged.head())
print("")
print(f"We have reduced our search space from {comps.shape[0]} to {inner_merged.shape[0]}")

# Plot the pair correlations of the control method
if embeddings_choice.casefold() == 'control':
    pair_corrs = CM.complete_pair_corrs()
    pair_corrs = pair_corrs.stack().reset_index(name='paircorr')
    pair_corrs = pair_corrs.rename(columns={0: 'ion_1', 1: 'ion_2'})
    mostcommonions = list(inner_merged['M'].value_counts()[0:49].index)
    # Find the most common ions in the dataset!
    pair_corrs = pair_corrs[pair_corrs['ion_1'].isin(mostcommonions)]
    pair_corrs = pair_corrs[pair_corrs['ion_2'].isin(mostcommonions)]
    pair_corrs['paircorr_log'] = np.log10(pair_corrs['paircorr'])

    mend_1 = [(Pymatgen_ele(parse_spec(ion)[0]).mendeleev_no, ion) for ion in pair_corrs["ion_1"]]
    mend_2 = [(Pymatgen_ele(parse_spec(ion)[0]).mendeleev_no, ion) for ion in pair_corrs["ion_2"]]
    pair_corrs["mend_1"] = mend_1
    pair_corrs["mend_2"] = mend_2
    pair_corrs = pair_corrs.sort_values(by=['mend_1', 'mend_2'])
    pair_corrs_pivot = pair_corrs.pivot_table(values="paircorr_log", index="ion_1", columns="ion_2", sort=False)
    col_order = list(pair_corrs.sort_values('mend_2')['ion_2'].unique())
    pair_corrs_pivot = pair_corrs_pivot[col_order]

    plt.figure(figsize=[10, 10])
    ax = sns.heatmap(pair_corrs_pivot,
                     cmap="Spectral_r",
                     square=True,
                     linecolor="k",
                     linewidths=0.5,
                     cbar_kws={"shrink": 0.7},
                     vmin=-1.2, vmax=1.2, center=0,
                     xticklabels=True, yticklabels=True)

# Create a list of test species
test_specs_list = [[parse_spec(inner_merged["M"][i]), parse_spec(inner_merged["X"][i])]
                   for i in range(inner_merged.shape[0])]

# Set up a for loop to store
start = datetime.now()
preds = []
parents_list = []
probs_list = []
for test_specs in test_specs_list:
    predictions = list(SP.predict_structs(test_specs, thresh=10e-4, include_same=False))
    predictions.sort(key=itemgetter(1), reverse=True)
    parents = [x[2].composition() for x in predictions]
    probs = [x[1] for x in predictions]
    preds.append(predictions)
    parents_list.append(parents)
    probs_list.append(probs)
print(f"Time taken to predict the crystal structures of our search space of {inner_merged.shape[0]} with a threshold "
      f"of 0.0001 is {datetime.now() - start} ")
# print(parents_list)
print("")
# print(probs_list)

# Add predictions to dataframe
pred_structs = []
probs = []
parent_structs = []
parent_pretty_formula = []
pred_structs_sec = []
probs_sec = []
parent_structs_sec = []
parent_pretty_formula_sec = []
pred_structs_trd = []
probs_trd = []
parent_structs_trd = []
parent_pretty_formula_trd = []
for i in preds:
    if len(i) == 0:
        pred_structs.append(None)
        probs.append(None)
        parent_structs.append(None)
        parent_pretty_formula.append(None)
        pred_structs_sec.append(None)
        probs_sec.append(None)
        parent_structs_sec.append(None)
        parent_pretty_formula_sec.append(None)
        pred_structs_trd.append(None)
        probs_trd.append(None)
        parent_structs_trd.append(None)
        parent_pretty_formula_trd.append(None)
    else:
        # Most Probable Structure [0] index
        pred_structs.append(i[0][0].as_poscar())
        probs.append(i[0][1])
        parent_structs.append(i[0][2].as_poscar())
        parent_pretty_formula.append(mg.Structure.from_str(i[0][2].as_poscar(),
                                                           fmt="poscar").composition.reduced_formula)
        try:
            # Find index  of next probable structure
            for j in np.linspace(0, 10, 11):
                if i[0][1] != i[int(j)][1]:
                    break

            pred_structs_sec.append(i[int(j)][0].as_poscar())
            probs_sec.append(i[int(j)][1])
            parent_structs_sec.append(i[int(j)][2].as_poscar())
            parent_pretty_formula_sec.append(mg.Structure.from_str(i[int(j)][2].as_poscar(),
                                                                   fmt="poscar").composition.reduced_formula)

            try:
                for k in np.linspace(1, 10, 11):
                    if i[int(j)][1] != i[int(k)][1] and int(k) > int(j):
                        break

                pred_structs_trd.append(i[int(k)][0].as_poscar())
                probs_trd.append(i[int(k)][1])
                parent_structs_trd.append(i[int(k)][2].as_poscar())
                parent_pretty_formula_trd.append(mg.Structure.from_str(i[int(k)][2].as_poscar(),
                                                                       fmt="poscar").composition.reduced_formula)
            except:
                pred_structs_trd.append(None)
                probs_trd.append(None)
                parent_structs_trd.append(None)
                parent_pretty_formula_trd.append(None)
        except:
            pred_structs_sec.append(None)
            probs_sec.append(None)
            parent_structs_sec.append(None)
            parent_pretty_formula_sec.append(None)
            pred_structs_trd.append(None)
            probs_trd.append(None)
            parent_structs_trd.append(None)
            parent_pretty_formula_trd.append(None)


# Add prediction results to dataframe
inner_merged["predicted_structure"] = pred_structs
inner_merged["probability"] = probs
inner_merged["Parent formula"] = parent_pretty_formula
inner_merged["parent_structure"] = parent_structs

if test_type == 'topthree':
    inner_merged["predicted_structure_2"] = pred_structs_sec
    inner_merged["probability_2"] = probs_sec
    inner_merged["Parent formula_2"] = parent_pretty_formula_sec
    inner_merged["parent_structure_2"] = parent_structs_sec

    inner_merged["predicted_structure_3"] = pred_structs_trd
    inner_merged["probability_3"] = probs_trd
    inner_merged["Parent formula_3"] = parent_pretty_formula_trd
    inner_merged["parent_structure_3"] = parent_structs_trd


# Filter dataframe to remove blank entries from dataframe
results = inner_merged.dropna()
results = results.reset_index(drop=True)
results.head()

# Check if composition exists in our local database
in_db = []
for i in results["predicted_structure"]:
    comp = SmactStructure.from_poscar(i).composition()
    if len(DB.get_structs(comp, "Experimental")) != 0:
        in_db.append("Yes")
    else:
        in_db.append("No")


results["In DB?"] = in_db
print(results["In DB?"].value_counts())

# Get only those found in the database
filtered_results = results[results["In DB?"].str.contains("No") == False]
filtered_results = filtered_results.reset_index(drop=True)

# Number of correctly predicted structures
sm = StructureMatcher()
preds_accuracy = []
CN_expt = []
CN_pred = []
for i, row in enumerate(filtered_results.values):
    comp = SmactStructure.from_poscar(filtered_results["predicted_structure"][i]).composition()
    experimental_structure = DB.get_structs(comp, "Experimental")
    experimental_structure = mg.Structure.from_str(experimental_structure[0].as_poscar(), fmt="poscar")

    predicted_structure = mg.Structure.from_str(filtered_results["predicted_structure"][i], fmt="poscar")

    if test_type == 'structure':
        # Direct Structure Comparison
        if sm.fit_anonymous(predicted_structure, experimental_structure):
            preds_accuracy.append("Correct")
        else:
            preds_accuracy.append("Incorrect")
    elif test_type == 'coornum':
        coornum_expt = coornum_calc.get_cn(structure=experimental_structure, n=0)
        coornum_pred = coornum_calc.get_cn(structure=predicted_structure, n=0)
        CN_expt.append(coornum_expt)
        CN_pred.append(coornum_pred)

        # Coordination Number Comparison
        if coornum_expt == coornum_pred:
            preds_accuracy.append("Correct")
        else:
            preds_accuracy.append("Incorrect")
    elif test_type == 'topthree':
        predicted_structure_2 = mg.Structure.from_str(filtered_results["predicted_structure_2"][i], fmt="poscar")
        predicted_structure_3 = mg.Structure.from_str(filtered_results["predicted_structure_3"][i], fmt="poscar")
        # Top 3 structures
        if sm.fit_anonymous(predicted_structure, experimental_structure):
            preds_accuracy.append("Correct")
        elif sm.fit_anonymous(predicted_structure_2, experimental_structure):
            preds_accuracy.append("Correct")
        elif sm.fit_anonymous(predicted_structure_3, experimental_structure):
            preds_accuracy.append("Correct")
        else:
            preds_accuracy.append("Incorrect")

filtered_results["Prediction"] = preds_accuracy

if test_type == 'coornum':
    filtered_results["CN Experiment"] = CN_expt
    filtered_results["CN Prediction"] = CN_pred
    fullpath = "./BinaryMX_" + embeddings_choice + "_Lambda_pred_results.csv"
    filtered_results.to_csv(fullpath)
elif test_type == 'structure':
    fullpath = "./BinaryMX_" + embeddings_choice + "_Lambda_struct_pred_results.csv"
    filtered_results.to_csv(fullpath)

print("The " + embeddings_choice + " trained algorithm in SMACT with the original embeddings correctly predicted " +
      str(filtered_results["Prediction"].value_counts().Correct / len(preds_accuracy) * 100) + "% of the structures")
