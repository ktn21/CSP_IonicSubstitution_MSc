# Contains functions used in the Crystal Structure Prediction using SMACT and CrabNet
from pymatgen.ext.matproj import MPRester
from pymatgen.core import Composition
from itertools import combinations_with_replacement
import pandas as pd
from ElMD import elmd
from pymatgen.core import Element
from pymatgen.analysis.local_env import CrystalNN
from robocrys import StructureCondenser
from robocrys.featurize.adapter import FeaturizerAdapter

mpr = MPRester("s7nGaCXnx6g4iuPoFT4")

coornum_calc = CrystalNN()
condenser = StructureCondenser()


def get_cat_oxi_state(x):
    comp = Composition(x)
    oxi_states = comp.oxi_state_guesses()
    if len(oxi_states) > 0:
        oxi_states_output = list(oxi_states[0].values())
    else:
        oxi_states_output = [0, 0]

    return oxi_states_output[0]


def get_an_oxi_state(x):
    comp = Composition(x)
    oxi_states = comp.oxi_state_guesses()
    if len(oxi_states) > 0:
        oxi_states_output = list(oxi_states[0].values())
    else:
        oxi_states_output = [0, 0]

    return oxi_states_output[1]


def foo(s):
    ret = ""
    i = True  # capitalize
    for char in s:
        if i:
            ret += char.upper()
        else:
            ret += char.lower()
        if char != ' ':
            i = not i
    return ret


# Local Env Approach
def get_cn_local(x):
    try:
        struct_local = mpr.get_structure_by_material_id(x)
        coornum = coornum_calc.get_cn(structure=struct_local, n=0)
    except:
        coornum = 0

    return coornum


# RoboCrys Approach
def get_cn_robo(x):
    struct_robo = mpr.get_structure_by_material_id(x)
    cond_struct = condenser.condense_structure(struct_robo)
    # Find value for coordination number in cond_struct
    coorenv = cond_struct["sites"][0]["geometry"]["type"]
    coornum = FeaturizerAdapter(cond_struct).average_cation_coordination_number

    return [coornum, coorenv]


def get_formula(x):
    if not str(x)[0].isupper():
        x = foo(str(x))
    comp = Composition(x)
    elements = list(comp.chemical_system.partition("-"))
    elements.remove("-")

    return elements


def emd_corr_df(x):
    """
    Generates a dataframe of normalized Earth Mover's Distances using the Modified
    Pettifor Number metric for a list of atomic embeddings.

    :param x: Atomic Embeddings element list (dict {'element': feature vector})
    :return neutral_lambda_tab: A dataframe containing element combinations and corresponding normalized EMD
    """

    emd_tab = []
    neutral_pairs = combinations_with_replacement(x, 2)
    for s1, s2 in neutral_pairs:
        prob = elmd(s1, s2, metric="mod_petti")
        emd_tab.append((s1, s2, prob))
        if s1 != s2:
            emd_tab.append((s2, s1, prob))

    neutral_lambda_tab = pd.DataFrame(emd_tab, columns=["ele_1", "ele_2", "EMD"])
    neutral_lambda_tab["EMD"] = neutral_lambda_tab["EMD"] / (neutral_lambda_tab["EMD"].to_numpy().max())

    mend_1 = [(Element(ele).mendeleev_no, ele) for ele in neutral_lambda_tab["ele_1"]]
    mend_2 = [(Element(ele).mendeleev_no, ele) for ele in neutral_lambda_tab["ele_2"]]
    neutral_lambda_tab["mend_1"] = mend_1
    neutral_lambda_tab["mend_2"] = mend_2

    return neutral_lambda_tab


def prediction_num(x):
    if x == "Correct":
        output = 1
    else:
        output = 0

    return output


def prediction_correctness(d):
    d["Pred Binary"] = d["Prediction"].apply(prediction_num)
    tab = pd.DataFrame(d, columns=["M", "X", "Pred Binary"])
    tab = tab.pivot_table(values="Pred Binary", index="M", columns="X")

    return tab


def sort_heatmap_correctness(tab):
    sort_vector = tab.sum(axis=0).T / (int(tab.max().max()) * tab.count())
    sort_index = tab.index
    sort_index = {x: y for x, y in enumerate(sort_index)}
    tab = tab.append(sort_vector, ignore_index=True)
    tab = tab.sort_values(by=len(tab) - 1, ascending=True, axis=1)
    tab = tab.drop(len(tab) - 1)
    tab = tab.rename(sort_index)

    sort_vector = tab.sum(axis=1) / (int(tab.max().max()) * tab.count(axis=1))
    tab = tab.join(sort_vector.rename("Sum"))
    tab = tab.sort_values(by="Sum", ascending=True)
    tab = tab.drop(columns=["Sum"])

    tab_cols = list(tab.columns)
    tab_ind = list(tab.index)[::-1]

    return tab_cols, tab_ind

