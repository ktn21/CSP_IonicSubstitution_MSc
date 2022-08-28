import smact
from smact import element_dictionary, ordered_elements, neutral_ratios
from smact.screening import pauling_test
from datetime import datetime
import itertools
from pathos.multiprocessing import ProcessPool as Pool
from pymatgen.core import Composition
from smact.structure_prediction.utilities import unparse_spec
import pandas as pd


'''
# This script generates the list of binary materials that are going to be evalauted for whether the structure can be 
# predicted by the SMACT Structure Predictor. This list is constructed by making every 1:1 combination of atoms that
# will pass the oxidation state balance test and eletronegativity test of the SMACT filter function.

# The output file is called BinaryMX_ForSMACT.csv, which is used in BinaryMX_StructurePredictor.py as the trial dataset
'''


def comp_maker(comp):
    form = []
    for el, ammt in zip(comp[0], comp[2]):
        form.append(el)
        form.append(ammt)
    form = ''.join(str(e) for e in form)
    pmg_form = Composition(form).reduced_formula
    return pmg_form


def smact_filter(els, stoichs=None, species_unique=True):
    if stoichs is None:
        stoichs = [[1], [1]]
    compositions = []

    # Get symbols and electronegativities
    symbols = tuple(e.symbol for e in els)
    electronegs = [e.pauling_eneg for e in els]
    ox_combos = [e.oxidation_states for e in els]
    for ox_states in itertools.product(*ox_combos):
        # Test for charge balance
        cn_e, cn_r = neutral_ratios(ox_states, stoichs=stoichs)
        # Electronegativity test
        if cn_e:
            electroneg_OK = pauling_test(ox_states, electronegs)
            if electroneg_OK:
                for ratio in cn_r:
                    compositions.append(tuple([symbols, ox_states, ratio]))

    # Return list depending on whether we are interested in unique species combinations
    # or just unique element combinations.
    if species_unique:
        return compositions
    else:
        compositions = [(c[0], c[2]) for c in compositions]
        compositions = list(set(compositions))
        return compositions


# Generate the dictionary of elements
all_el = element_dictionary(elements=ordered_elements(1, 83))
symbols_list = list(all_el.keys())
dont_want = ["He", "Ne", "Ar", "Kr", "Xe", "Pm", "Tc"]

for unwanted in dont_want:
    symbols_list.remove(unwanted)
all_els = [all_el[symbol] for symbol in symbols_list]
coord_els = [el.coord_envs for el in all_els]
MX_pairs = [(x, y) for x in all_els for y in all_els]

# Use multiprocessing and smact_filter to quickly generate our list of compositions
start = datetime.now()

if __name__ == '__main__':  # Always use pool protected in an if statement
    with Pool(4) as p:  # start 4 worker processes
        result = p.map(smact_filter, MX_pairs)
print('Time taken to generate list:  {0}'.format(datetime.now() - start))

# Flatten the list of lists
flat_list = [item for sublist in result for item in sublist]
print('Number of compositions: --> {0} <--'.format(len(flat_list)))
print('Each list entry looks like this:\n  elements, oxidation states, stoichiometries')
for i in flat_list[:5]:
    print(i)

if __name__ == '__main__':
    with Pool() as p:
        pretty_formulas = p.map(comp_maker, flat_list)

print('Each list entry now looks like this: ')
for i in pretty_formulas[:5]:
    print(i)

species = []
M = []
X = []
for i in range(len(flat_list)):
    species.append(pretty_formulas[i])
    M.append((unparse_spec((flat_list[i][0][0], flat_list[i][1][0]))))
    X.append((unparse_spec((flat_list[i][0][1], flat_list[i][1][1]))))

columns = ["Pretty Formula", "M", "X"]
df_list = [[species[i], M[i], X[i]] for i in range(len(species))]
df = pd.DataFrame(data=df_list, columns=columns)

df.head()
# Save the list of materials
df.to_csv("./BinaryMX_ForSMACT.csv", index=False)
