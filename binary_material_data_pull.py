from pymatgen.ext.matproj import MPRester
import pandas as pd
import matplotlib.pyplot as plt
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
import logging
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import SimplestChemenvStrategy, MultiWeightsChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments
from myCSPfunctions import get_an_oxi_state, get_cat_oxi_state
from datetime import datetime
from myCSPfunctions import get_cn_local, get_cn_robo

# Default CN method is local environment (any number NOT 2 or 3)
# cn_method = 2: Robocrystallographer
# cn_method = 3: ChemEnv
cn_method = 1
# User may rename file of final dataset output - will be saved as a csv
filename = "Binary_Dataset_for_RRRule_localenv"

mpr = MPRester("s7nGaCXnx6g4iuPoFT4")

# Query all binary 1 to 1 materials with an ICSD ID -> only experimental structures!
# "**" is all binary compounds of 1 to 1 ratio
# Alternative: anon_formula = {'A': 1, 'B': 1}
anon_formula = {'A': 1, 'B': 1}
binary_data = mpr.query(criteria={"anonymous_formula": anon_formula, "icsd_ids": {"$gte": 0}},
                        properties=["task_id", "pretty_formula",
                                    "spacegroup.symbol", "spacegroup.crystal_system", "e_above_hull",
                                    "structure", "density", "band_gap", "final_energy"])

# Change to Dataframe for ease of table manipulation
binary_df = pd.DataFrame(binary_data)
# Get the lowest energy above hull structure and place into separate dataframe
binary_df = binary_df.sort_values(by=['pretty_formula', 'e_above_hull'])
# Sort back by index
binary_df_les = binary_df.reset_index(drop=True)
# Visualization of Number of each structure in the data set
binaryRR_data = binary_df_les['spacegroup.crystal_system'].value_counts()
plot1 = binaryRR_data.plot(kind='bar')
plot1.set_xlabel('Crystal System')
plot1.set_ylabel('Count')
# Annotation of number of compositions per crystal system
for rect in plot1.patches:
    y_value = rect.get_height()
    x_value = rect.get_x() + rect.get_width() / 7
    space = 1
    label = "{:.0f}".format(y_value)
    plot1.annotate(label, (x_value+0.2, y_value), xytext=(0, space),
                   textcoords="offset points", ha='center', va='bottom')

# Statistical Histograms for Numerical Data in Dataset
plt.figure()
hist1 = binary_df_les['e_above_hull'].hist(bins=100)
hist1.set_xlabel('Energy Above the Convex Hull [eV/atom]')
hist1.set_ylabel('Count')
hist1.set_xlim(0, 0.5)
plt.figure()
hist2 = binary_df_les['band_gap'].hist(bins=100)
hist2.set_xlabel('Band Gap [eV]')
hist2.set_ylabel('Count')
hist2.set_xlim(0, 2)
plt.figure()
hist3 = binary_df_les['density'].hist(bins=100)
hist3.set_xlabel('Density [g cm-3]')
hist3.set_ylabel('Count')

# Get oxidation states from pymatgen compositions
binary_df_les['cat_oxi_state'] = binary_df_les["pretty_formula"].apply(get_cat_oxi_state)
binary_df_les['an_oxi_state'] = binary_df_les["pretty_formula"].apply(get_an_oxi_state)

start = datetime.now()
# Different methods for calculating the coordination environment
if cn_method == 2:
    # RoboCrys Approach
    binary_df_les['coordination'] = binary_df_les["task_id"].apply(get_cn_robo)
    binary_df_les[['coordination_num', 'coordination_env']] = \
        pd.DataFrame(binary_df_les.coordination.tolist(), index=binary_df_les.index)
    print("done")
elif cn_method == 3:
    # ChemEnv Approach
    # Determine oxidation state and coordination environment from binary composition
    CN = []
    CE = []
    lgf = LocalGeometryFinder()
    count = 0
    for i, row in enumerate(binary_df_les.values):
        print(i)

        # Coordination Environment
        struct = mpr.get_structure_by_material_id(binary_df_les['task_id'][i])
        logging.basicConfig(format='%(levelname)s:%(module)s:%(funcName)s:%(message)s',
                            level=logging.DEBUG)
        lgf.setup_structure(structure=struct)
        try:
            se = lgf.compute_structure_environments(maximum_distance_factor=1.41, only_cations=True)
            strategy = SimplestChemenvStrategy(distance_cutoff=1.4, angle_cutoff=0.3)
            lse = LightStructureEnvironments.from_structure_environments(strategy=strategy, structure_environments=se)
            isite = 1
            CN_str = lse.coordination_environments[isite][0]['ce_symbol']

            CN.append(int(CN_str.partition(":")[2]))
            CE.append(CN_str.partition(":")[0])
        except:
            CN.append(0)
            CE.append("None")

    binary_df_les['coordination_num'] = CN
    binary_df_les['coordination_env'] = CE

    print("done")
else:
    binary_df_les['coordination_num'] = binary_df_les["task_id"].apply(get_cn_local)
    print("done")

print(datetime.now() - start)
# Save the dataset to a csv
binary_df_les.to_csv(filename+".csv")

