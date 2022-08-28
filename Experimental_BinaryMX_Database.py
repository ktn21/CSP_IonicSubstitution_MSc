import pandas as pd
from smact.structure_prediction import database
from pymatgen.analysis import structure_matcher
from pymatgen.ext.matproj import MPRester
from myCSPfunctions import get_formula, get_cat_oxi_state, get_an_oxi_state
from smact.structure_prediction.utilities import unparse_spec


'''
# This script generates the SMACT database of materials from the Materials Project which is to be used for the SMACT 
# Structure Predictor as the list of "true" structures to base the structural analogies on.

'''


SM = structure_matcher.StructureMatcher(attempt_supercell=True)
mpr = MPRester("s7nGaCXnx6g4iuPoFT4")

# Query all binary 1 to 1 materials with an ICSD Id for experimental structures
anon_formula = {'A': 1, 'B': 1}
initial_data = mpr.query(criteria={"anonymous_formula": anon_formula, "icsd_ids": {"$gte": 0}},
                         properties=["task_id", "pretty_formula", "spacegroup.symbol",
                                     "spacegroup.crystal_system", "e_above_hull", "structure"])

initial_df = pd.DataFrame(initial_data)

# Sort by e_above_hull for stability per chemical formula
data = initial_df.sort_values(by=['pretty_formula', 'e_above_hull'])
# Check lowest e_above_hull state only per composition
data = data.drop_duplicates(subset=['pretty_formula'], keep='first')
data = data.reset_index(drop=True)

# Get the chemical species and oxidation states
data['elements'] = data["pretty_formula"].apply(get_formula)
data['cat_oxi_state'] = data["pretty_formula"].apply(get_cat_oxi_state)
data['an_oxi_state'] = data["pretty_formula"].apply(get_an_oxi_state)

list_data = data.to_dict('records')

for i in list_data:
    i['material_id'] = i.pop('task_id')

structs = []
for i in list_data:
    structs.append(database.parse_mprest(i))

# Generate the Database
species = []
cations = []
anions = []
for i in range(len(data)):
    species.append(data["pretty_formula"][i])
    cations.append((unparse_spec((data["elements"][i][0],
                                  int(data["cat_oxi_state"][i])))))
    anions.append((unparse_spec((data["elements"][i][1],
                                 int(data["an_oxi_state"][i])))))

columns = ["Pretty Formula", "M", "X"]
df_list = [[species[i], cations[i], anions[i]] for i in range(len(species))]
df = pd.DataFrame(data=df_list, columns=columns)

print(df.head())

# Store Data into SMACT Compatible Database
DB = database.StructureDB("BinaryMX.db")
# Create tables within the database
DB.add_table("Experimental")
DB.add_structs(structs, "Experimental")
