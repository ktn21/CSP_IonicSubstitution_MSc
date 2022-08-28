import smact
import pandas as pd
from pymatgen.core import Composition

'''

Searching for which ionic species have ionic and shannon radii, and their averaged values, within pymatgen in order to 
add them into the SMACT database!

'''


data = pd.read_csv("Binary_Dataset_for_RRRule_localenv.csv")

# Filter out materials with band_gap = 0
data = data[data.band_gap != 0]
data = data.reset_index(drop=True)

atom = []
oxistate = []
cn = []
averageionicradius = []
averageshannonradius = []
ionicradius = []
shannonradius = []

for i, row in enumerate(data.values):
    comp = Composition(data["pretty_formula"][i])
    elem1 = comp.chemical_system.partition("-")[0]
    atom.append(elem1)
    elem2 = comp.chemical_system.partition("-")[2]
    atom.append(elem2)
    cat_oxi_state = data["cat_oxi_state"][i]
    oxistate.append(cat_oxi_state)
    an_oxi_state = data["an_oxi_state"][i]
    oxistate.append(an_oxi_state)

    cn.append(int(data["coordination_num"][i]))
    cn.append(int(data["coordination_num"][i]))
    try:
        species1 = smact.Species(elem1,
                                 oxidation=int(cat_oxi_state),
                                 coordination=int(data["coordination_num"][i]))
        species2 = smact.Species(elem2,
                                 oxidation=int(an_oxi_state),
                                 coordination=int(data["coordination_num"][i]))
        averageionicradius.append(species1.average_ionic_radius)
        averageionicradius.append(species2.average_ionic_radius)
        averageshannonradius.append(species1.average_shannon_radius)
        averageshannonradius.append(species2.average_shannon_radius)
        ionicradius.append(species1.ionic_radius)
        ionicradius.append(species2.ionic_radius)
        shannonradius.append(species1.shannon_radius)
        shannonradius.append(species2.shannon_radius)
    except:
        averageionicradius.append(0)
        averageionicradius.append(0)
        averageshannonradius.append(0)
        averageshannonradius.append(0)
        ionicradius.append(0)
        ionicradius.append(0)
        shannonradius.append(0)
        shannonradius.append(0)

dataframe = pd.DataFrame(list(zip(atom, oxistate, cn, averageionicradius,
                                  averageshannonradius, ionicradius, shannonradius)),
                         columns=['atom', 'oxistate', 'cn', 'averageionicradius',
                                  'averageshannonradius', 'ionicradius', 'shannonradius'])

dataframe = dataframe.drop_duplicates(subset=['atom', 'oxistate', 'cn'], keep='first')
dataframe = dataframe.sort_values(by=['atom', 'oxistate', 'cn'])
dataframe = dataframe.reset_index(drop=True)

dataframe.to_csv("SMACT_Radii_by_AtomOxistateCoornum.csv")
