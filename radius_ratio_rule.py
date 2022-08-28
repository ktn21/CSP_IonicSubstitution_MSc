# Radius Ratio Rule Implementation for Binary 1:1 Ionic Materials
import smact
import pandas as pd
from pymatgen.core import Composition
import matplotlib.pyplot as plt
import re

# Create radius ratio structure table
'''
d = {'Radius Ratio Upper': [1.0, 1.0, 0.732, 0.732, 0.414, 0.225],
     'Radius Ratio Lower': [1.0, 0.732, 0.414, 0.414, 0.225, 0.155],
     'Coordination Number': [12, 8, 6, 4, 4, 3],
     'Coordination': ['CCP/HCCP', 'Cubic', 'Octahedral', 'Square Planar', 'Tetrahedral', 'Triangular']
     }
'''

d = {'Radius Ratio Upper': [1.0, 1.0, 0.717, 0.717, 0.326, 0.225],
     'Radius Ratio Lower': [1.0, 0.717, 0.326, 0.326, 0.225, 0.155],
     'Coordination Number': [12, 8, 6, 4, 4, 3],
     'Coordination': ['CCP/HCCP', 'Cubic', 'Octahedral', 'Square Planar', 'Tetrahedral', 'Triangular']
     }

RR_table = pd.DataFrame(data=d)

data = pd.read_csv("Binary_Dataset_for_RRRule_Robo.csv")

# Bar chart of coordination numbers in the dataset
plt.figure()
binaryCN_data = data['coordination_num'].value_counts()
plot1 = binaryCN_data.plot(kind='bar')
plot1.set_xlabel('Coordination Number')
plot1.set_ylabel('Count')
plt.show()

# Filter out materials with band_gap = 0
# data = data[data.band_gap != 0]
# Filter out materials without an oxidation state
interdata = data[data.cat_oxi_state != 0]
interdata = interdata.reset_index(drop=True)

prediction = []
RR_ratio = []
# Prediction of Coordination Numbers
for i, row in enumerate(interdata.values):
    try:
        comp = Composition(interdata["pretty_formula"][i])
        elem1 = comp.chemical_system.partition("-")[0]
        elem2 = comp.chemical_system.partition("-")[2]
        cat_oxi_state = interdata["cat_oxi_state"][i]
        an_oxi_state = interdata["an_oxi_state"][i]
        species1 = smact.Species(elem1,
                                 oxidation=int(cat_oxi_state),
                                 coordination=int(interdata["coordination_num"][i]))
        species2 = smact.Species(elem2,
                                 oxidation=int(an_oxi_state),
                                 coordination=int(interdata["coordination_num"][i]))
        radius_ratio = species1.average_ionic_radius/species2.average_ionic_radius
    except:
        radius_ratio = 0

    RR_ratio.append(radius_ratio)

    # Check if radius ratio rule prediction is correct
    for j in range(len(RR_table)):
        rr_range = [d["Radius Ratio Upper"][j], d["Radius Ratio Lower"][j]]
        if radius_ratio <= 0 or pd.isnull(radius_ratio) or radius_ratio >= 1:
            prediction.append("Nonphysical")
            break
        elif rr_range[0] >= radius_ratio >= rr_range[1]:
            if d["Coordination Number"][j] == interdata["coordination_num"][i]:
                prediction.append("Correct")
                break
            elif j < max(range(len(RR_table))):
                k = j + 1
                rr_range_temp = [d["Radius Ratio Upper"][k], d["Radius Ratio Lower"][k]]
                if rr_range_temp[0] == rr_range[0]:
                    if d["Coordination Number"][k] == interdata["coordination_num"][i]:
                        prediction.append("Correct")
                        break
                    else:
                        prediction.append("Incorrect")
                        break
                else:
                    prediction.append("Incorrect")
                    break
            else:
                prediction.append("Incorrect")
                break
        elif j == RR_table.tail(1).index.item():
            prediction.append("Incorrect")

interdata["RR"] = RR_ratio
interdata["RR_Prediction"] = prediction

# Filter out rows with nan as radius ratio
finaldata = interdata.dropna()
# Sort by e_above_hull for stability per chemical formula
finaldata = finaldata.sort_values(by=['pretty_formula', 'e_above_hull'])
# Check lowest e_above_hull state only
finaldata = finaldata.drop_duplicates(subset=['pretty_formula'], keep='first')
finaldata = finaldata.reset_index(drop=True)

finaldata.to_csv("Final_RR_Dataset_Robo.csv")

print("The percent of correctly predicted structures is "
      + str(finaldata["RR_Prediction"].value_counts().Correct/len(finaldata)*100) + "%")

# Bar Chart of correctly predicted coordination numbers
plt.figure()
prediction_data = finaldata['RR_Prediction'].value_counts()
prediction_plot = prediction_data.plot(kind='bar', stacked=True)
plt.show()
