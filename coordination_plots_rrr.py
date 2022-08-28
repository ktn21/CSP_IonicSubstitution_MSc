import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


'''

Plotting the coordination numbers of the RRR dataset before and after the following filters:
- abs(Oxidation state) > 0 for all ions
- Lowest Ehull per composition for lowest energy structure
- Valid radius ratio between 0 and 1

Data can be obtained from:
- Before filters: binary_material_data_pull.py
- After filters: radius_ratio_rule.py

'''


chemenv_data = pd.read_csv("Binary_Dataset_for_RRRule_ChemEnv.csv")
robocrys_data = pd.read_csv("Binary_Dataset_for_RRRule_Robo.csv")
localenv_data = pd.read_csv("Binary_Dataset_for_RRRule_localenv.csv")

robocrys_int = robocrys_data[robocrys_data['coordination_num'] == robocrys_data['coordination_num'].astype(int)]
chemenv_int = chemenv_data[chemenv_data['coordination_num'] == chemenv_data['coordination_num'].astype(int)]

chemenv_cn = chemenv_int['coordination_num'].value_counts()
robocrys_cn = robocrys_int['coordination_num'].value_counts()
localenv_cn = localenv_data['coordination_num'].value_counts()

plt.figure()
robo_ce = robocrys_data['coordination_env'].value_counts()
plot1 = robo_ce.plot(kind='bar')
plot1.set_xlabel('Coordination Environment')
plot1.set_ylabel('Count')
plt.show()

plt.figure()
chemenv_ce = chemenv_data['coordination_env'].value_counts()
plot2 = chemenv_ce.plot(kind='bar')
plot2.set_xlabel('Coordination Environment')
plot2.set_ylabel('Count')
plt.show()

plt.figure()
plot3 = robocrys_cn.plot(kind='bar')
plot3.set_xlabel('Coordination Number')
plot3.set_ylabel('Count')
plt.show()

plt.figure()
X_axis = np.arange(len(localenv_cn))
plt.bar(localenv_cn.index, localenv_cn.values, 0.2, label='local_env')
plt.bar(chemenv_cn.index-0.2, chemenv_cn.values, 0.2, label='ChemEnv')
plt.bar(robocrys_cn.index+0.2, robocrys_cn.values, 0.2, label='Robocrys')
plt.xticks(range(len(localenv_cn)))
plt.xlabel("Coordination Number")
plt.ylabel("Count")
plt.legend()
plt.show()

# Filtered dataset
chemRRdata = pd.read_csv("Final_RR_Dataset_ChemEnv.csv")
localRRdata = pd.read_csv("Final_RR_Dataset_localenv.csv")
roboRRdata = pd.read_csv("Final_RR_Dataset_robo.csv")

chemenv_cn = chemRRdata['coordination_num'].value_counts()
robocrys_cn = localRRdata['coordination_num'].value_counts()
localenv_cn = roboRRdata['coordination_num'].value_counts()

plt.figure()
X_axis = np.arange(len(localenv_cn)+4)
plt.bar(localenv_cn.index, localenv_cn.values, 0.2, label='local_env')
plt.bar(chemenv_cn.index-0.2, chemenv_cn.values, 0.2, label='ChemEnv')
plt.bar(robocrys_cn.index+0.2, robocrys_cn.values, 0.2, label='Robocrys')
plt.xticks(range(len(localenv_cn)))
plt.xlabel("Coordination Number")
plt.ylabel("Count")
plt.legend()
plt.show()




