import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from myCSPfunctions import prediction_num, prediction_correctness


'''

Plotting the heatmap for correct/incorrect predictions of binary MX materials for each SMACT Structure Predictor method.

'''


def sort_heatmap_bynan(tab):
    sort_vector = tab.isna().sum(axis=0).T
    sort_index = tab.index
    sort_index = {x: y for x, y in enumerate(sort_index)}

    tab = tab.append(sort_vector, ignore_index=True)
    tab = tab.sort_values(by=len(tab) - 1, ascending=True, axis=1)
    tab = tab.drop(len(tab) - 1)
    tab = tab.rename(sort_index)

    sort_vector = tab.isna().sum(axis=1)
    tab = tab.join(sort_vector.rename("Sum"))
    tab = tab.sort_values(by="Sum", ascending=True)
    output_tab = tab.drop(columns=["Sum"])

    figsize = [10, 14]
    plt.figure(figsize=figsize)
    ax = sns.heatmap(output_tab,
                     cmap=sns.diverging_palette(10, 150, as_cmap=True),
                     linewidths=0.5,
                     linecolor="k",
                     )
    return ax


# Load Data
control = pd.read_csv("BinaryMX_controlLambda_pred_results.csv")
sradius = pd.read_csv("BinaryMX_shannonLambda_pred_results.csv")
mat2vec = pd.read_csv("BinaryMX_mat2vecLambda_pred_results.csv")
megnet = pd.read_csv("BinaryMX_megnet16Lambda_pred_results.csv")
mod_petti = pd.read_csv("BinaryMX_mod_pettiLambda_pred_results.csv")
crabnet = pd.read_csv("BinaryMX_crabnet_Lambda_pred_results.csv")

control_plot = prediction_correctness(control)
shannon_plot = prediction_correctness(sradius)
mat2vec_tab = prediction_correctness(mat2vec)
megnet_tab = prediction_correctness(megnet)
mod_petti_tab = prediction_correctness(mod_petti)
crabnet_tab = prediction_correctness(crabnet)

# Plotting
plot1 = sort_heatmap_bynan(control_plot)
plot1.set_title("Control Results")
plt.savefig('Control Results.png')

plot2 = sort_heatmap_bynan(shannon_plot)
plot2.set_title("Shannon Radius Model Results")
plt.savefig('Shannon Results.png')

plot3 = sort_heatmap_bynan(mat2vec_tab)
plot3.set_title("mat2vec Results")
plt.savefig('mat2vec Results.png')

plot4 = sort_heatmap_bynan(megnet_tab)
plot4.set_title("MEGNet Results")
plt.savefig('MEGNet Results.png')

plot5 = sort_heatmap_bynan(mod_petti_tab)
plot5.set_title("Modified Pettifor Results")
plt.savefig('mod petti Results.png')

plot6 = sort_heatmap_bynan(crabnet_tab)
plot6.set_title("CrabNet Results")
plt.savefig('crabnet Results.png')
