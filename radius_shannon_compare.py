import pandas as pd

'''
# Comparison of Performance between the Radius Ratio Rule and the Shannon Radius method within SMACT Structure Predictor

Radius Ratio Rule data obtained from: radius_ratio_rule.py
Shannon results can be obtained from: BinaryMX_StructurePredictor.py

'''

rrr_data = pd.read_csv('Final_RR_Dataset_localenv.csv')
shannon_data = pd.read_csv('BinaryMX_shannonLambda_pred_results.csv')
rrr_data = rrr_data.rename({'pretty_formula': 'Pretty Formula'}, axis=1)
merged_data = shannon_data.merge(rrr_data, how='left', on='Pretty Formula')

rrr_shannon_compare = merged_data.dropna()

missing_formula = rrr_data.merge(rrr_shannon_compare, how='left', on='Pretty Formula')
missing_formula = missing_formula[missing_formula['RR_y'].isnull()]
# Final List of compositions that are shared between the Shannon method and radius ratio rule
rrr_shannon_compare = rrr_shannon_compare[rrr_shannon_compare["RR_Prediction"].str.contains("Nonphysical") == False]

comparison = rrr_shannon_compare['Prediction'] == rrr_shannon_compare['RR_Prediction']
# Prediction Rate of Shannon radius within rrr_shannon_compare
shannon_rate = rrr_shannon_compare["Prediction"].value_counts().Correct/len(rrr_shannon_compare)*100
# Prediction Rate of Radius Ratio Rule within rrr_shannon_compare
rrr_rate = rrr_shannon_compare["RR_Prediction"].value_counts().Correct/len(rrr_shannon_compare)*100

print(f"The shannon prediction accuracy is {shannon_rate}%, and the radius ratio accuracy is {rrr_rate}%")
print(f"Of {len(comparison)} compositions, {comparison.sum()} compositions shared the same prediction result.")
