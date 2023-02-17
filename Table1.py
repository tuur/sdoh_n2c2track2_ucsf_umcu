import dill as pickle
import os, shutil
from tableone import TableOne
import numpy as np
import matplotlib.pyplot as plt
import openpyxl

results_path_n2c2 = 'study_resultsassociation_study_n2c2_results.p'
results_path_mimic='study_resultsassociation_study_all_mimic_discharge.p'

results = {}

with open(results_path_n2c2, 'rb') as f:
    results['n2c2']= pickle.load(f)[0]
with open(results_path_mimic, 'rb') as f:
    results['mimic']= pickle.load(f)[0]


determinants = ['employment_status', 'tobacco_status', 'alcohol_status', 'drug_status', 'living_status']
cats = determinants + ['DNR_ANY'] + ['ETHNICITY', 'RELIGION'] + ['GENDER','ADMISSION_TYPE']

# Table 1
print('N2C2')
n2c2_table = TableOne(results['n2c2'][['AGE', 'GENDER','ADMISSION_TYPE'] + ['DNR_ANY'] + determinants + ['ETHNICITY', 'RELIGION']], nonnormal=['AGE'], pval=False)
print(n2c2_table.tabulate(tablefmt="fancy_grid"))
n2c2_table.to_excel('n2c2_table1.xlsx')

for r in results['n2c2']['N2C2_ID']:
    print(r)


results['mimic']['AGE'] = results['mimic']['AGE'].astype(float)

print('MIMIC')
mimic_table = TableOne(results['mimic'][['AGE', 'GENDER','ADMISSION_TYPE'] + ['DNR_ANY'] + determinants+ ['ETHNICITY', 'RELIGION']], categorical=cats, nonnormal=['AGE'],pval=False)
print(mimic_table.tabulate(tablefmt="fancy_grid"))
mimic_table.to_excel('mimic_table1.xlsx')