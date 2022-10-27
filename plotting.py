import dill as pickle
import os, shutil
from tableone import TableOne
import numpy as np
import matplotlib.pyplot as plt

results_path = 'study_resultsassociation_study_n2c2_results.p'
#results_path='study_resultsassociation_study_all_mimic_discharge.p'

with open(results_path, 'rb') as f:
    SELECTION, study_results = pickle.load(f)
    resname =results_path.split('/')[-1].split('.')[0]

covars = ['AGE', 'GENDER', 'ETHNICITY', 'RELIGION']
determinants = ['employment_status', 'tobacco_status', 'alcohol_status', 'drug_status', 'living_status']

SELECTION['AGE'] = SELECTION['AGE'].astype(float)

plotdir='plots/'+resname
if os.path.exists(plotdir):
    shutil.rmtree(plotdir)
os.makedirs(plotdir)

# Table 1
mytable = TableOne(SELECTION[determinants + covars + ['DNR_ANY']], groupby='DNR_ANY', pval=True)
print(mytable.tabulate(tablefmt="fancy_grid"))

def get_est_and_ci(result, detname, alpa=0.05):
    det_indices = [i for i,name in enumerate(result.exog_names) if detname in name]
    det_index = det_indices[0]
    mu = result.params[det_index]
    ci = result.conf_int(alpa)[det_index]
    return mu,ci[0],ci[1]

for det in determinants:
    print(det)

    uw_mu, uw_lower, uw_upper = get_est_and_ci(study_results['S1'][det], det)
    if 'GT' in study_results:
       gt_mu, gt_lower, gt_upper = get_est_and_ci(study_results['GT'][det], det)

    Xs= [100,200,300,400,500]

    transfer_mus_and_cis = [get_est_and_ci(study_results['S6'][det], det), get_est_and_ci(study_results['S5'][det], det),get_est_and_ci(study_results['S4'][det], det),get_est_and_ci(study_results['S3'][det], det),get_est_and_ci(study_results['S2'][det], det) ]
    transfer_mus = [mu for mu,_,_ in transfer_mus_and_cis]
    transfer_lowers = [clower for _,clower,_ in transfer_mus_and_cis]
    transfer_uppers = [cupper for _,_,cupper in transfer_mus_and_cis]

    scratch_mus_and_cis = [get_est_and_ci(study_results['S11'][det], det), get_est_and_ci(study_results['S10'][det], det),get_est_and_ci(study_results['S9'][det], det),get_est_and_ci(study_results['S8'][det], det),get_est_and_ci(study_results['S7'][det], det) ]
    scratch_mus = [mu for mu,_,_ in scratch_mus_and_cis]
    scratch_lowers = [clower for _,clower,_ in scratch_mus_and_cis]
    scratch_uppers = [cupper for _,_,cupper in scratch_mus_and_cis]

    xsize = 8
    ysize = 5

    plt.figure(figsize=(xsize, ysize))
    # transfer
    plt.plot([0] + Xs,[uw_mu] + transfer_mus, 'o', color='blue', linewidth=1, linestyle='dashed')
    plt.fill_between([0] +Xs, [uw_lower]+ transfer_lowers, [uw_upper] + transfer_uppers, facecolor='blue', alpha=0.1)

    # scratch
    plt.plot(Xs,scratch_mus, 'o',  color='red', linewidth=1, linestyle='dashed')
    plt.fill_between(Xs, scratch_lowers, scratch_uppers, facecolor='red', alpha=0.1)

    if 'GT' in study_results:
        plt.axhline(y=gt_mu, color='green') # GT
        plt.fill_between(Xs, [gt_lower]*5,  [gt_upper]*5, facecolor='green', alpha=0.1)

    plt.ylim(bottom=-1.5, top=1.5)
    plt.xticks([0] + Xs, ['0']+[str(x) for x in Xs])
    plt.savefig(plotdir+ '/'+det+'_'+resname+'.png',dpi=300)


