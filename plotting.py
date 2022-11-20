import dill as pickle
import os, shutil
from tableone import TableOne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

results_path = 'study_resultsassociation_study_n2c2_results.p'
ymaxvalue={'mice':2.5,'cca':5}

#results_path='study_resultsassociation_study_all_mimic_discharge.p'
#ymaxvalue={'mice':2.5,'cca':2}


with open(results_path, 'rb') as f:
    SELECTION, study_results_mice, study_results_cca = pickle.load(f)
    resname =results_path.split('/')[-1].split('.')[0]

determinants = ['employment_status', 'living_status','drug_status','alcohol_status','tobacco_status']
# Table 1
mytable = TableOne(SELECTION[['AGE', 'GENDER', 'ETHNICITY', 'RELIGION'] + determinants + ['DNR_ANY']], groupby='DNR_ANY', pval=True)
print(mytable.tabulate(tablefmt="fancy_grid"))

def get_est_and_ci(result, detname, setting, alpa=0.05):
    if setting == 'mice':
        det_indices = [i for i,name in enumerate(result.exog_names) if detname in name]
        names =  [name for i,name in enumerate(result.exog_names) if detname in name]
    else:
        det_indices = [i for i,name in enumerate(dict(result.params).keys()) if detname in name]
        names = [name for i,name in enumerate(dict(result.params).keys()) if detname in name]

    det_index = det_indices[0]
    det_name = names[0]
    print(names)
    #print(names[det_index])
    if setting =='mice':
        mu = result.params[det_index]
        ci = result.conf_int(alpa)[det_index]
    else:
        mu = result.params[det_index]
        #print('cca conf int',dict(result.conf_int(alpa)[0]))
        ci = dict(result.conf_int(alpa)[0])[det_name],dict(result.conf_int(alpa)[1])[det_name]
    print(setting,mu, ci)

    coeff_to_log_odds = lambda x: np.exp(x)

    return coeff_to_log_odds(mu),coeff_to_log_odds(ci[0]),coeff_to_log_odds(ci[1])



subplot_captions = {'employment_status':"(a) Employment status: employed",
                    'living_status':"(b) Living status: with family",
                    'drug_status':"(c) Drugs: current or past use",
                    'tobacco_status':"(d) Alcohol: current or past use",
                    'alcohol_status':"(e) Tobacco: current or past use"}

ylabels = {'employment_status': 'Odds Ratio',
                     'living_status': '',
                     'drug_status': 'Odds Ratio',
                     'alcohol_status': '',
                     'tobacco_status': ''
                     }

for study_results, missing_data_handling_method in [(study_results_mice, 'mice'),(study_results_cca,'cca')]:
    print(missing_data_handling_method)
    plotdir = 'plots/' + resname + '/' + missing_data_handling_method +'/'
    if os.path.exists(plotdir):
        shutil.rmtree(plotdir)
    os.makedirs(plotdir)

    gs = gridspec.GridSpec(7, 17)
    lefttop = plt.subplot(gs[:4, :8])
    righttop = plt.subplot(gs[:4, -8:])
    leftbottom = plt.subplot(gs[-2:, :5])
    middlebottom = plt.subplot(gs[-2:, 6:11])
    rightbottom = plt.subplot(gs[-2:, -5:])

    subplot_grids = {'employment_status': lefttop,
                     'living_status': righttop,
                     'drug_status': leftbottom,
                     'alcohol_status': middlebottom,
                     'tobacco_status': rightbottom
                     }

    for det in determinants:
        print(det)
        ax = subplot_grids[det]
        uw_mu, uw_lower, uw_upper = get_est_and_ci(study_results['S1'][det], det, missing_data_handling_method)
        if 'GT' in study_results:
           gt_mu, gt_lower, gt_upper = get_est_and_ci(study_results['GT'][det], det, missing_data_handling_method)

        Xs= [100,200,300,400,500]

        transfer_mus_and_cis = [get_est_and_ci(study_results['S6'][det], det, missing_data_handling_method), get_est_and_ci(study_results['S5'][det], det, missing_data_handling_method),get_est_and_ci(study_results['S4'][det], det, missing_data_handling_method),get_est_and_ci(study_results['S3'][det], det, missing_data_handling_method),get_est_and_ci(study_results['S2'][det], det, missing_data_handling_method) ]
        transfer_mus = [mu for mu,_,_ in transfer_mus_and_cis]
        transfer_lowers = [clower for _,clower,_ in transfer_mus_and_cis]
        transfer_uppers = [cupper for _,_,cupper in transfer_mus_and_cis]

        scratch_mus_and_cis = [get_est_and_ci(study_results['S11'][det], det, missing_data_handling_method), get_est_and_ci(study_results['S10'][det], det, missing_data_handling_method),get_est_and_ci(study_results['S9'][det], det, missing_data_handling_method),get_est_and_ci(study_results['S8'][det], det, missing_data_handling_method),get_est_and_ci(study_results['S7'][det], det, missing_data_handling_method) ]
        scratch_mus = [mu for mu,_,_ in scratch_mus_and_cis]
        scratch_lowers = [clower for _,clower,_ in scratch_mus_and_cis]
        scratch_uppers = [cupper for _,_,cupper in scratch_mus_and_cis]

        #xsize = 8
        #ysize = 5


        # transfer
        ax.plot([0] + Xs,[uw_mu] + transfer_mus, 'o', color='blue', linewidth=.9,markersize=3, linestyle='dashed')
        ax.fill_between([0] +Xs, [uw_lower]+ transfer_lowers, [uw_upper] + transfer_uppers, facecolor='blue', alpha=0.1)

        # scratch
        ax.plot(Xs,scratch_mus, 'o',  color='red', linewidth=.9,markersize=3,linestyle='dashed')
        ax.fill_between(Xs, scratch_lowers, scratch_uppers, facecolor='red', alpha=0.1)

        if 'GT' in study_results:
            ax.axhline(y=gt_mu, color='green', linewidth=.9) # GT
            ax.fill_between([0]+Xs, [gt_lower]*6,  [gt_upper]*6, facecolor='green', alpha=0.1)

        ax.axis(ymin=0, ymax=ymaxvalue[missing_data_handling_method])
        ax.set_ylabel(ylabels[det], fontfamily='serif', fontsize='x-small')
        #ax.set_xlabel()
        ax.set_title(subplot_captions[det], fontfamily='serif', loc='left', fontsize='x-small')

        ax.set_xticks([0] + Xs, ['0']+[str(x) for x in Xs], fontfamily='serif', fontsize='x-small')
        yticks = [x*ymaxvalue[missing_data_handling_method]/10 for x in range(1,10)]
        ax.set_yticks(yticks)#, fontfamily='serif', fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), fontfamily='serif', fontsize='x-small')
        ax.axhline(y=1, linestyle='dotted',color='grey',linewidth=0.8)

        print(ax.get_yticks())
        #plt.savefig(plotdir+ '/'+det+'_'+resname+'.png',dpi=300)
    plt.savefig(plotdir + '/'+ resname + '.png', dpi=300,  bbox_inches='tight')


