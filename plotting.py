import glob

import dill as pickle
import os, shutil, sys

import pandas
from tableone import TableOne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, r2_score

matplotlib.rcParams['font.family'] = ['serif'] #['Family1', 'serif', 'Family2']

topplotdir = './plots-29-03-2023/'
if not os.path.exists(topplotdir):
    os.makedirs(topplotdir)#shutil.rmtree(topplotdir)
#os.makedirs(topplotdir)

def get_est_and_ci(result, detname, setting, alpa=0.05):
    if 'mice' in setting:
        det_indices = [i for i,name in enumerate(result.exog_names) if detname in name]
        names =  [name for i,name in enumerate(result.exog_names) if detname in name]
    else:
        det_indices = [i for i,name in enumerate(dict(result.params).keys()) if detname in name]
        names = [name for i,name in enumerate(dict(result.params).keys()) if detname in name]

    det_index = det_indices[0]
    det_name = names[0]
    print(names)
    #print(names[det_index])
    if 'mice' in setting:
        mu = result.params[det_index]
        ci = result.conf_int(alpa)[det_index]
    else:
        mu = result.params[det_index]
        #print('cca conf int',dict(result.conf_int(alpa)[0]))
        ci = dict(result.conf_int(alpa)[0])[det_name],dict(result.conf_int(alpa)[1])[det_name]
    print(setting,mu, ci)

    if 'los' in setting:
        coeff_to_log_odds = lambda x: x
    else:
        coeff_to_log_odds = lambda x: np.exp(x)

    return coeff_to_log_odds(mu),coeff_to_log_odds(ci[0]),coeff_to_log_odds(ci[1])

#results_path='study_resultsassociation_study_all_mimic_discharge.p'
#ymaxvalue={'mice':2.5,'cca':2}

results_path = sys.argv[1] # first argument of the script

ymaxvalue={'los_mice':2,'los_cca':2,'dnr_mice':2,'dnr_cca':5, 'mort_mice':3.0,'mort_cca':5}
yminvalue={'los_mice':-3,'los_cca':-3,'dnr_mice':0,'dnr_cca':0, 'mort_mice':0,'mort_cca':0}

with open(results_path, 'rb') as f:
    MEASURES, PATHS, SELECTIONS, study_results_all = pickle.load(f)

for measurement_error_recalibration_setting in ['determinant_recalibration', 'none']:
    for adjustment_setting in ['crude', 'adjusted']:

        print('>>>>>>', measurement_error_recalibration_setting, adjustment_setting)

        study_results_los_mice = study_results_all['LOS'][adjustment_setting]['mice'][measurement_error_recalibration_setting]
        study_results_los_cca = study_results_all['LOS'][adjustment_setting]['cca'][measurement_error_recalibration_setting]
        study_results_dnr_mice = study_results_all['DNR_ANY'][adjustment_setting]['mice'][measurement_error_recalibration_setting]
        study_results_dnr_cca = study_results_all['DNR_ANY'][adjustment_setting]['cca'][measurement_error_recalibration_setting]
        study_results_mort_mice = study_results_all['HOSPITAL_EXPIRE_FLAG'][adjustment_setting]['mice'][measurement_error_recalibration_setting]
        study_results_mort_cca = study_results_all['HOSPITAL_EXPIRE_FLAG'][adjustment_setting]['cca'][measurement_error_recalibration_setting]

        resname =results_path.split('/')[-1].split('.')[0] + '_' + adjustment_setting + 'mmec='+str(measurement_error_recalibration_setting)

        determinants = ['employment_status', 'living_status','drug_status','alcohol_status','tobacco_status']


        #print(SELECTIONS.keys())
        #print(SELECTIONS['BERT-1'])


        #print(pandas.crosstab(SELECTIONS['GT'].alcohol_status, SELECTIONS['GT'].HOSPITAL_EXPIRE_FLAG))

        #print(pandas.crosstab(SELECTIONS['BERT-1'].alcohol_status, SELECTIONS['BERT-1'].HOSPITAL_EXPIRE_FLAG))


        #exit()


        # Get precise F1 scores

        DEF_MEASURES = {d:{} for d in determinants}

        if 'GT' in PATHS:
            pos_labels = {'employment_status':'employed', 'living_status':'with_family','drug_status':'current_or_past','alcohol_status':'current_or_past','tobacco_status':'current_or_past'}
            for det in determinants:
                gt = [1 if v==pos_labels[det] else 0 for v in SELECTIONS['GT'][det].values]
                gt_mort = [1 if v==pos_labels[det] else 0 for v in SELECTIONS['GT'].loc[SELECTIONS['GT']['HOSPITAL_EXPIRE_FLAG'] == 1][det].values] # df.loc[df['col1'] == value]
                gt_surv = [1 if v==pos_labels[det] else 0 for v in SELECTIONS['GT'].loc[SELECTIONS['GT']['HOSPITAL_EXPIRE_FLAG'] == 0][det].values] # df.loc[df['col1'] == value]
                for m in SELECTIONS:
                    if not m=='GT':

                        print(m)
                        pred = [1 if v==pos_labels[det] else 0 for v in SELECTIONS[m][det].values]
                        pred_mort = [1 if v==pos_labels[det] else 0 for v in SELECTIONS[m].loc[SELECTIONS[m]['HOSPITAL_EXPIRE_FLAG'] == 1][det].values]
                        pred_surv = [1 if v==pos_labels[det] else 0 for v in SELECTIONS[m].loc[SELECTIONS[m]['HOSPITAL_EXPIRE_FLAG'] == 0][det].values]
                        p,r,f,_ = precision_recall_fscore_support(gt, pred, average='binary')

                        p_mort, r_mort, f_mort,_ = precision_recall_fscore_support(gt_mort, pred_mort, average='binary')
                        p_surv, r_surv, f_surv,_ = precision_recall_fscore_support(gt_surv, pred_surv, average='binary')

                        CITL = np.mean(gt) - np.mean(pred)
                        citl_mort, citl_surv = np.mean(gt_mort) - np.mean(pred_mort), np.mean(gt_surv) - np.mean(pred_surv)

                        acc = accuracy_score(gt, pred)
                        DEF_MEASURES[det][m]= {'Precision':p,'Recall':r,'F1-score':f,'CITL':CITL, 'Accuracy':acc, 'Precision_mort':p_mort,
                                               'Recall_mort':r_mort,'F1-score_mort':f_mort,  'Precision_surv':p_surv, 'Recall_surv':r_surv,'F1-score_surv':f_surv,
                                               'CITL_mort':citl_mort, 'CITL_surv':citl_surv
                                               }

        # ======================== F1 PLOT ========================

        print(PATHS.keys())
        if 'GT' in PATHS:
            fig = plt.figure(constrained_layout=True, figsize=(6, 2))
            #gs = fig.add_gridspec(ncols=2, nrows=2)
            gs = fig.add_gridspec(ncols=3, nrows=1)
            left = plt.subplot(gs[0, 0])
            middle = plt.subplot(gs[0, 1])
            right = plt.subplot(gs[0, 2])

            subplot_grids_prf = {'Precision': left,
                             'Recall': middle,
                             'F1-score': right
                             }
            Xs = [0,100,200,300,400,500]
            for measure in ['Precision','Recall','F1-score']:
                ax = subplot_grids_prf[measure]
                handles = []
                for prefix, markerstyle, colorstyle in [('LSTM-', 'o', 'blue'),('BERT-', 'D', 'red')]:

                    # transfer
                    ptr, = ax.plot([0,100,200,300,400,500],[MEASURES[measure][prefix+'1'],MEASURES[measure][prefix+'6'],MEASURES[measure][prefix+'5'],MEASURES[measure][prefix+'4'],MEASURES[measure][prefix+'3'],MEASURES[measure][prefix+'2']], markerstyle,  color=colorstyle, linewidth=.6,markersize=2,linestyle='solid',label='Fine-tuned '+prefix.replace('-',''))
                    handles.append(ptr)

                    # scratch
                    pts, = ax.plot([100,200,300,400,500],[MEASURES[measure][prefix+'11'],MEASURES[measure][prefix+'10'],MEASURES[measure][prefix+'9'],MEASURES[measure][prefix+'8'],MEASURES[measure][prefix+'7']], markerstyle,  color=colorstyle, linewidth=.6,markersize=2,linestyle='dashed',label='Retrained '+prefix.replace('-',''))
                    handles.append(pts)

                ax.set_ylabel(measure, fontsize='x-small')
                ax.set_xticks([0] + Xs, ['0']+[str(x) for x in Xs], fontsize='x-small')
                #ax.yticks(fontsize='x-small')
                ax.tick_params(axis='both', which='both', labelsize='x-small')
                ax.set_ylim(0.45,0.85)
                if measure =='Precision':
                    ax.legend(handles=handles, prop={'size': 6})


                #ax.set_yticks(fontfamily=fontfam, fontsize='x-small')

            plt.savefig(topplotdir + '/performance_plot.png', dpi=300) #,  bbox_inches='tight')
            plt.cla()
            plt.clf()
            plt.close()






        # ======================== TABLE 1 ========================
        if 'GT' in SELECTIONS:
            mytable = TableOne(SELECTIONS['GT'][['AGE', 'GENDER', 'ETHNICITY', 'RELIGION'] + determinants + ['DNR_ANY','HOSPITAL_EXPIRE_FLAG','LOS']]) #, pval=True)#groupby='DNR_ANY'
            print(mytable.tabulate(tablefmt="fancy_grid"))


        # ======================== ODDS RATIO PLOTS ========================
        subplot_captions = {'employment_status':"(a) Employment status: employed",
                            'living_status':"(b) Living status: with family",
                            'drug_status':"(c) Drugs: current or past use",
                            'tobacco_status':"(d) Alcohol: current or past use",
                            'alcohol_status':"(e) Tobacco: current or past use"}
        ylabels_cont = {'employment_status': 'Beta coefficient',
                             'living_status': '',
                             'drug_status': 'Beta coefficient',
                             'alcohol_status': '',
                             'tobacco_status': ''
                             }


        ylabels_bin = {'employment_status': 'Odds Ratio',
                             'living_status': '',
                             'drug_status': 'Odds Ratio',
                             'alcohol_status': '',
                             'tobacco_status': ''
                             }


        settings = [(study_results_los_mice, 'los_mice'),(study_results_los_cca,'los_cca'),(study_results_dnr_mice, 'dnr_mice'),(study_results_dnr_cca,'dnr_cca'),(study_results_mort_mice, 'mort_mice'),(study_results_mort_cca,'mort_cca')]

        VALUES = {setting:{d:{} for d in determinants} for _, setting in settings}
        GTS = {setting:{d:{} for d in determinants} for _, setting in settings}

        for study_results, missing_data_handling_method in settings:
            print(resname,missing_data_handling_method)
            #print(study_results)
            plotdir = topplotdir + resname + '/' + missing_data_handling_method +'/'
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
                handles=[]
                ax = subplot_grids[det]
                if 'GT' in study_results:
                   gt_mu, gt_lower, gt_upper = get_est_and_ci(study_results['GT'][det], det, missing_data_handling_method)

                Xs = [100, 200, 300, 400, 500]

                print(study_results.keys())
                i=0
                #handles = []
                for prefix, markerstyle, colorstyle in [('LSTM-', 'o', 'blue'),('BERT-', 'D', 'red')]:
                    #print(study_results.keys())
                    uw_mu, uw_lower, uw_upper = get_est_and_ci(study_results[prefix+'1'][det], det, missing_data_handling_method)
                    uw_err = uw_upper-uw_lower
                    VALUES[missing_data_handling_method][det][prefix+'1']= uw_mu
                    GTS[missing_data_handling_method][det][prefix+'1']= gt_mu


                    transfer_mus_and_cis = [get_est_and_ci(study_results[prefix+'6'][det], det, missing_data_handling_method), get_est_and_ci(study_results[prefix+'5'][det], det, missing_data_handling_method),get_est_and_ci(study_results[prefix+'4'][det], det, missing_data_handling_method),get_est_and_ci(study_results[prefix+'3'][det], det, missing_data_handling_method),get_est_and_ci(study_results[prefix+'2'][det], det, missing_data_handling_method) ]
                    transfer_mus = [mu for mu,_,_ in transfer_mus_and_cis]
                    transfer_lowers = [clower for _,clower,_ in transfer_mus_and_cis]
                    transfer_uppers = [cupper for _,_,cupper in transfer_mus_and_cis]
                    transfer_err = [cupper-clower for _,clower,cupper in transfer_mus_and_cis]
                    for num,est in zip(['6','5','4','3','2'],transfer_mus):
                        GTS[missing_data_handling_method][det][prefix + num] = gt_mu
                        VALUES[missing_data_handling_method][det][prefix + num] =  est

                    transfer_biases = {gt_mu-tr_mu for tr_mu in transfer_mus}

                    scratch_mus_and_cis = [get_est_and_ci(study_results[prefix+'11'][det], det, missing_data_handling_method), get_est_and_ci(study_results[prefix+'10'][det], det, missing_data_handling_method),get_est_and_ci(study_results[prefix+'9'][det], det, missing_data_handling_method),get_est_and_ci(study_results[prefix+'8'][det], det, missing_data_handling_method),get_est_and_ci(study_results[prefix+'7'][det], det, missing_data_handling_method) ]
                    scratch_mus = [mu for mu,_,_ in scratch_mus_and_cis]
                    scratch_lowers = [clower for _,clower,_ in scratch_mus_and_cis]
                    scratch_uppers = [cupper for _,_,cupper in scratch_mus_and_cis]
                    scratch_err = [cupper-clower for _,clower,cupper in scratch_mus_and_cis]
                    for num,est in zip(['11','10','9','8','7'],scratch_mus):
                        GTS[missing_data_handling_method][det][prefix + num] = gt_mu
                        VALUES[missing_data_handling_method][det][prefix + num] = est

                    #xsize = 8
                    #ysize = 5

                    dev=i*10
                    # transfer #
                    ptr = ax.errorbar([0+dev] + [j+dev for j in Xs],[uw_mu] + transfer_mus, marker=markerstyle, color=colorstyle, linewidth=.6,markersize=2, linestyle='solid',capsize=2,capthick=0.6,label='Fine-tuned '+prefix.replace('-',''))
                    #ax.errorbar([0+dev] + [j+dev for j in Xs],[uw_mu] + transfer_mus, yerr=[[uw_lower]+transfer_lowers,[uw_upper]+transfer_uppers], marker=markerstyle, color=colorstyle, linewidth=.6,markersize=2, linestyle='solid',capsize=2,capthick=0.6,label='Finetuned-'+prefix.replace('-',''))
                    handles.append(ptr)
                    #ax.fill_between([0] +Xs, [uw_lower]+ transfer_lowers, [uw_upper] + transfer_uppers, facecolor=colorstyle, alpha=0.1)


                    # scratch # , yerr=[scratch_lowers,scratch_uppers]
                    pts = ax.errorbar([j+dev+dev for j in Xs],scratch_mus, marker=markerstyle,  color=colorstyle, linewidth=.6,markersize=2,linestyle='dashed',capsize=2,capthick=0.6,label='Retrained '+prefix.replace('-',''))
                    #ax.errorbar([j+dev+dev for j in Xs],scratch_mus, yerr=[scratch_lowers,scratch_uppers], marker=markerstyle,  color=colorstyle, linewidth=.6,markersize=2,linestyle='dashed',capsize=2,capthick=0.6,label=prefix.replace('-',''))
                    handles.append(pts)
                    i += 2
                    #ax.fill_between(Xs, scratch_lowers, scratch_uppers, facecolor=colorstyle, alpha=0.1)

                if 'GT' in study_results:
                    gtr = ax.axhline(y=gt_mu, color='green', linewidth=.9, label='Reference (annotated)') # GT
                    handles.append(gtr)
                    #ax.fill_between([-50]+[j+50 for j in Xs], [gt_lower]*6,  [gt_upper]*6, facecolor='green', alpha=0.1) # confidence interval

                ax.axis(ymin=yminvalue[missing_data_handling_method], ymax=ymaxvalue[missing_data_handling_method])
                if 'los' in missing_data_handling_method:
                    ax.set_ylabel(ylabels_cont[det], fontsize='x-small')
                else:
                    ax.set_ylabel(ylabels_bin[det], fontsize='x-small')
                #ax.set_xlabel()
                ax.set_title(subplot_captions[det], loc='left', fontsize='x-small')

                ax.set_xticks([0] + Xs, ['0']+[str(x) for x in Xs], fontsize='x-small')

                if 'los' in missing_data_handling_method:
                    ntr = ax.axhline(y=0, linestyle='dotted',color='grey',linewidth=0.8)
                    yticks = [yminvalue[missing_data_handling_method]+(x * (ymaxvalue[missing_data_handling_method]-yminvalue[missing_data_handling_method]) / 10) for x in range(1, 10)]
                    ax.set_yticks(yticks)  # , fontfamily='serif', fontsize=8)
                else:
                    ntr = ax.axhline(y=1, linestyle='dotted',color='grey',linewidth=0.8)
                    yticks = [yminvalue[missing_data_handling_method]+(x * (ymaxvalue[missing_data_handling_method]-yminvalue[missing_data_handling_method]) / 10) for x in range(1, 10)]
                    ax.set_yticks(yticks)  # , fontfamily='serif', fontsize=8)
                handles.append(ntr)
                ax.set_yticklabels(ax.get_yticklabels(), fontsize='x-small')

                ax.set_xlim(left=-50, right=550)
                if det == 'employment_status':
                    ax.legend(handles=handles, prop={'size': 6})

                #print(ax.get_yticks())
                #plt.savefig(plotdir+ '/'+det+'_'+resname+'.png',dpi=300)
            plt.savefig(plotdir + '/'+ resname + '_' + missing_data_handling_method + '_association_plot.png', dpi=300,  bbox_inches='tight')
            plt.cla()
            plt.clf()


        # ======================== F1 against % dir of effect ========================
        if 'GT' in PATHS:
            clin_relevant_threshold = 0.5
            abs_bias_color_norm = plt.Normalize(0, clin_relevant_threshold)
            for setting in VALUES:
                plotdir = topplotdir + resname + '/' + setting +'/'

                ests = VALUES[setting]
                gts = GTS[setting]


                # DEF F1 AGAINST ABSOLUTE BIAS
                for prefix, markerstyle, colorstyle in [('LSTM-', 'o', 'blue'),('BERT-', 'D', 'red')]:
                    depairs = [(d, m) for d in determinants for m in ests[d] if prefix in m]
                    bias = [gts[d][m]-ests[d][m] for d,m in depairs]
                    #print(list((d,DEF_MEASURES[d].keys()) for d,m in depairs))
                    f1s = [DEF_MEASURES[d][m]['F1-score'] for d,m in depairs]
                    plt.scatter(f1s, bias, color=colorstyle, alpha=0.7)
                    #plt.hist2d(f1s, abs_bias)
                plt.savefig(plotdir + '/'+ resname + '_' + setting + '_def_f1_abs_bias_plot_modelgrouped.png', dpi=300,  bbox_inches='tight')
                plt.cla()
                plt.clf()



                # DEF CITL AGAINST ABSOLUTE BIAS
                for prefix, markerstyle, colorstyle in [('LSTM-', 'o', 'blue'),('BERT-', 'D', 'red')]:
                    depairs = [(d, m) for d in determinants for m in ests[d] if prefix in m]
                    bias = [gts[d][m]-ests[d][m] for d,m in depairs]
                    citls = [DEF_MEASURES[d][m]['CITL'] for d,m in depairs]
                    plt.scatter(citls, bias, color=colorstyle, alpha=0.7)
                    #plt.hist2d(citls, abs_bias, color=colorstyle)
                plt.savefig(plotdir + '/'+ resname + '_' + setting + '_def_CITL_abs_bias_plot_modelgrouped.png', dpi=300,  bbox_inches='tight')
                plt.cla()
                plt.clf()

                # DEF ACC AGAINST ABSOLUTE BIAS: MODELGROUPED
                for prefix, markerstyle, colorstyle in [('LSTM-', 'o', 'blue'),('BERT-', 'D', 'red')]:
                    depairs = [(d, m) for d in determinants for m in ests[d] if prefix in m]
                    bias = [gts[d][m]-ests[d][m] for d,m in depairs]
                    accs = [DEF_MEASURES[d][m]['Accuracy'] for d,m in depairs]
                    plt.scatter(accs, bias, color=colorstyle, alpha=0.7)
                    #plt.hist2d(accs, abs_bias, color=colorstyle)
                plt.savefig(plotdir + '/'+ resname + '_' + setting + '_def_accuracy_abs_bias_plot_modelgrouped.png', dpi=300,  bbox_inches='tight')
                plt.cla()
                plt.clf()

                # DEF Prec AGAINST ABSOLUTE BIAS
                for det, colorstyle in zip(determinants,['red','blue','green','black','purple']): #  ['employment_status', 'living_status','drug_status','alcohol_status','tobacco_status']
                    depairs = [(det, m) for m in ests[det]]
                    bias = [gts[d][m]-ests[d][m] for d,m in depairs]
                    precs = [DEF_MEASURES[d][m]['Precision'] for d,m in depairs]
                    plt.scatter(precs, bias, color=colorstyle, alpha=0.7)
                    #plt.hist2d(f1s, abs_bias)
                plt.savefig(plotdir + '/'+ resname + '_' + setting + '_def_prec_abs_bias_plot_detgrouped.png', dpi=300,  bbox_inches='tight')
                plt.cla()
                plt.clf()

                # DEF Recall AGAINST ABSOLUTE BIAS
                for det, colorstyle in zip(determinants,['red','blue','green','black','purple']): #  ['employment_status', 'living_status','drug_status','alcohol_status','tobacco_status']
                    depairs = [(det, m) for m in ests[det]]
                    bias = [gts[d][m]-ests[d][m] for d,m in depairs]
                    recs = [DEF_MEASURES[d][m]['Recall'] for d,m in depairs]
                    plt.scatter(recs, bias, color=colorstyle, alpha=0.7)
                    #plt.hist2d(f1s, abs_bias)
                plt.savefig(plotdir + '/'+ resname + '_' + setting + '_def_rec_abs_bias_plot_detgrouped.png', dpi=300,  bbox_inches='tight')
                plt.cla()
                plt.clf()

                # DEF diff AGAINST ABSOLUTE BIAS
                for det, colorstyle in zip(determinants,['red','blue','green','black','purple']): #  ['employment_status', 'living_status','drug_status','alcohol_status','tobacco_status']
                    depairs = [(det, m) for m in ests[det]]
                    bias = [gts[d][m]-ests[d][m] for d,m in depairs]
                    diffs = [DEF_MEASURES[d][m]['Recall']-DEF_MEASURES[d][m]['Precision'] for d,m in depairs]
                    plt.scatter(diffs, bias, color=colorstyle, alpha=0.7)
                    #plt.hist2d(f1s, abs_bias)
                plt.savefig(plotdir + '/'+ resname + '_' + setting + '_def_diffs_abs_bias_plot_detgrouped.png', dpi=300,  bbox_inches='tight')
                plt.cla()
                plt.clf()

                # Mort F1 vs Surv F1 diff AGAINST ABSOLUTE BIAS
                depairs = [(det, m) for m in ests[det] for det in determinants]
                bias = [abs(gts[d][m]-ests[d][m]) for d,m in depairs]
                points = plt.scatter([DEF_MEASURES[d][m]['F1-score_mort'] for d,m in depairs], [DEF_MEASURES[d][m]['F1-score_surv'] for d,m in depairs], c=bias, cmap="jet", alpha=0.8, norm=abs_bias_color_norm)
                plt.colorbar(points)
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                plt.axline((0, 0), slope=1, linestyle='--', color='red')
                plt.savefig(plotdir + '/'+ resname + '_' + setting + '_def_mortf1_survf1_abs_bias_plot.png', dpi=300,  bbox_inches='tight')
                plt.cla()
                plt.clf()

                # CITL F1 vs CITL F1 diff AGAINST ABSOLUTE BIAS
                depairs = [(det, m) for m in ests[det] for det in determinants]
                bias = [abs(gts[d][m]-ests[d][m]) for d,m in depairs]
                points = plt.scatter([DEF_MEASURES[d][m]['CITL_mort'] for d,m in depairs], [DEF_MEASURES[d][m]['CITL_surv'] for d,m in depairs], c=bias, cmap="jet", alpha=0.8, norm=abs_bias_color_norm)
                plt.colorbar(points)
                #plt.xlim([0, .2])
                #plt.ylim([0, .2])
                plt.axline((0, 0), slope=1, linestyle='--', color='red')
                plt.savefig(plotdir + '/'+ resname + '_' + setting + '_def_mortcitl_survcitl_abs_bias_plot.png', dpi=300,  bbox_inches='tight')
                plt.cla()
                plt.clf()

                # CITL F1 vs CITL F1 diff AGAINST ABSOLUTE BIAS
                depairs = [(det, m) for m in ests[det] for det in determinants]
                bias = [abs(gts[d][m]-ests[d][m]) for d,m in depairs]
                points = plt.scatter([abs(DEF_MEASURES[d][m]['CITL_mort']) for d,m in depairs], [abs(DEF_MEASURES[d][m]['CITL_surv']) for d,m in depairs], c=bias, cmap="jet", alpha=0.8, norm=abs_bias_color_norm)
                plt.colorbar(points)
                #plt.xlim([0, .2])
                #plt.ylim([0, .2])
                plt.axline((0, 0), slope=1, linestyle='--', color='red')
                plt.savefig(plotdir + '/'+ resname + '_' + setting + '_def_absmortcitl_abssurvcitl_abs_bias_plot.png', dpi=300,  bbox_inches='tight')
                plt.cla()
                plt.clf()


                # OVERALL F1 AGAINST ABSOLUTE BIAS: MODELGROUPED
                for prefix, markerstyle, colorstyle in [('LSTM-', 'o', 'blue'),('BERT-', 'D', 'red')]:
                    depairs = [(d, m) for d in determinants for m in ests[d] if prefix in m]
                    bias = [gts[d][m]-ests[d][m] for d,m in depairs]
                    f1s = [MEASURES['F1-score'][m] for d,m in depairs]
                    plt.scatter(f1s, bias, color=colorstyle, alpha=0.7)

                    #plt.hist2d(f1s, abs_bias, color=colorstyle)
                plt.savefig(plotdir + '/'+ resname + '_' + setting + '_overall_f1_abs_bias_plot.png', dpi=300,  bbox_inches='tight')
                plt.cla()
                plt.clf()




            # Plot DNR and LOS next to each other: Fig 6
            gs = gridspec.GridSpec(1, 17)
            left = plt.subplot(gs[:, :8])
            right = plt.subplot(gs[:, -8:])
            plt_grid = {'dnr_mice':left,'los_mice':right}
            plotdir = topplotdir + resname + '/'
            for setting in plt_grid:
                ax = plt_grid[setting]
                ests = VALUES[setting]
                gts = GTS[setting]

                for det, colorstyle in zip(determinants,['red','blue','green','black','purple']): #  ['employment_status', 'living_status','drug_status','alcohol_status','tobacco_status']
                    depairs = [(det, m) for m in ests[det]]
                    absbias = [abs(gts[d][m]-ests[d][m]) for d,m in depairs]
                    f1s = [DEF_MEASURES[d][m]['F1-score'] for d,m in depairs]
                    ax.scatter(f1s, absbias, color=colorstyle, alpha=0.8)
                    print('f1s,absbias:',f1s,absbias)
                    m, b = np.polyfit(f1s, absbias, deg=1)
                    r2 = r2_score(absbias, np.array(f1s)*m+b)
                    num_spaces = max([len(determinants)])-len(det)
                    label = det[0].title() + det[1:].replace('_',' ') + ' '*num_spaces*2 + '\t' + f'$y = {m:.2f}x {b:+.2f}$' + '\t' + f'$r^2 = {r2:.2f}$'
                    ax.axline(xy1=(0, b), slope=m, color=colorstyle, label=label,linewidth=0.7, alpha=0.8)
                ax.set_xlim(0,1)
                ax.set_ylim(-0.1, 3)
                if 'los' in setting:
                    ax.set_ylabel("Absolute error in beta coefficient")#, fontsize='small')
                else:
                    ax.set_ylabel("Absolute error in odds ratio")#, fontsize='small')
                ax.set_xlabel("F1-score")#, fontsize='small')
                ax.legend(fontsize='small')
            #    ax.set_aspect('equal')
                ax.get_figure().set_figwidth(12)
                #ax.get_figure().set_size_inches(15,15)
            plt.savefig(plotdir + '/'+ resname + '_def_f1_abs_bias_plot_detgrouped.png', dpi=300)#,  bbox_inches='tight')
            plt.cla()
            plt.clf()


            # !!! Plots BINS Fig 7
            gs = gridspec.GridSpec(1, 17)
            left = plt.subplot(gs[:, :8])
            right = plt.subplot(gs[:, -8:])
            plt_grid = {'dnr_mice':left,'los_mice':right}
            plotdir = topplotdir + resname + '/'
            for setting in plt_grid:
                print(setting)
                ax = plt_grid[setting]
                ests = VALUES[setting]
                gts = GTS[setting]
                depairs = [(det, m) for m in ests[det] for det in determinants]
                if 'los' in setting:
                    signcorrect = [np.sign(gts[d][m])== np.sign(ests[d][m]) for d, m in depairs] # for beta coefficients the center is 0
                else:
                    signcorrect = [np.sign(gts[d][m]-1)== np.sign(ests[d][m]-1) for d, m in depairs] # OR center is 1 instead of 0

                f1s = [DEF_MEASURES[d][m]['F1-score'] for d, m in depairs]
                bins = [('<0.5',[0,0.5]),('0.5-0.6',[0.5,0.6]),('0.6-0.7',[0.6,0.7]),('0.7-0.8',[0.7,0.8]),('0.8-0.9',[0.8,0.9]),('0.9-1.0',[0.9,1.0])]

                xlabs = []
                ys = []
                for i,(bin_name, (b_min,b_max)) in enumerate(bins):
                    binvalues = [v for v,f1 in zip(signcorrect,f1s) if f1 >= b_min and f1 < b_max]
                    binsize = len(binvalues)
                    perc_correct = np.mean(binvalues)
                    print(bin_name, perc_correct)
                    xlabs.append(bin_name + '('+str(binsize)+')')
                    ys.append(perc_correct)
                ax.bar(xlabs, ys)
                    #ax.bar(i, perc_correct, label=str(binsize), tick_label=bin_name)

                #ax.get_figure().set_size_inches(15,15)
            plt.savefig(plotdir + '/'+ resname + '_def_f1_bins_dir_plot.png', dpi=300)#,  bbox_inches='tight')
            plt.cla()
            plt.clf()





