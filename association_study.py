import argparse, re, pandas, collections, math, numpy
import statsmodels.formula.api as smf
from dateutil.parser import parse as parsedatetime
from brat_scoring.corpus import Corpus
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.imputation.mice import MICEData, MICE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import statsmodels.api as sm
from copy import copy
from tableone import TableOne
from glob import glob
# statsmodels.imputation.mice.MICEData
import dill as pickle
import sys, os, shutil
old_stdout = sys.stdout
import gc
from brat_scoring.constants import EXACT, LABEL, OVERLAP
from brat_scoring.scoring import score_brat_sdoh, score_docs, micro_average_subtypes_csv
from brat_scoring.constants_sdoh import LABELED_ARGUMENTS as SDOH_LABELED_ARGUMENTS

class AssociationStudy():

    def __init__(self, df, xvars, yvar, imp_vars, cca=False, linear_outcome=False, det_error_calibration=False, det_error_calibration_gt_df=False, det_error_calibration_pred_df=False, det_error_vars=[]):
        all_relevant_vars = list(set(xvars + [yvar] + imp_vars + det_error_vars))
        self.df = df[all_relevant_vars]
        self.xvars=xvars
        self.yvar=yvar
        self.results=None
        self.imp_vars=imp_vars
        self.det_error_calibration=det_error_calibration
        self.det_error_calibration_gt_df=det_error_calibration_gt_df
        self.det_error_calibration_pred_df=det_error_calibration_pred_df
        if self.det_error_calibration:
            self.det_error_vars = det_error_vars + ['*_'+det_error_calibration]

            print(det_error_calibration_gt_df[det_error_calibration])
            print(det_error_calibration_pred_df[det_error_calibration])



        if cca:
            self.df = self.df.dropna(subset=all_relevant_vars)


        # TODO!!!! for regression recalibration we also need to include the extracted X* to fit X <- X*,Z

        if det_error_calibration: # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7450672/ (section 6)
            # set target det as outcome
            det_error_calibration_pred_df['*_'+det_error_calibration]=det_error_calibration_pred_df[det_error_calibration]
            print(det_error_calibration_pred_df.columns)
            # add the predicted determinant X* to the gt dataframe
            det_error_calibration_gt_df = pandas.merge(det_error_calibration_gt_df, det_error_calibration_pred_df[['EVENTNOTEROWID','*_'+det_error_calibration]], on=['EVENTNOTEROWID'], how='left')

            det_error_calibration_df = det_error_calibration_gt_df[[det_error_calibration] + self.det_error_vars].dropna() # use cca data for recalibration model (missings are not recalibrated anyway)
            print(det_error_calibration_df[det_error_calibration])
            print(det_error_calibration)
            det_dummies = pandas.get_dummies(det_error_calibration_gt_df[self.det_error_calibration], drop_first=True, prefix=det_error_calibration)
            det_dummy_names = det_dummies.columns
            mmec_dummies = pandas.get_dummies(det_error_calibration_gt_df[self.det_error_vars], drop_first=True)
            mmec_dummies['Intercept']=1
            mmec_dummy_names = mmec_dummies.columns
            recal_model = GLM(det_dummies,mmec_dummies , family=families.Binomial()).fit(attach_wls=True, atol=1e-10)
            print('DETERMINANT RECALIBRATION MODEL:')
            print(recal_model.summary())
            print(det_dummy_names)
            #self.df[det_error_calibration] = recal_model.predict(self.df[imp_vars])# recalibrated variable using X <- X*
            #print(self.df[det_error_calibration])

        self.X_df = pandas.get_dummies(self.df[self.xvars], drop_first=True)
        if not cca:
            self.imp_df = pandas.get_dummies(self.df[self.imp_vars], drop_first=True)
            imp_vars_names = self.imp_df.columns

        self.y_df = self.df[self.yvar]
        #self.Xy_df = copy(self.X_df)

        self.Xy_df = pandas.concat([self.X_df, self.y_df], axis=1)

        #self.Xy_df[self.yvar]=self.y_df

        if not cca: # add potentially missing imputation var columns
            newcolumns = [colname for colname in self.imp_df.columns if not colname in self.Xy_df.columns]
            self.Xy_df = pandas.concat([self.Xy_df, self.imp_df[newcolumns]], axis=1)


        if det_error_calibration: # add potentially missing MMEC var columns
            self.df['*_'+det_error_calibration]=self.df[det_error_calibration]
            self.mmec_df = pandas.get_dummies(self.df[self.det_error_vars], drop_first=True)
            self.mmec_df['Intercept']=1
            newcolumns = [colname for colname in mmec_dummy_names if not colname in self.Xy_df.columns]
            self.Xy_df = pandas.concat([self.Xy_df, self.mmec_df[newcolumns]], axis=1)
            print(self.Xy_df.columns)
            print(self.Xy_df[list(det_dummy_names)])
            print(mmec_dummy_names)

            newDet = recal_model.predict(self.Xy_df[list(mmec_dummy_names)])
            print(newDet)
            self.Xy_df[list(det_dummy_names)[0]] =newDet
            print(det_dummy_names)

            print('!')
            #exit()

        if cca:
            print('CCA ANALYSIS')
            self.X_df['Intercept']=1
            self.formula = self.yvar + ' ~ ' + ' + '.join(['"'+v+'"' for v in self.xvars])
            print('\n\n',60*'=','> outcome events:',self.df[self.yvar].sum(),'\n',self.formula)
            if linear_outcome:
                model = sm.OLS(self.y_df,self.X_df).fit()
            else:
                model = GLM(self.y_df,self.X_df, family=families.Binomial()).fit(attach_wls=True, atol=1e-10)
            print(model.summary())
            self.results=model
        else:
            print('MICE ANALYSIS')
            imp = MICEData(self.Xy_df, k_pmm=10)
            for xvar in self.Xy_df.columns:
                imp.set_imputer(xvar, xvar + ' ~ ' + ' + '.join(imp_vars_names))
            fml = self.yvar + ' ~ ' + ' + '.join(self.X_df.columns)
            print(fml)
            print('impvars',self.imp_vars)
            if linear_outcome:
                mice = MICE(fml, sm.OLS, imp)
            else:
                mice = MICE(fml, sm.GLM, imp, init_kwds={"family": sm.families.Binomial()})
            results = mice.fit(10, 10)
            print(results.summary())
            self.results=results

        print(cca,'SIZE:',sys.getsizeof(self.results))

def add_n2c2_note_ids(admissions, file_alignment):
    alignment = pandas.read_csv(file_alignment)
    alignment_dict = {int(r[1]['MIMIC ROW_ID']):r[1]['SHAC FILENAME'] for r in alignment.iterrows()}
    alignment_dict_complete = {r[1]['EVENTNOTEROWID']:alignment_dict[r[1]['EVENTNOTEROWID']] if r[1]['EVENTNOTEROWID'] in alignment_dict else False for r in admissions.iterrows()}
    alignment_dict_complete[numpy.nan]=False
    admissions['N2C2_ID'] = admissions['EVENTNOTEROWID'].replace(alignment_dict_complete)
    return admissions

def add_patient_info(admissions, patients_path):
    patients = pandas.read_csv(patients_path)

    # gender
    gender = {r[1]['SUBJECT_ID']:r[1]['GENDER'] for r in patients.iterrows()}
    admissions['GENDER'] = admissions['SUBJECT_ID'].map(gender)

    # date of birth (dob)
    dob = {r[1]['SUBJECT_ID']:r[1]['DOB'] for r in patients.iterrows()}
    admissions['DOB'] = admissions['SUBJECT_ID'].map(dob)

    # ! Careful: patients > 89 have age 300 in MIMIC III (https://github.com/MIT-LCP/mimic-code/issues/170)
    admissions['AGE'] = admissions.apply(lambda e: int(round((parsedatetime(e['ADMITTIME']).year - parsedatetime(e['DOB']).year), 0)), axis=1)

    return admissions

def add_dnr_status(admissions, dnr_status_path):
    DNR_OUTCOMESET = set(['CPR Not Indicated','Do Not Intubate','Do Not Resuscitate']) #'Comfort Measures',
    with open(dnr_status_path, 'r') as f:
        dnr_status = {int(l.split(',')[0]):l.split(',')[1].strip() for l in f.readlines()}

    dnr_status_composite = {hadm: dnr_status[hadm] in DNR_OUTCOMESET if hadm in dnr_status else False for hadm in admissions['HADM_ID']}
    admissions['DNR'] = admissions['HADM_ID'].map(dnr_status_composite)
    return admissions

def add_annotated_dnr_status(admissions, dnr_ann_dir):
    #print(add_annotated_dnr_status)
    ids = {}
    for pred_path in glob(dnr_ann_dir + "/*.ann", recursive=False):
        eventnoteid = int(pred_path.split("/")[-1].replace(".ann",""))
        #print(eventnoteid, pred_path)

        with open(pred_path, 'r') as f:
            annotation_txt = f.read()
            ids[eventnoteid] = True if ("DNR" in annotation_txt or "DNI" in annotation_txt) else False
            #print(annotation_txt)
            #print(ids[eventnoteid])

        #ids.add(eventnoteid)
    #print(len(ids))

    #idsinadm = [id for id in ids if id in admissions['EVENTNOTEROWID']]
    #idsnotinadm = [id for id in ids if not id in admissions['EVENTNOTEROWID']]

    admissions['ANNOTATED_DNR_DNI'] = admissions['EVENTNOTEROWID'].map(ids)

    return admissions

def extract_code_status_from_discharge_summary(admissions): # TODO: bugfix, lowertxt seems to be float sometimes...

    txt_dnr = {}
    for r in admissions.iterrows():
        txt = r[1]['DISCHARGE_SUMMARY']
        hadmid = int(r[1]['HADM_ID'])
        txt_dnr[hadmid] = False
        if isinstance(txt, str):
            codestatuslines = re.findall(r'.+dnr.*\n',txt.lower())
            codestatuslines += re.findall(r'.+dni.*\n',txt.lower()) # added dni
            codestatuslines += re.findall(r'.+do[ -]not[ -]resuscitate.+\n',txt.lower())
            codestatuslines += re.findall(r'.+do not intubate.+\n',txt.lower())
            #codestatuslines += re.findall(r'.+code status.+\n',txt.lower())
            codestatuslines += re.findall(r'.+code.+\n',txt.lower())

            for codestatusline in codestatuslines:
                if (('dnr' in codestatusline or 'dni ' in codestatusline or 'dni\n' in codestatusline or 'do-not-resuscitate' in codestatusline or 'do not resuscitate' in codestatusline or 'do not intubate' in codestatusline) ): # added dni
                    txt_dnr[hadmid]= True
            #if 'DNI ' in txt or 'DNI\n' in txt:
            #    txt_dnr[hadmid] = True


    admissions['TXT_DNR'] = admissions['HADM_ID'].map(txt_dnr)
    return admissions

def add_soc_hist_and_note_id(admissions, mimic_discharge_sochist_path, mimic_noteevents):
    mimic_discharge_sochist = pandas.read_csv(mimic_discharge_sochist_path)
    eventnoterowids = pandas.read_csv(mimic_noteevents,usecols=["ROW_ID", "HADM_ID","CATEGORY","TEXT"])

    row_id_to_hadmid = {int(row[1]['ROW_ID']):int(row[1]['HADM_ID']) for row in eventnoterowids.iterrows() if not math.isnan(row[1]['HADM_ID'])}
    hadm_to_sochhist = {row_id_to_hadmid[int(row[1]['ROW_ID'])]:row[1]['SOCIAL_HISTORY'] for row in mimic_discharge_sochist.iterrows()}
    rowid_to_dischargesum = {int(row[1]['ROW_ID']):row[1]['TEXT'] for row in eventnoterowids.iterrows() if (row[1]['CATEGORY']).lower()=='discharge summary'}
    hadm_to_eventnoterowid = {row_id_to_hadmid[int(rowid)]:int(rowid) for rowid in rowid_to_dischargesum}
    hadm_to_dischargesum = {row_id_to_hadmid[rowid]:rowid_to_dischargesum[rowid] for rowid in rowid_to_dischargesum}

    admissions['SOCIAL_HISTORY_SECTION'] = admissions['HADM_ID'].map(hadm_to_sochhist)
    admissions['EVENTNOTEROWID'] = admissions['HADM_ID'].map(hadm_to_eventnoterowid)
    admissions['DISCHARGE_SUMMARY'] = admissions['HADM_ID'].map(hadm_to_dischargesum)

    return admissions

def add_first_addmission_column(admissions):
    pid_to_first_admission_date = {}
    for adm in admissions.iterrows():
        pid, admtime = adm[1]['SUBJECT_ID'], adm[1]['ADMITTIME']
        if not pid in pid_to_first_admission_date:
            pid_to_first_admission_date[pid] = admtime
        else:
            if admtime < pid_to_first_admission_date[pid]:
                pid_to_first_admission_date[pid] = admtime
    hadm_to_first = {}
    for adm in admissions.iterrows():
        pid, admtime = adm[1]['SUBJECT_ID'], adm[1]['ADMITTIME']
        if admtime == pid_to_first_admission_date[pid]:
            hadm_to_first[adm[1]['HADM_ID']]= True
        else:
            hadm_to_first[adm[1]['HADM_ID']] = False

    admissions['FIRST_ADMISSION'] = admissions['HADM_ID'].map(hadm_to_first)
    return admissions

def exclude_n2c2_rows(admissions, ann_file_dir_of_files_to_be_excluded, n2c2_alignment_path):
    corpus = Corpus()
    corpus.import_dir(path=ann_file_dir_of_files_to_be_excluded)
    docs = corpus.docs(as_dict=True)
    alignment = pandas.read_csv(n2c2_alignment_path)
    alignment_dict = {int(r[1]['SHAC FILENAME'].split('/')[-1]):int(r[1]['MIMIC ROW_ID']) for r in alignment.iterrows()}
    to_exclude_rowids = {alignment_dict[int(doc_id)] for doc_id, _ in docs.items()}

    admissions['TO_EXCLUDE'] = admissions.apply(lambda row: row.EVENTNOTEROWID in to_exclude_rowids, axis=1)

    print(sum(admissions['TO_EXCLUDE']))
    return admissions

def read_sdoh_annotations(admissions, ann_file_dir, n2c2_alignment_path, n2c2_alignment=False):

    print('reading',ann_file_dir)
    sdoh_data = []

    corpus = Corpus()
    corpus.import_dir(path=ann_file_dir)
    docs = corpus.docs(as_dict=True)

    if n2c2_alignment:
        alignment = pandas.read_csv(n2c2_alignment_path)
        alignment_dict = {int(r[1]['SHAC FILENAME'].split('/')[-1]):int(r[1]['MIMIC ROW_ID']) for r in alignment.iterrows()}

    for doc_id, doc in docs.items():
        employment_status, tobacco_status, drug_status, alcohol_status, living_status = None, None, None, None, None
        if n2c2_alignment:
            doc_id=alignment_dict[int(doc_id)]

        for e in doc.events():
            if e.type_ =='LivingStatus':
                current=False
                status=False
                for a in e.arguments:
                    if a.type_=='TypeLiving':
                        status=a.subtype
                    if a.type_=='StatusTime' and a.subtype=='current':
                        current=True
                if current and status:
                    living_status=status
            if e.type_ =='Alcohol':
                status='current_or_past'
                for a in e.arguments:
                    if a.type_=='StatusTime' and a.subtype!=None:
                        status=a.subtype
                alcohol_status=status
            if e.type_ =='Drug':
                status='current_or_past'
                for a in e.arguments:
                    if a.type_=='StatusTime' and a.subtype!=None:
                        status=a.subtype
                drug_status=status
            if e.type_ =='Tobacco':
                status='current_or_past'
                for a in e.arguments:
                    if a.type_=='StatusTime' and a.subtype!=None:
                        status=a.subtype
                tobacco_status=status
            if e.type_=='Employment':
                status='employed'
                for a in e.arguments:
                    if a.type_=='StatusEmploy':
                        status=a.subtype
                employment_status=status

        sdoh_data += [[int(doc_id), employment_status, tobacco_status, drug_status, alcohol_status, living_status, True]]


    sdoh_df = pandas.DataFrame(sdoh_data, columns=['EVENTNOTEROWID','employment_status','tobacco_status','drug_status','alcohol_status','living_status','sdoh_extracted'])

    # remove the columns that are going to be added
    for column_name in sdoh_df.columns:
        if column_name in admissions.columns and not column_name=='EVENTNOTEROWID':
            admissions = admissions.drop(column_name, axis=1)

    admissions=pandas.merge(admissions, sdoh_df, on=['EVENTNOTEROWID'], how='left')
    # TODO; CHECK IF THOSE 93 IN THE DIFFERENCE ARE outside of the age range.
    return sdoh_df, admissions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('-mimic_file_alignment', required=False, help='csv aligning mimic note event ids and note ids', default="/Users/aleeuw15/Desktop/Research/N2C2 - SDOH/JAMIA paper/code/data/MIMIC_file_alignment.csv")
    parser.add_argument('-mimic_patients', required=False, help='...', default="/Users/aleeuw15/Desktop/Research/NLP - TextImp/datasets/mimic/PATIENTS.csv")
    parser.add_argument('-mimic_admissions', required=False, help='...', default="/Users/aleeuw15/Desktop/Research/NLP - TextImp/datasets/mimic/ADMISSIONS.csv")
    parser.add_argument('-mimic_noteevents', required=False, help='...', default="/Users/aleeuw15/Desktop/Research/NLP - TextImp/datasets/mimic/NOTEEVENTS.csv")
    parser.add_argument('-dnr_codes', required=False, help='...', default='HADM_ID_DNR_status.csv')
    parser.add_argument('-excl_dir', required=False, help='Directory where the .ann files are stored that are used for text mining fine tuning and should be excluded', default="/Users/aleeuw15/Desktop/Research/N2C2 - SDOH/JAMIA paper/code/data/association_study data splits/mimic_train_500/")
    parser.add_argument('-mmec_pred', required=False, help='Directory where the .ann extracted files are stored for the 500 tuning set (needed for measurement error calibration)', default="/Users/aleeuw15/Desktop/Research/N2C2 - SDOH/npj Dig Med/results/NLP/500SUBSET/")
    parser.add_argument('-mimic_discharge_sochist', required=False, help='...', default="/Users/aleeuw15/Desktop/Research/N2C2 - SDOH/JAMIA paper/code/data/mimic_discharge_sochist.csv")
    parser.add_argument('-n2c2_alignment', required=False, help='Use n2c2 txt ids (if ann dir uses n2c2 ids instead of mimic row ids).', type=int)
    parser.add_argument('-load_df', required=False, help='Directory where the .ann files are stored.')
    #parser.add_argument('-ann_dir', required=False, help='Directory where the .ann files are stored.', default="/Users/aleeuw15/Desktop/Research/N2C2 - SDOH/JAMIA paper/code/data/association_study SHAC/S4/")
    parser.add_argument('-dnr_ann_dir', required=False, help='Directory where the .ann files are stored for the DNR/DNI codes.', default="/Users/aleeuw15/Desktop/Research/N2C2 - SDOH/public_repo/sdoh_n2c2track2_ucsf_umcu/dnr_annotations/")
    parser.add_argument('-pred_ann_dirs', required=False, help='Directory where the .ann files are stored.')

    args = parser.parse_args()
    out_dir = [x for x in args.pred_ann_dirs.split('/') if not x == ''][-1]
    log_file = open('study_results'+out_dir+'.log', "w")
    #sys.stdout = log_file # TODO: uncomment line

    if not(args.load_df):

        # Obtain admission information in a single dataframe
        ADMISSIONS = pandas.read_csv(args.mimic_admissions)
        ADMISSIONS = add_first_addmission_column(ADMISSIONS)
        ADMISSIONS = add_dnr_status(ADMISSIONS, args.dnr_codes) # 2% outcome prevalence ...
        ADMISSIONS = add_soc_hist_and_note_id(ADMISSIONS, args.mimic_discharge_sochist, args.mimic_noteevents)
        ADMISSIONS = add_patient_info(ADMISSIONS, args.mimic_patients)
        ADMISSIONS = add_n2c2_note_ids(ADMISSIONS, args.mimic_file_alignment)
        ADMISSIONS = extract_code_status_from_discharge_summary(ADMISSIONS)
        #ADMISSIONS = ADMISSIONS[ADMISSIONS['FIRST_ADMISSION']==True] Not needed: as code status can be different and also SDOH can be different at second admissions.

        with open('admissions_df.p', 'wb') as f:
            pickle.dump(ADMISSIONS,f)

    with open(args.load_df, 'rb') as f:
        ADMISSIONS = pickle.load(f)
        ADMISSIONS = exclude_n2c2_rows(ADMISSIONS, args.excl_dir, args.mimic_file_alignment)
        ADMISSIONS = add_annotated_dnr_status(ADMISSIONS, args.dnr_ann_dir) # TODO: possibly move this to before loading admissions df
        print('ADMISSIONS initial:', len(ADMISSIONS))
        ADMISSIONS = ADMISSIONS[~ADMISSIONS['EVENTNOTEROWID'].isnull()]
        print('ADMISSIONS with linkable notes:', len(ADMISSIONS))
        # select only 18-69
        ADMISSIONS = ADMISSIONS[ADMISSIONS['AGE'] >= 18]
        ADMISSIONS = ADMISSIONS[ADMISSIONS['AGE'] <= 89]
        print('ADMISSIONS after age 18-89:', len(ADMISSIONS))

        print('ADMISSIONS minus fine-tuning data:',len(ADMISSIONS))
        ADMISSIONS['DNR_ANY'] = ADMISSIONS.apply(lambda row: int(row.DNR or row.TXT_DNR), axis=1) # otherwise use regex
        print('ADMISSIONS DNR prevalence (struct+regex):', round(sum(ADMISSIONS['DNR_ANY'])/len(ADMISSIONS)*100,2),'%')
        print('ADMISSIONS MORT prevalence:', round(sum(ADMISSIONS['HOSPITAL_EXPIRE_FLAG'])/len(ADMISSIONS)*100,2),'%')
        ADMISSIONS['LOS'] = (ADMISSIONS['DISCHTIME'].apply(pandas.to_datetime)-ADMISSIONS['ADMITTIME'].apply(pandas.to_datetime)).dt.days



    #study_results_dnr_mice_crude, study_results_dnr_cca_crude, study_results_dnr_mice_adj, study_results_dnr_cca_adj = {}, {}, {}, {}
    #study_results_mort_mice_crude, study_results_mort_cca_crude, study_results_mort_mice_adj, study_results_mort_cca_adj = {}, {}, {}, {}
    #study_results_los_mice_crude, study_results_los_cca_crude, study_results_los_mice_adj, study_results_los_cca_adj = {}, {}, {}, {}

    adjustment_settings = ['crude', 'adjusted']
    measurement_error_handling = ['determinant_recalibration','none']
    missing_data_mechanisms = ['mice', 'cca']
    outcomes = ['HOSPITAL_EXPIRE_FLAG','LOS','DNR_ANY']
    study_results = {o:{adj: {mdm: {mmec:{} for mmec in measurement_error_handling} for mdm in missing_data_mechanisms} for adj in adjustment_settings} for o in outcomes}

    SELECTIONS = {}
    PATHS = {}
    for pred_path in glob(args.pred_ann_dirs+"/*", recursive = False):
        #print(pred_path)
        dirname = pred_path.split('/')[-1]
        PATHS[dirname]=pred_path

    for dirname, pred_path in PATHS.items():
        print('>>>>', dirname)
        #with open(args.load_df, 'rb') as f:
        #    ADMISSIONS = pickle.load(f)

        # reset SDOH status
        ADMISSIONS['sdoh_extracted'] = ADMISSIONS.apply(lambda row: pandas.NA, axis=1)

        print('pre',ADMISSIONS.columns)
        _, ADMISSIONS = read_sdoh_annotations(ADMISSIONS, pred_path, args.mimic_file_alignment, args.n2c2_alignment)
        print('post',ADMISSIONS.columns)


        # excluded documents used to train / fine tune the text mining models
        #SUBSET500 = ADMISSIONS[ADMISSIONS['TO_EXCLUDE']==True]
        #print('SUBSET500 length:',len(SUBSET500))
        _, SUBSET500 = read_sdoh_annotations(ADMISSIONS, args.excl_dir, args.mimic_file_alignment, args.n2c2_alignment)
        _,SUBSET500EXTR = read_sdoh_annotations(ADMISSIONS, args.mmec_pred + dirname +'/', args.mimic_file_alignment, args.n2c2_alignment)
        print(args.mmec_pred +'/' +dirname +'/')

        SUBSET500 = SUBSET500[SUBSET500['sdoh_extracted']==True]
        SUBSET500EXTR = SUBSET500EXTR[SUBSET500EXTR['sdoh_extracted']==True]

        print('SUBSET500 length:',len(SUBSET500))
        print('SUBSET500EXTR length:',len(SUBSET500EXTR))

        #ADMISSIONS = ADMISSIONS[ADMISSIONS['TO_EXCLUDE']==False]
        SELECTION = ADMISSIONS[ADMISSIONS['sdoh_extracted']==True]
        SELECTION = SELECTION[SELECTION['TO_EXCLUDE']==False]

        print('SELECTION DNR prevalence (struct+regex):', round(sum(SELECTION['DNR_ANY'])/len(SELECTION)*100,2),'%')

        #SELECTION = extract_code_status_from_discharge_summary(SELECTION) # TODO: possibly move this to before loading admissions df

        if (args.n2c2_alignment):
            SELECTION['DNR_ANY'] = SELECTION.apply(lambda row: int(row.DNR or row.ANNOTATED_DNR_DNI), axis=1) # use DNR/DNI annotations for N2C2 subset
            print('SELECTION DNR prevalence (struct+txt_ann):', round(sum(SELECTION['DNR_ANY'])/len(SELECTION)*100,2),'%')

        SELECTION['AGE'] = SELECTION['AGE'].astype(float)

        print('SELECTION:',len(SELECTION))

        # -------> calculate length of stay

        #SELECTION['LOS'] = (SELECTION['DISCHTIME'].apply(pandas.to_datetime)-SELECTION['ADMITTIME'].apply(pandas.to_datetime)).dt.days

        # <--------
        print('SELECTION DNR prevalence:', round(sum(SELECTION['DNR_ANY'])/len(SELECTION)*100,2),'%')
        print('SELECTION DNR N:', round(sum(SELECTION['DNR_ANY']),2))

        print('SELECTION MORT prevalence:', round(sum(SELECTION['HOSPITAL_EXPIRE_FLAG'])/len(SELECTION)*100,2),'%')
        print('SELECTION MORT N:', round(sum(SELECTION['HOSPITAL_EXPIRE_FLAG']),2))

        print('employment_status',collections.Counter(SELECTION['employment_status']))
        print('tobacco_status',collections.Counter(SELECTION['tobacco_status']))
        print('drug_status',collections.Counter(SELECTION['drug_status']))
        print('alcohol_status',collections.Counter(SELECTION['alcohol_status']))
        print('living_status',collections.Counter(SELECTION['living_status']))

        # composite predictors
        employment_composite = {'employed':'employed','unemployed':'REST','on_disability':'REST','retired':'REST',None:None,'student':'REST','homemaker':'REST'}
        SELECTION['employment_status']=SELECTION.apply(lambda row: employment_composite[row.employment_status], axis=1)
        SUBSET500['employment_status']=SUBSET500.apply(lambda row: employment_composite[row.employment_status], axis=1)
        print(SUBSET500EXTR.columns)
        SUBSET500EXTR['employment_status']=SUBSET500EXTR.apply(lambda row: employment_composite[row.employment_status], axis=1)


        living_composite = {'homeless':'REST','with_family':'with_family','with_others':'REST','alone':'REST',None:None}
        SELECTION['living_status']=SELECTION.apply(lambda row: living_composite[row.living_status], axis=1)
        SUBSET500['living_status']=SUBSET500.apply(lambda row: living_composite[row.living_status], axis=1)
        SUBSET500EXTR['living_status']=SUBSET500EXTR.apply(lambda row: living_composite[row.living_status], axis=1)

        tobacco_composite = {'current':'current_or_past','none':'REST','past':'current_or_past','current_or_past':'current_or_past',None:None,'future':'REST'}
        SELECTION['tobacco_status']=SELECTION.apply(lambda row: tobacco_composite[row.tobacco_status], axis=1)
        SUBSET500['tobacco_status']=SUBSET500.apply(lambda row: tobacco_composite[row.tobacco_status], axis=1)
        SUBSET500EXTR['tobacco_status']=SUBSET500EXTR.apply(lambda row: tobacco_composite[row.tobacco_status], axis=1)

        alcohol_composite = {'current':'current_or_past','none':'REST','past':'current_or_past','current_or_past':'current_or_past',None:None, 'future':'REST'}
        SELECTION['alcohol_status']=SELECTION.apply(lambda row: alcohol_composite[row.alcohol_status], axis=1)
        SUBSET500['alcohol_status']=SUBSET500.apply(lambda row: alcohol_composite[row.alcohol_status], axis=1)
        SUBSET500EXTR['alcohol_status']=SUBSET500EXTR.apply(lambda row: alcohol_composite[row.alcohol_status], axis=1)

        drug_composite = {'current':'current_or_past','none':'REST','past':'current_or_past','current_or_past':'current_or_past',None:None,'future':'REST'}
        SELECTION['drug_status']=SELECTION.apply(lambda row: drug_composite[row.drug_status], axis=1)
        SUBSET500['drug_status']=SUBSET500.apply(lambda row: drug_composite[row.drug_status], axis=1)
        SUBSET500EXTR['drug_status']=SUBSET500EXTR.apply(lambda row: drug_composite[row.drug_status], axis=1)

        covars = ['AGE','GENDER','ETHNICITY','RELIGION']
        determinants = ['employment_status','tobacco_status','alcohol_status','drug_status','living_status']
        imputation_vars = list(set(covars + determinants + ['AGE','GENDER','ETHNICITY','RELIGION','MARITAL_STATUS','ADMISSION_LOCATION','INSURANCE','ADMISSION_TYPE']))
        mmec_vars = ['AGE','GENDER']
        print('imputation_vars',imputation_vars)
        print('mmec_vars',mmec_vars)

        ETHNICITY_map = {'UNKNOWN/NOT SPECIFIED':'OTHER',
                         'PATIENT DECLINED TO ANSWER':'OTHER',
                         'UNABLE TO OBTAIN':'OTHER',
                         'HISPANIC/LATINO - PUERTO RICAN':'HISPANIC_LATINO',
                         'WHITE - OTHER EUROPEAN':'WHITE',
                         'ASIAN - ASIAN INDIAN':'ASIAN',
                         'HISPANIC/LATINO - DOMINICAN':'HISPANIC_LATINO',
                         'WHITE - BRAZILIAN':'WHITE',
                         'BLACK/CAPE VERDEAN':'OTHER',
                         'ASIAN - CHINESE':'ASIAN',
                         'ASIAN - CAMBODIAN':'ASIAN',
                         'HISPANIC/LATINO - CENTRAL AMERICAN (OTHER)':'HISPANIC_LATINO',
                         'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER':'OTHER',
                         'WHITE - RUSSIAN':'WHITE',
                         'MULTI RACE ETHNICITY':'OTHER',
                         'AMERICAN INDIAN/ALASKAm NATIVE FEDERALLY RECOGNIZED TRIBE':'OTHER',
                         'AMERICAN INDIAN/ALASKA NATIVE':'OTHER',
                         'BLACK/AFRICAN':'OTHER',
                         'PORTUGUESE':'OTHER',
                         'ASIAN - VIETNAMESE':'ASIAN',
                         'ASIAN - FILIPINO':'ASIAN',
                         'HISPANIC/LATINO - SALVADORAN':'HISPANIC_LATINO',
                         'HISPANIC/LATINO - CUBAN':'HISPANIC_LATINO',
                         'BLACK/HAITIAN':'OTHER',
                         'WHITE':'WHITE',
                         'BLACK/AFRICAN AMERICAN':'BLACK_AFRICAN_AMERICAN',
                         'HISPANIC OR LATINO':'HISPANIC_LATINO'
                         }

        RELIGION_map = {
            'CATHOLIC':'CATHOLIC',
            'NOT SPECIFIED':'OTHER',
            'NOT_SPECIFIED':'OTHER',
            'PROTESTANT QUAKER':'PROTESTANT',
            'UNOBTAINABLE':'OTHER',
            'JEWISH':'JEWISH'
        }

        #SELECTION['MARITAL_STATUS'] = SELECTION['MARITAL_STATUS'].apply(lambda x: numpy.nan if x=='MARITAL_STATUS_UNKNOWN_DEFAULT' else x.strip().replace(' ','_').replace('(','').replace(')','') if isinstance(x, str) else x)
        SELECTION['RELIGION'] = SELECTION['RELIGION'].apply(lambda x: x.strip().replace(' ','_').replace('(','').replace(')','') if isinstance(x, str) else x)
        SELECTION['ETHNICITY'] = SELECTION['ETHNICITY'].apply(lambda x: ETHNICITY_map[x] if x in ETHNICITY_map else 'OTHER' if isinstance(x, str) else 'OTHER')
        SELECTION['RELIGION'] = SELECTION['RELIGION'].apply(lambda x: RELIGION_map[x] if x in RELIGION_map else 'OTHER' if isinstance(x, str) else 'OTHER')

        # Table 1
        mytable = TableOne(SELECTION[determinants + covars + ['DNR_ANY','HOSPITAL_EXPIRE_FLAG']])# , groupby='DNR_ANY', pval=True)
        print(mytable.tabulate(tablefmt="fancy_grid"))

        #if len(SELECTION) == 1174: # USED to contruct data to annotate DNR/DNI in (using brat; so a .txt and an empty .ann file)
        #    txtdir = "./n2c2_subset_txts/"
        #    SELECTION[['EVENTNOTEROWID','DNR','TXT_DNR','DNR_ANY','DISCHARGE_SUMMARY']].to_csv('DNR_info_N2C2_subset.csv')
        #    for row in SELECTION[['EVENTNOTEROWID','DNR','TXT_DNR','DNR_ANY','DISCHARGE_SUMMARY']].iterrows():
        #        row=row[1]
        #        with open(txtdir +'/'+ str(int(row['EVENTNOTEROWID'])) + '.txt', 'w') as f:
        #            f.write(row['DISCHARGE_SUMMARY'])
        #        with open(txtdir +'/'+ str(int(row['EVENTNOTEROWID'])) + '.ann', 'w') as f:
        #            f.write("")
        #exit()

        #ANNOTATED_DNR_DNI

        if 'GT' in PATHS:
            conf = confusion_matrix(y_true=SELECTION['ANNOTATED_DNR_DNI'].astype(int),y_pred=SELECTION['TXT_DNR'].astype(int))
            print(conf)
            p, r, f = precision_score(y_true=SELECTION['ANNOTATED_DNR_DNI'].astype(int),y_pred=SELECTION['TXT_DNR'].astype(int)),recall_score(y_true=SELECTION['ANNOTATED_DNR_DNI'].astype(int),y_pred=SELECTION['TXT_DNR'].astype(int)),f1_score(y_true=SELECTION['ANNOTATED_DNR_DNI'].astype(int),y_pred=SELECTION['TXT_DNR'].astype(int))
            print(p,r,f)
            print()
            for row in SELECTION.iterrows():
                rowinfo = row[1]
                if rowinfo['ANNOTATED_DNR_DNI'] == 1 and rowinfo['TXT_DNR'] == 0:
                    print('missed:', int(rowinfo['EVENTNOTEROWID']))
                if rowinfo['ANNOTATED_DNR_DNI'] == 0 and rowinfo['TXT_DNR'] == 1:
                    print('to check:', int(rowinfo['EVENTNOTEROWID']))

            conf = confusion_matrix(y_true=SELECTION['DNR_ANY'].astype(int),
                                    y_pred=SELECTION['HOSPITAL_EXPIRE_FLAG'].astype(int))
            print(conf)

        # Multivariate analysis
        #study_results_dnr_mice_adj[dirname] = {}
        #study_results_dnr_cca_adj[dirname] = {}
        #study_results_dnr_mice_crude[dirname] = {}
        #study_results_dnr_cca_crude[dirname] = {}
        #study_results_mort_mice_crude[dirname] = {}
        #study_results_mort_cca_crude[dirname] = {}
        #study_results_mort_mice_adj[dirname] = {}
        #study_results_mort_cca_adj[dirname] = {}
        #study_results_los_mice_crude[dirname] = {}
        #study_results_los_cca_crude[dirname] = {}
        #study_results_los_mice_adj[dirname] = {}
        #study_results_los_cca_adj[dirname] = {}

        for mmec in measurement_error_handling:
            for adjustment_setting in adjustment_settings:
                for missing_data_mechanism in missing_data_mechanisms:
                    for outcome in outcomes:
                        study_results[outcome][adjustment_setting][missing_data_mechanism][mmec][dirname]={}
                        for determinant in determinants:
                            study_results[outcome][adjustment_setting][missing_data_mechanism][mmec][dirname][
                                    determinant] = AssociationStudy(SELECTION, [
                                                                                   determinant] + covars if adjustment_setting == 'adjusted' else [
                                    determinant], outcome, imputation_vars, cca=missing_data_mechanism == 'cca',
                                                                    linear_outcome=outcome == 'LOS',
                                                                    det_error_calibration=determinant if (mmec == 'determinant_recalibration' and not dirname=='GT') else False,
                                                                    det_error_calibration_gt_df=SUBSET500,
                                                                    det_error_calibration_pred_df=SUBSET500EXTR,
                                                                    det_error_vars=mmec_vars).results



                                #study_results[outcome][adjustment_setting][missing_data_mechanism][mmec][dirname][determinant] = AssociationStudy(SELECTION, [determinant]+covars if adjustment_setting=='adjusted' else [determinant] ,outcome, imputation_vars, cca=missing_data_mechanism=='cca', linear_outcome=outcome=='LOS', det_error_calibration=determinant if mmec=='determinant_recalibration' else False, det_error_calibration_gt_df=SUBSET500,det_error_calibration_pred_df=SUBSET500EXTR, det_error_vars=mmec_vars).results

    #                        study_results_los_mice_crude[dirname][determinant] = AssociationStudy(SELECTION, [determinant] ,'LOS', imputation_vars, cca=False, linear_outcome=True).results
    #                        study_results_los_cca_crude[dirname][determinant] = AssociationStudy(SELECTION, [determinant] ,'LOS', [], cca=True, linear_outcome=True).results
    #                        study_results_los_mice_adj[dirname][determinant] = AssociationStudy(SELECTION, [determinant] + covars,'LOS', imputation_vars, cca=False, linear_outcome=True).results
    #                        study_results_los_cca_adj[dirname][determinant] = AssociationStudy(SELECTION, [determinant] + covars,'LOS', [], cca=True, linear_outcome=True).results

    #                        study_results_dnr_mice_crude[dirname][determinant] = AssociationStudy(SELECTION, [determinant] ,'DNR_ANY', imputation_vars, cca=False).results
    #                        study_results_dnr_cca_crude[dirname][determinant] = AssociationStudy(SELECTION, [determinant] ,'DNR_ANY', [], cca=True).results
    #                        study_results_dnr_mice_adj[dirname][determinant] = AssociationStudy(SELECTION, [determinant] + covars,'DNR_ANY', imputation_vars, cca=False).results
    #                        study_results_dnr_cca_adj[dirname][determinant] = AssociationStudy(SELECTION, [determinant] + covars,'DNR_ANY', [], cca=True).results

    #                        study_results_mort_mice_crude[dirname][determinant] = AssociationStudy(SELECTION, [determinant] ,'HOSPITAL_EXPIRE_FLAG', imputation_vars, cca=False).results
    #                        study_results_mort_cca_crude[dirname][determinant] = AssociationStudy(SELECTION, [determinant] ,'HOSPITAL_EXPIRE_FLAG', [], cca=True).results
    #                        study_results_mort_mice_adj[dirname][determinant] = AssociationStudy(SELECTION, [determinant] + covars,'HOSPITAL_EXPIRE_FLAG', imputation_vars, cca=False).results
    #                        study_results_mort_cca_adj[dirname][determinant] = AssociationStudy(SELECTION, [determinant] + covars,'HOSPITAL_EXPIRE_FLAG', [], cca=True).results

        if 'GT' in PATHS:
            SELECTIONS[dirname] = SELECTION[determinants + covars + outcomes]


    measures = {'Precision': {}, 'Recall': {}, 'F1-score': {}}

    if 'GT' in PATHS:

        path_to_num_ann_files = {model: len(list(glob(PATHS[model] + '/*.ann'))) for model in PATHS}
        m = min(path_to_num_ann_files,
                key=path_to_num_ann_files.get)  # evaluate all models w.r.t. folder with smallest number of .ann files (to make sure the scores are comparable; 1174 ann files)

        print(m, path_to_num_ann_files[m])
        tmpgtdir = './tml_gt/'
        tmppreddir = './tml_pred/'
        if os.path.exists(tmpgtdir):
            shutil.rmtree(tmpgtdir)
        os.makedirs(tmpgtdir)
        ann_file_names = [fpth.split('/')[-1] for fpth in glob(PATHS[m] + '/*.ann')]
        print(len(ann_file_names))
        for fname in ann_file_names:
            if os.path.exists(PATHS['GT'] + '/' + fname):
                shutil.copy(PATHS['GT'] + '/' + fname, tmpgtdir)
                shutil.copy(PATHS['GT'] + '/' + fname.replace('.ann', '.txt'), tmpgtdir)

        for model in PATHS:
            if not model == 'GT':
                if os.path.exists(tmppreddir):
                    shutil.rmtree(tmppreddir)
                os.makedirs(tmppreddir)
                for fname in ann_file_names:
                    if os.path.exists(PATHS[model] + '/' + fname):
                        shutil.copy(PATHS[model] + '/' + fname, tmppreddir)
                        shutil.copy(PATHS[model] + '/' + fname.replace('.ann', '.txt'), tmppreddir)

                evaluation = score_brat_sdoh( \
                    gold_dir=tmpgtdir,
                    predict_dir=tmppreddir,
                    labeled_args=SDOH_LABELED_ARGUMENTS,
                    score_trig=OVERLAP,
                    score_span=EXACT,
                    score_labeled=LABEL,
                    output_path=None,
                    include_detailed=False,
                    loglevel='info')

                print(model)
                print(evaluation)
                measures['Precision'][model] = evaluation['P'].iloc[0]
                measures['Recall'][model] = evaluation['R'].iloc[0]
                measures['F1-score'][model] = evaluation['F1'].iloc[0]
                if os.path.exists(tmppreddir):
                    shutil.rmtree(tmppreddir)
        if os.path.exists(tmpgtdir):
            shutil.rmtree(tmpgtdir)


    with open('study_results_'+out_dir+'.p', 'wb') as f:
        pickle.dump((measures,PATHS,SELECTIONS, study_results),f)

#    with open('study_results_'+out_dir+'_adj.p', 'wb') as f:
#        pickle.dump((measures,PATHS,SELECTIONS, study_results_los_mice_adj, study_results_los_cca_adj, study_results_dnr_mice_adj, study_results_dnr_cca_adj, study_results_mort_mice_adj, study_results_mort_cca_adj),f)

#    with open('study_results_'+out_dir+'_crude.p', 'wb') as f:
#        pickle.dump((measures,PATHS,SELECTIONS, study_results_los_mice_crude, study_results_los_cca_crude, study_results_dnr_mice_crude, study_results_dnr_cca_crude, study_results_mort_mice_crude, study_results_mort_cca_crude),f)


    sys.stdout = old_stdout
    log_file.close()

