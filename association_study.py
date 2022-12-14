import argparse, re, pandas, collections, math, numpy
import statsmodels.formula.api as smf
from dateutil.parser import parse as parsedatetime
from brat_scoring.corpus import Corpus
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.imputation.mice import MICEData, MICE
import statsmodels.api as sm
from copy import copy
from tableone import TableOne
from glob import glob
# statsmodels.imputation.mice.MICEData
import dill as pickle

class AssociationStudy():

    def __init__(self, df, xvars, yvar, imp_vars, cca=False):
        all_relevant_vars = list(set(xvars + [yvar] + imp_vars))
        self.df = df[all_relevant_vars]
        self.xvars=xvars
        self.yvar=yvar
        self.results=None
        self.imp_vars=imp_vars


        if cca:
            self.df = self.df.dropna()

        self.X_df = pandas.get_dummies(self.df[self.xvars], drop_first=True)
        self.imp_df = pandas.get_dummies(self.df[self.imp_vars], drop_first=True)
        imp_vars_names = self.imp_df.columns
        self.y_df = self.df[self.yvar]

        #self.Xy_df = pandas.concat([self.X_df, self.y_df])
        self.Xy_df = copy(self.X_df)
        self.Xy_df[self.yvar]=self.y_df
        for v in self.imp_df.columns:
            if not v in self.Xy_df.columns:
                #print(v)
                self.Xy_df[v] = copy(self.imp_df[v])

        if cca:
            self.X_df['Intercept']=1
            self.formula = self.yvar + ' ~ ' + ' + '.join(['"'+v+'"' for v in self.xvars])
            print('\n\n',60*'=','> outcome events:',self.df[self.yvar].sum(),'\n',self.formula)
            model = GLM(self.y_df,self.X_df, family=families.Binomial()).fit(attach_wls=True, atol=1e-10)
            print(model.summary())
            self.results=model
        else:
            imp = MICEData(self.Xy_df, k_pmm=10)
            for xvar in self.Xy_df.columns:
                imp.set_imputer(xvar, xvar + ' ~ ' + ' + '.join(imp_vars_names))
            fml = self.yvar + ' ~ ' + ' + '.join(self.X_df.columns)
            print(fml)
            mice = MICE(fml, sm.GLM, imp, init_kwds={"family": sm.families.Binomial()})
            results = mice.fit(10, 10)
            print(results.summary())
            self.results=results


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

def extract_code_status_from_discharge_summary(admissions): # TODO: bugfix, lowertxt seems to be float sometimes...

    txt_dnr = {}
    for r in admissions.iterrows():
        txt = r[1]['DISCHARGE_SUMMARY']
        hadmid = int(r[1]['HADM_ID'])
        txt_dnr[hadmid] = False
        if isinstance(txt, str):
            codestatuslines = re.findall(r'.+dnr.+\n',txt.lower())
            codestatuslines += re.findall(r'.+do not resuscitate.+\n',txt.lower())
            codestatuslines += re.findall(r'.+do not intubate.+\n',txt.lower())
            codestatuslines += re.findall(r'.+code status.+\n',txt.lower())
            for codestatusline in codestatuslines:
                if (('dnr' in codestatusline or 'do not resuscitate' in codestatusline or 'do not intubate' in codestatusline) and not 'full code' in codestatusline):
                    txt_dnr[hadmid]= True

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
# TODO: # TODO: option to recognize and read zip ann folders!!!!

    print(ann_file_dir)
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
    admissions=pandas.merge(admissions, sdoh_df, on=['EVENTNOTEROWID'], how='left')

    #print('difference',set(sdoh_df['EVENTNOTEROWID']).difference(admissions['EVENTNOTEROWID']))
    # TODO; CHECK IF THOSE 93 IN THE DIFFERENCE ARE outside of the age range.
    #pd.merge(a, b, on=['A', 'B'])

    return sdoh_df, admissions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('-mimic_file_alignment', required=False, help='csv aligning mimic note event ids and note ids', default="/Users/aleeuw15/Desktop/Research/N2C2 - SDOH/JAMIA paper/code/data/MIMIC_file_alignment.csv")
    parser.add_argument('-mimic_patients', required=False, help='...', default="/Users/aleeuw15/Desktop/Research/NLP - TextImp/datasets/mimic/PATIENTS.csv")
    parser.add_argument('-mimic_admissions', required=False, help='...', default="/Users/aleeuw15/Desktop/Research/NLP - TextImp/datasets/mimic/ADMISSIONS.csv")
    parser.add_argument('-mimic_noteevents', required=False, help='...', default="/Users/aleeuw15/Desktop/Research/NLP - TextImp/datasets/mimic/NOTEEVENTS.csv")
    parser.add_argument('-dnr_codes', required=False, help='...', default='HADM_ID_DNR_status.csv')
    parser.add_argument('-excl_dir', required=False, help='Directory where the .ann files are stored that are used for text mining fine tuning and should be excluded', default="/Users/aleeuw15/Desktop/Research/N2C2 - SDOH/JAMIA paper/code/data/association_study data splits/mimic_train_500/")
    parser.add_argument('-mimic_discharge_sochist', required=False, help='...', default="/Users/aleeuw15/Desktop/Research/N2C2 - SDOH/JAMIA paper/code/data/mimic_discharge_sochist.csv")
    parser.add_argument('-n2c2_alignment', required=False, help='Use n2c2 txt ids (if ann dir uses n2c2 ids instead of mimic row ids).', type=int)
    parser.add_argument('-load_df', required=False, help='Directory where the .ann files are stored.')
    #parser.add_argument('-ann_dir', required=False, help='Directory where the .ann files are stored.', default="/Users/aleeuw15/Desktop/Research/N2C2 - SDOH/JAMIA paper/code/data/association_study SHAC/S4/")
    parser.add_argument('-pred_ann_dirs', required=False, help='Directory where the .ann files are stored.')

    args = parser.parse_args()

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
    #else:
    #    with open(args.load_df, 'rb') as f:
    #        ADMISSIONS = pickle.load(f)

    # get SDOH annotations info

    study_results = {}
    for pred_path in glob(args.pred_ann_dirs+"/*", recursive = False):
        #print(pred_path)
        dirname = pred_path.split('/')[-1]
        print('>>>>', dirname)
        with open(args.load_df, 'rb') as f:
            ADMISSIONS = pickle.load(f)
        sdoh_df, ADMISSIONS = read_sdoh_annotations(ADMISSIONS, pred_path, args.mimic_file_alignment, args.n2c2_alignment)
        ADMISSIONS = exclude_n2c2_rows(ADMISSIONS, args.excl_dir, args.mimic_file_alignment)

        print('ADMISSIONS initial:',len(ADMISSIONS))
        ADMISSIONS = ADMISSIONS[~ADMISSIONS['EVENTNOTEROWID'].isnull()]
        print('ADMISSIONS with linkable notes:',len(ADMISSIONS))
        # select only 18-69
        ADMISSIONS = ADMISSIONS[ADMISSIONS['AGE'] >= 18]
        ADMISSIONS = ADMISSIONS[ADMISSIONS['AGE'] <= 89]
        print('ADMISSIONS after age 18-89:',len(ADMISSIONS))
        # excluded documents used to train / fine tune the text mining models
        ADMISSIONS = ADMISSIONS[ADMISSIONS['TO_EXCLUDE']==False]
        print('ADMISSIONS minus fine-tuning data:',len(ADMISSIONS))

        ADMISSIONS['DNR_ANY'] = ADMISSIONS.apply(lambda row: int(row.DNR or row.TXT_DNR), axis=1)
        print('ADMISSIONS DNR prevalence:', round(sum(ADMISSIONS['DNR_ANY'])/len(ADMISSIONS)*100,2),'%')

        SELECTION = ADMISSIONS[ADMISSIONS['sdoh_extracted']==True]
        print('SELECTION:',len(SELECTION))
        print('SELECTION DNR prevalence:', round(sum(SELECTION['DNR_ANY'])/len(SELECTION)*100,2),'%')
        print('SELECTION DNR N:', round(sum(SELECTION['DNR_ANY']),2))

        print('employment_status',collections.Counter(SELECTION['employment_status']))
        print('tobacco_status',collections.Counter(SELECTION['tobacco_status']))
        print('drug_status',collections.Counter(SELECTION['drug_status']))
        print('alcohol_status',collections.Counter(SELECTION['alcohol_status']))
        print('living_status',collections.Counter(SELECTION['living_status']))

        # composite predictors
        employment_composite = {'employed':'employed','unemployed':'REST','on_disability':'REST','retired':'REST',None:None,'student':'REST','homemaker':'REST'}
        SELECTION['employment_status']=SELECTION.apply(lambda row: employment_composite[row.employment_status], axis=1)

        living_composite = {'homeless':'REST','with_family':'with_family','with_others':'REST','alone':'REST',None:None}
        SELECTION['living_status']=SELECTION.apply(lambda row: living_composite[row.living_status], axis=1)

        tobacco_composite = {'current':'current_or_past','none':'REST','past':'current_or_past','current_or_past':'current_or_past',None:None,'future':'REST'}
        SELECTION['tobacco_status']=SELECTION.apply(lambda row: tobacco_composite[row.tobacco_status], axis=1)

        alcohol_composite = {'current':'current_or_past','none':'REST','past':'current_or_past','current_or_past':'current_or_past',None:None, 'future':'REST'}
        SELECTION['alcohol_status']=SELECTION.apply(lambda row: alcohol_composite[row.alcohol_status], axis=1)

        drug_composite = {'current':'current_or_past','none':'REST','past':'current_or_past','current_or_past':'current_or_past',None:None,'future':'REST'}
        SELECTION['drug_status']=SELECTION.apply(lambda row: drug_composite[row.drug_status], axis=1)

        covars = ['AGE','GENDER','ETHNICITY','RELIGION']
        determinants = ['employment_status','tobacco_status','alcohol_status','drug_status','living_status']
        imputation_vars = list(set(covars + determinants + ['AGE','GENDER','ETHNICITY','RELIGION','MARITAL_STATUS','ADMISSION_LOCATION','INSURANCE','ADMISSION_TYPE','DIAGNOSIS']))
        #print(collections.Counter(SELECTION['ETHNICITY']))
        print('imputation_vars',imputation_vars)

        ETHNICITY_map = {'UNKNOWN/NOT SPECIFIED':'NOT_SPECIFIED',
                         'PATIENT DECLINED TO ANSWER':'NOT_SPECIFIED',
                         'UNABLE TO OBTAIN':'NOT_SPECIFIED',
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
                         'AMERICAN INDIAN/ALASKA NATIVE FEDERALLY RECOGNIZED TRIBE':'OTHER',
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

        #print(collections.Counter(SELECTION['RELIGION']))

        RELIGION_map = {
            'CATHOLIC':'CATHOLIC',
            'NOT SPECIFIED':'NOT_SPECIFIED',
            'NOT_SPECIFIED':'NOT_SPECIFIED',
            'PROTESTANT QUAKER':'PROTESTANT',
            'UNOBTAINABLE':'NOT_SPECIFIED',
            'JEWISH':'JEWISH'
        }

        #SELECTION['MARITAL_STATUS'] = SELECTION['MARITAL_STATUS'].apply(lambda x: numpy.nan if x=='MARITAL_STATUS_UNKNOWN_DEFAULT' else x.strip().replace(' ','_').replace('(','').replace(')','') if isinstance(x, str) else x)
        SELECTION['RELIGION'] = SELECTION['RELIGION'].apply(lambda x: x.strip().replace(' ','_').replace('(','').replace(')','') if isinstance(x, str) else x)
        SELECTION['ETHNICITY'] = SELECTION['ETHNICITY'].apply(lambda x: ETHNICITY_map[x] if x in ETHNICITY_map else 'OTHER' if isinstance(x, str) else 'NOT_SPECIFIED')
        SELECTION['RELIGION'] = SELECTION['RELIGION'].apply(lambda x: RELIGION_map[x] if x in RELIGION_map else 'OTHER' if isinstance(x, str) else 'NOT_SPECIFIED')

        # Table 1
        mytable = TableOne(SELECTION[determinants + covars + ['DNR_ANY']], groupby='DNR_ANY', pval=True)
        print(mytable.tabulate(tablefmt="fancy_grid"))

        # Multivariate analysis
        study_results[dirname] = {}
        for determinant in determinants:
            study = AssociationStudy(SELECTION, [determinant] + covars,'DNR_ANY', imputation_vars, cca=False)
            study_results[dirname][determinant] = study.results


    out_dir = [x for x in args.pred_ann_dirs.split('/') if not x==''][-1]
    with open('study_results'+out_dir+'.p', 'wb') as f:
        pickle.dump((SELECTION, study_results),f)



