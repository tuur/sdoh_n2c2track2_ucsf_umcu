
charteventspath= "/Users/aleeuw15/Desktop/Research/NLP - TextImp/datasets/mimic/CHARTEVENTS.csv"
#charteventspath= "/Users/aleeuw15/Desktop/Research/NLP - TextImp/datasets/demo/mimic/CHARTEVENTS.csv"

# based on https://github.com/MIT-LCP/mimic-code/issues/180
dnr_terms=set(['Full code','Comfort Measures','CPR Not Indicated','Do Not Intubate','Do Not Resuscitate']) #,'Other/Remarks'])
dnr_csv = 'HADM_ID_DNR_status.csv'
with open(dnr_csv, 'w') as f_out:c
    with open(charteventspath, 'r') as f_in:
        for l in f_in:
            spl = l.replace('"','').split(',')
            if spl[-7] in dnr_terms:
                print(spl)
                f_out.write(','.join([spl[2],spl[-7]]) + '\n')


