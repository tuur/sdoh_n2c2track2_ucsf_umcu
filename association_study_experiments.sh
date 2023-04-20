

prepared_df="admissions_df.p"

# Analysis on N2C2 data (with GT reference)
#python association_study.py -load_df $prepared_df -pred_ann_dirs "/Users/aleeuw15/Desktop/Research/N2C2 - SDOH/npj Dig Med/results/NLP/N2C2/" -n2c2_alignment 1


#python plotting.py 'study_results_N2C2.p'

#python plotting_jama.py 'study_results_N2C2.p'

#python plotting.py 'study_results_N2C2_adj.p' #
#python plotting.py 'study_results_N2C2_crude.p' #





python association_study.py -load_df $prepared_df -pred_ann_dirs "/Users/aleeuw15/Desktop/Research/N2C2 - SDOH/npj Dig Med/results/NLP/MIMIC/" -n2c2_alignment 0




# Analysis on MIMIC data (no reference)
#pred_ann_dir="/Users/aleeuw15/Desktop/Research/N2C2 - SDOH/JAMIA paper/code/data/association_study_all_mimic_discharge/"
#python association_study.py -load_df $prepared_df -pred_ann_dir $pred_ann_dir -n2c2_alignment 0


