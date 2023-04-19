
This code is associated with our submissions to the [N2C2 Shared Task - Track 2](https://n2c2.dbmi.hms.harvard.edu/2022-track-2), on extraction of social determinants of health from clinical notes. It was also used in a consequent study on the effects of different NLP models on downstream medical association study results.

# What is here:

This code contains the used submission script, and the two main python files to train or apply our BIO-scheme base SDOH models:

- `sdoh_model_bert_bio.py`: The code used for all BERT settings (call `sdoh_model_bert_bio.py -h` for more detailed information).
- `sdoh_model_bio.py`: The code used for all other settings (call `sdoh_model_bio.py` for more detailed information).
- `Submission_script.sh`: The script that was used to make our submissions for the shared task.
- `pretrain_embs.py`: The script used to pretrain the fastText embeddings (on the MIMIC III and the UCSF data).
- `association_study_experiments.sh`: The script used to conduct the experiments from the arXiv article.

# What is not here:

The text data (clinical notes from MIMIC III and the University of Washington) and annotations were provided by the task organizers under a data sharing agreement, for patient privacy reasons.
For this reason we cannot share this data here.

# References

Results from our submissions are reported in the attached abstract:

> Madhumita Sushil, Atul J. Butte, Ewoud Schuit, Artuur M. Leeuwenberg. [Cross-institution extraction of social determinants of health from 
clinical notes: an evaluation of methods](https://github.com/tuur/sdoh_n2c2track2_ucsf_umcu/blob/main/N2C2%20Abstract.pdf). AMIA Natural Language Processing Working Group Pre-Symposium. November, 2022.

Results from consequent experiments about the impact of NLP modeling choices on downstream association study results:

> Sushil, Madhumita, et al. [Cross-institution text mining to uncover clinical associations: a case study relating social factors and code status in intensive care medicine.](https://arxiv.org/abs/2301.06570) arXiv preprint arXiv:2301.06570 (2023).

