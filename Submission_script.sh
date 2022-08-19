export PATH=$PATH:~/anaconda3/bin
source ~/anaconda3/etc/profile.d/conda.sh
conda activate n2c2-tr2-socdet
export PYTHONPATH=~/N2C2-TR2-SOCDET:$PYTHONPATH
# -------------------------------------
# BEFORE TASK A & B START -------------
# -------------------------------------

mimic_train_and_dev_path="/wynton/protected/project/outcome_pred/n2c2-tr2-socdet/data/train+dev/" # provide training data DIRECTORY (containing all .txt and .ann files of both the mimic train and dev sets)
mimic_selftraining_path='/wynton/protected/project/outcome_pred/n2c2-tr2-socdet/unannotated/mimic/selftrain/' # data DIRECTORY for self training in MIMIC data (containing only .txt files; one doc per .txt, NOT all docs in one .txt)
ucsf_selftraining_path="/wynton/protected/project/outcome_pred/n2c2-tr2-socdet/unannotated/ucsf/selftrain/" # data DIRECTORY for self training in UCSF data (containing only .txt files; one doc per .txt, NOT all docs in one .txt)
mimic_embs="/wynton/protected/project/outcome_pred/clinical_emb/mimic_tokenized_unk1_fasttext_5it_250.vec" # MIMIC embeddings FILE (.vec or .txt files)
ucsf_embs="/wynton/protected/project/outcome_pred/clinical_emb/all_ucsf_notes_2022-03-27_tokenized_unk1_fasttext_1it_250.vec" # UCSF embeddings FILE (.vec or .txt files)
ucsf_bert='/wynton/protected/project/outcome_pred/ucsf_bert_pytorch/512/500k-275k/' # path for UCSF-BERT model
trained_models_path="/wynton/protected/project/outcome_pred/n2c2-tr2-socdet/trained_models/" # DIRECTORY where the trained models should be saved to
OUTPUT_path="/wynton/protected/project/outcome_pred/n2c2-tr2-socdet/outputs/" # top output DIRECTORY (subdirectories per run are automatically generated below)

conda activate n2c2-tr2-socdet # load the right environment

trained_model_A1=$trained_models_path"/A1.model"
trained_model_A2=$trained_models_path"/A2.model"
trained_model_A3=$trained_models_path"/A3.model"

# TODO: comment all python statements but the ones in this section

## Train model A1
echo "Training model A1"
python sdoh_model_bio.py -labeled_train_corpus $mimic_train_and_dev_path -model_saving_path $trained_model_A1
#
## Train model A2 (With pre-trained embeddings)
echo "Training model A2"
python sdoh_model_bio.py -labeled_train_corpus $mimic_train_and_dev_path -model_saving_path $trained_model_A2 -pretrained_wembs1 $mimic_embs -pretrained_wembs2 $ucsf_embs
#
## Train model A3 (With self training on MIMIC)
echo "Training model A3"
python sdoh_model_bio.py -labeled_train_corpus $mimic_train_and_dev_path -model_saving_path $trained_model_A3 -pretrained_wembs1 $mimic_embs -pretrained_wembs2 $ucsf_embs -selftraining $mimic_selftraining_path

# No need to train B1 (Same as A2)
trained_model_B2=$trained_models_path"/B2.model"
trained_model_B3=$trained_models_path"/B3.model"

# Train model B2 (With UW trigger proportion weighted loss) TODO: replace  B2 for the call to train a UCSF-BERT model
echo "Training model B2"
python sdoh_model_bert_bio.py -labeled_train_corpus $mimic_train_and_dev_path -model_saving_path $trained_model_B2 -unk1_prob 0 -max_num_epochs 250 -rnn_dim 100 -dropout 0.2 -lowercase_tokens 0 -bert_model_name_or_path $ucsf_bert -freeze_emb False -conflate_digits 0 -subtask_weight 1
#
## Train model B3 (With self training on UCSF)
echo "Training model B3"
python sdoh_model_bio.py -labeled_train_corpus $mimic_train_and_dev_path -model_saving_path $trained_model_B3 -pretrained_wembs1 $mimic_embs -pretrained_wembs2 $ucsf_embs -selftraining $ucsf_selftraining_path

# -------------------------------------
# ---- TASK A -------------------------
# -------------------------------------
# TODO: comment all python statements but the ones in this section

#INPUT_mimic_test_path_A="/wynton/protected/project/outcome_pred/n2c2-tr2-socdet/data/test/Track2_SubtaskA_test_txt_only/Text/test/mimic/" # path to unlabeled input DIRECTORY (with .txt files)
#
#OUTPUT_A1=$OUTPUT_path"/A1-preds/Text/"
#OUTPUT_A2=$OUTPUT_path"/A2-preds/Text/"
#OUTPUT_A3=$OUTPUT_path"/A3-preds/Text/"
#
## Predict A-1
python sdoh_model_bio.py -model_loading_path $trained_model_A1 -unlabeled_test_corpus $INPUT_mimic_test_path_A -output_test_predictions $OUTPUT_A1
## Predict A-2
python sdoh_model_bio.py -model_loading_path $trained_model_A2 -unlabeled_test_corpus $INPUT_mimic_test_path_A -output_test_predictions $OUTPUT_A2
## Predict A-3
python sdoh_model_bio.py -model_loading_path $trained_model_A3 -unlabeled_test_corpus $INPUT_mimic_test_path_A -output_test_predictions $OUTPUT_A3

## -------------------------------------
## ---- TASK B -------------------------
## -------------------------------------
## TODO: comment all python statements but the ones in this section
#
INPUT_uw_test_path_B_train="/wynton/protected/project/outcome_pred/n2c2-tr2-socdet/data/test/Track2_SubtaskB_test_txt_only/Text/train/uw/" # unlabeled input DIRECTORY (with .txt files)
INPUT_uw_test_path_B_dev="/wynton/protected/project/outcome_pred/n2c2-tr2-socdet/data/test/Track2_SubtaskB_test_txt_only/Text/dev/uw/" # unlabeled input DIRECTORY (with .txt files)

OUTPUT_B1_train=$OUTPUT_path"B1-preds/Text/train/uw/"
OUTPUT_B2_train=$OUTPUT_path"/B2-preds/Text/train/uw/"
OUTPUT_B3_train=$OUTPUT_path"/B3-preds/Text/train/uw/"

OUTPUT_B1_dev=$OUTPUT_path"/B1-preds/Text/dev/uw/"
OUTPUT_B2_dev=$OUTPUT_path"/B2-preds/Text/dev/uw/"
OUTPUT_B3_dev=$OUTPUT_path"/B3-preds/Text/dev/uw/"
#
# Predict train
# Predict B-1 (with model A2)
python sdoh_model_bio.py -model_loading_path $trained_model_A2 -unlabeled_test_corpus $INPUT_uw_test_path_B_train -output_test_predictions $OUTPUT_B1_train
# Predict B-2
python sdoh_model_bert_bio.py -labeled_train_corpus $mimic_train_and_dev_path -model_saving_path $trained_model_B2 -unk1_prob 0 -max_num_epochs 250 -rnn_dim 100 -dropout 0.2 -lowercase_tokens 0 -bert_model_name_or_path $ucsf_bert -freeze_emb False -conflate_digits 0 -subtask_weight 1 -unlabeled_test_corpus $INPUT_uw_test_path_B_train -output_test_predictions $OUTPUT_B2_train
# Predict B-3
python sdoh_model_bio.py -model_loading_path $trained_model_B3 -unlabeled_test_corpus $INPUT_uw_test_path_B_train -output_test_predictions $OUTPUT_B3_train

# Predict dev
# Predict B-1 (with model A2)
python sdoh_model_bio.py -model_loading_path $trained_model_A2 -unlabeled_test_corpus $INPUT_uw_test_path_B_dev -output_test_predictions $OUTPUT_B1_dev
# Predict B-2
python sdoh_model_bert_bio.py -labeled_train_corpus $mimic_train_and_dev_path -model_saving_path $trained_model_B2 -unk1_prob 0 -max_num_epochs 250 -rnn_dim 100 -dropout 0.2 -lowercase_tokens 0 -bert_model_name_or_path $ucsf_bert -freeze_emb False -conflate_digits 0 -subtask_weight 1 -unlabeled_test_corpus $INPUT_uw_test_path_B_dev -output_test_predictions $OUTPUT_B2_dev
# Predict B-3
python sdoh_model_bio.py -model_loading_path $trained_model_B3 -unlabeled_test_corpus $INPUT_uw_test_path_B_dev -output_test_predictions $OUTPUT_B3_dev


## -------------------------------------
## ---- BEFORE TASK C STARTS -----------
## -------------------------------------
## TODO: comment all python statements but the ones in this section
mimic_train_and_dev_path="/wynton/protected/project/outcome_pred/n2c2-tr2-socdet/data/task_c/Track2_SubtaskC_train_V2/Annotations/train+dev/mimic/" # MIMIC data DIRECTORY (containing all .txt and .ann files of both the mimic train and dev sets)
INPUT_uw_train_path_C="/wynton/protected/project/outcome_pred/n2c2-tr2-socdet/data/task_c/Track2_SubtaskC_train_V2/Annotations/train+dev/uw/" # path to LABELED input DIRECTORY (with .txt and .ann files)

trained_model_C1=$trained_models_path"/C1.model"
trained_model_C2=$trained_models_path"/C2.model"
trained_model_C3=$trained_models_path"/C3.model"

## Train model C1 (MIMIC train + UW train)
python sdoh_model_bio.py -labeled_train_corpus $mimic_train_and_dev_path -second_labeled_train_corpus $INPUT_uw_train_path_C -pretrained_wembs1 $mimic_embs -pretrained_wembs2 $ucsf_embs -model_saving_path $trained_model_C1
## Train model C2 (MIMIC train + UW train) + triggers recalibrated (last layers updated) in UW train
python sdoh_model_bio.py -labeled_train_corpus $mimic_train_and_dev_path -second_labeled_train_corpus $INPUT_uw_train_path_C -pretrained_wembs1 $mimic_embs -pretrained_wembs2 $ucsf_embs -model_saving_path $trained_model_C2 -calibrate $INPUT_uw_train_path_C
## Train model C3 MIMIC train + UW train with 10* weight for loss in UW train
python sdoh_model_bio.py -labeled_train_corpus $mimic_train_and_dev_path -second_labeled_train_corpus $INPUT_uw_train_path_C -pretrained_wembs1 $mimic_embs -pretrained_wembs2 $ucsf_embs -model_saving_path $trained_model_C3 -loss_weight_train2 10

#
## -------------------------------------
## ---- TASK C -------------------------
## -------------------------------------
## TODO: comment all python statements but the ones in this section
#

INPUT_uw_test_path_C="/wynton/protected/project/outcome_pred/n2c2-tr2-socdet/data/test/Track2_SubtaskC_test_txt_only/Text/test/uw/" # path to unlabeled input DIRECTORY (with .txt files)

OUTPUT_C1=$OUTPUT_path"/C1-preds/Text/uw/"
OUTPUT_C2=$OUTPUT_path"/C2-preds/Text/uw/"
OUTPUT_C3=$OUTPUT_path"/C3-preds/Text/uw/"

# Predict C-1
python sdoh_model_bio.py -model_loading_path $trained_model_C1 -unlabeled_test_corpus $INPUT_uw_test_path_C -output_test_predictions $OUTPUT_C1
# Predict C-2
python sdoh_model_bio.py -model_loading_path $trained_model_C2 -unlabeled_test_corpus $INPUT_uw_test_path_C -output_test_predictions $OUTPUT_C2
# Predict C-3
python sdoh_model_bio.py -model_loading_path $trained_model_C3 -unlabeled_test_corpus $INPUT_uw_test_path_C -output_test_predictions $OUTPUT_C3
