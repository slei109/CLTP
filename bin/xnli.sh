
### First, train the pretrained model on the En dataset for xnli task
python ../run_xnli.py xlmr_large_zero_shot 0

### Second, self-learning on the unlabeled dataset of 14 languages (baseline of non-parital-label-learning, can speed convergence for third step)
# python ../run_xnli_sl.py xlmr_large_zero_shot_sl 0

### Third, partial-label-learning on the unlabeled dataset of 14 languages
# python ../run_xnli_sl.py xlmr_large_zero_shot_parial_sl 0

