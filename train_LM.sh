#!/bin/bash

current_exp=current_exp
session=Articulation
data_path=$current_exp/$session/data_CV
LM_order=2

for CV in $data_path/*; do

    text_path=$CV/Phone_LM_train_list.txt
    kenlm/build/bin/lmplz -o $LM_order <$text_path --discount_fallback >$CV/LM.arpa

done
