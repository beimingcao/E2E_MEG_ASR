#!/bin/bash

### Fetch the path of training text ###

text_path=$(python3 2_data_loadin.py 2>&1 > /dev/null)

### Training language model ###

#text_path=current_exp/Articulation/data_CV/CV7/Phone_LM_train_list.txt
out_dir="$(dirname "${text_path}")"
 
echo $out_dir

kenlm/build/bin/lmplz -o 2 <$text_path --discount_fallback >$out_dir/LM.arpa
