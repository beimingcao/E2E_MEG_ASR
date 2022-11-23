import time
import yaml
import os
import glob
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from shutil import copyfile
import random

from utils.database import MEG_PHONE_ASR

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import torchaudio

# Fix random seeds

seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
    
def data_loadin(args):

    config = yaml.load(open(args.conf_dir, 'r'), Loader=yaml.FullLoader)
    data_path = config['Corpus']['path']
    label_file = config['Corpus']['label_file']
    data_session = config['Data_setup']['session']
    CV_num = config['Data_setup']['CV_num']
    train_ratio = config['Data_setup']['train_ratio']
    CV_list = list(range(CV_num))
    label_file_path = os.path.join(data_path, label_file)
    f = open(label_file_path, 'r')
    phone_label_list = f.readlines()
    f.close()
   # phone_label_list.sort()
    
    processed_data_path = os.path.join(args.buff_dir, data_session, 'data')
    MEG_path_list = glob.glob(processed_data_path + '/*.pt')
    MEG_path_list.sort()   
    
    CV_data_path = os.path.join(args.buff_dir, data_session, 'data_CV')
    
    for i in CV_list:
        CV = 'CV' + format(i, '01d')
        CV_data_out_path = os.path.join(CV_data_path, CV)
        if not os.path.exists(CV_data_out_path):
            os.makedirs(CV_data_out_path)            
        _CV_list = CV_list.copy()
        _CV_list.remove(i)
        
        test_idx = np.arange(len(MEG_path_list)//CV_num) + i*(len(MEG_path_list)//CV_num)        
        train_val_idx = np.delete(np.arange(len(MEG_path_list)), test_idx)
        random.seed(123)
        random.shuffle(train_val_idx)
        train_idx = train_val_idx[:int(len(train_val_idx)*train_ratio)]
        val_idx = train_val_idx[int(len(train_val_idx)*train_ratio):]
        
        train_dataset = MEG_PHONE_ASR(MEG_path_list, phone_label_list, train_idx)
        
            
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_dir', default = 'conf/conf.yaml')
    parser.add_argument('--buff_dir', default = 'current_exp')
    args = parser.parse_args()
    data_loadin(args)



        
