import time
import yaml
import os
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from utils.models import SpeechRecognitionModel
from utils.models import save_model
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from utils.utils import EarlyStopping, IterMeter, data_processing_DeepSpeech, GreedyDecoder
import torch.nn.functional as F
from jiwer import wer
import random
import numpy as np
import torchaudio

def test_MEG_ASR(CV, test_dataset, exp_output_folder, args):
    ### Dimension setup ###
    config = yaml.load(open(args.conf_dir, 'r'), Loader=yaml.FullLoader)
    
    ### Model setup ###
    n_cnn_layers = config['NN_setup']['n_cnn_layers']
    n_rnn_layers = config['NN_setup']['n_rnn_layers']    
    rnn_dim = config['NN_setup']['rnn_dim']
    stride = config['NN_setup']['stride']
    dropout = config['NN_setup']['dropout']
            
    ### Test ###
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                            batch_size=1,
                            shuffle=False,
                            collate_fn=lambda x: data_processing_DeepSpeech(x, transforms = None))
                            
    model_out_folder = os.path.join(exp_output_folder, 'trained_models')
        
    SPK_model_path = os.path.join(model_out_folder)
    model_path = os.path.join(SPK_model_path, 'CV' + str(CV) + '_DS.pt')
    model = SpeechRecognitionModel(n_cnn_layers, n_rnn_layers, rnn_dim, 41, 204, stride, dropout)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    pred = []
    label = []

    for batch_idx, _data in enumerate(test_loader):
        meg, labels, input_lengths, label_lengths = _data 
        x = torch.mean(meg, 2).transpose(2, 3)  
        output = model(x)  # (batch, time, n_class)

        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1) # (time, batch, n_class)

        decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths)

        pred.append(' '.join(decoded_preds[0]))
        label.append(' '.join(decoded_targets[0]))
    print(pred)
    print('#########################')
    print(label)
    error = wer(pred, label)
    return error




if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_dir', default = 'conf/conf.yaml')
    parser.add_argument('--buff_dir', default = 'current_exp')
    args = parser.parse_args()
    config = yaml.load(open(args.conf_dir, 'r'), Loader=yaml.FullLoader)
    data_session = config['Data_setup']['session']
    data_path = os.path.join(args.buff_dir, data_session, 'data_CV')
    CV_to_run = config['Data_setup']['CV_to_run']

    results_all = os.path.join(args.buff_dir, 'results_all.txt')
    with open(results_all, 'w') as r:
        for CV in CV_to_run:
            data_path_CV = os.path.join(data_path, 'CV' + str(CV))
           
            te = open(os.path.join(data_path_CV, 'test_data.pkl'), 'rb')
            test_dataset = pickle.load(te)
        
            WER = test_MEG_ASR(CV, test_dataset, args.buff_dir, args)
            print('CV' + str(CV), '\t', file = r)
            print('WER = %0.4f' % WER, file = r)
    r.close()
