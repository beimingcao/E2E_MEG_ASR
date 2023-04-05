import time
import yaml
import os
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from utils.models_spec import SpeechRecognitionModel
from utils.models_spec import save_model
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from utils.utils import EarlyStopping, IterMeter, GreedyDecoder
import torch.nn.functional as F
from jiwer import wer
import random
import numpy as np
import torchaudio
from utils.database import PhoneTransform
from speechbrain.lm.ngram import BackoffNgramLM
from speechbrain.lm.arpa import read_arpa
import io
import math
from typing import List, Tuple
from collections import defaultdict, Counter
from string import ascii_lowercase

def log_add(args: List[int]) -> float:
    """
    Stable log add
    """
    if all(a == -float('inf') for a in args):
        return -float('inf')
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max) for a in args))
    return a_max + lsp

def prefix_beam_search(probs, LM, blank_index=40, beam_size = 5, alpha=0.30):
    
    cur_probs = probs.squeeze(1)
    cur_hyps = [(tuple(), (0.0, -float('inf')))]
    for t in range(cur_probs.shape[0]):
        logp = cur_probs[t].cpu()
        next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
        top_k_logp, top_k_index = logp.topk(beam_size)  # (beam_size,)
        for s in top_k_index:
            s = s.item()
            ps = logp[s].item()
            for prefix, (pb, pnb) in cur_hyps:
                last = prefix[-1] if len(prefix) > 0 else None
                if s == blank_index:  # blank
                    n_pb, n_pnb = next_hyps[prefix]
                    n_pb = log_add([n_pb, pb + ps, pnb + ps])
                    next_hyps[prefix] = (n_pb, n_pnb)
                elif s == last:
                    #  Update *ss -> *s;
                    n_pb, n_pnb = next_hyps[prefix]
                    n_pnb = log_add([n_pnb, pnb + ps])
                    next_hyps[prefix] = (n_pb, n_pnb)
                    # Update *s-s -> *ss, - is for blank
                    n_prefix = prefix + (s, )
                    n_pb, n_pnb = next_hyps[n_prefix]
                    n_pnb = log_add([n_pnb, pb + ps])
                    next_hyps[n_prefix] = (n_pb, n_pnb)
                else:
                    # Beginning of a sentence
                    cur_text = PhoneTransform().int_to_text([s])
                    if last == None:
                        LM_score = (LM.logprob(cur_text[0], context = ("SIL",)))
                    else:
                        LM_score = (LM.logprob(cur_text[0], context = (PhoneTransform().int_to_text([last])[0],)))

                    n_prefix = prefix + (s, )
                    n_pb, n_pnb = next_hyps[n_prefix]
                    n_pb = log_add([n_pb, pb + ps + LM_score * alpha])
                    n_pnb = log_add([n_pnb, pnb + ps + LM_score * alpha])
                #    n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                    next_hyps[n_prefix] = (n_pb, n_pnb)

            # 2.2 Second beam prune
        next_hyps = sorted(next_hyps.items(),
                            key=lambda x: log_add(list(x[1])),
                            reverse=True)
        cur_hyps = next_hyps[:beam_size]
    hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in cur_hyps]
    return hyps

def GreedyDecoder(output, labels, label_lengths, blank_label=40, collapse_repeated=True):

    text_transform = PhoneTransform()

    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):

        decode = []
        targets.append(text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
        decodes.append(text_transform.int_to_text(decode))
    return decodes, targets


def data_processing_DeepSpeech(data, transforms = None):
    meg = []
    labels = []
    input_lengths = []
    label_lengths = []
    
    for x, y in data:
        if transforms is not None:
            x = transforms(x)
        x = x.transpose(0,2)
        meg.append(x)
        labels.append(y)
        input_lengths.append(x.shape[0] // 2)
        label_lengths.append(len(y))
        
    meg = torch.nn.utils.rnn.pad_sequence(meg, batch_first=True)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)        
    
    return meg, labels, input_lengths, label_lengths

def test_MEG_ASR(CV, test_dataset, exp_output_folder, args):
    ### Dimension setup ###
    config = yaml.load(open(args.conf_dir, 'r'), Loader=yaml.FullLoader)
    if config['MEG_data']['dim_selection'] == True:
        sel_dim = config['MEG_data']['selected_dims']
        if all(isinstance(x, int) for x in sel_dim):
            in_dim = 0
            for i in range(len(sel_dim) // 2):
                in_dim += sel_dim[2*i+1] - sel_dim[2*i]
        elif all(isinstance(x, str) for x in sel_dim):
            in_dim = len(sel_dim)
        else:
            raise Exception('Incorrect format for specifying selected sensors.')
    else:
        in_dim = config['MEG_data']['max_dim']
    
    ### Model setup ###
    n_cnn_layers = config['NN_setup']['n_cnn_layers']
    n_rnn_layers = config['NN_setup']['n_rnn_layers']    
    rnn_dim = config['NN_setup']['rnn_dim']
    stride = config['NN_setup']['stride']
    dropout = config['NN_setup']['dropout']
            
    ### Test ###
    prefix_BS = config['Testing_setup']['prefix_beam'] 
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                            batch_size=1,
                            shuffle=False,
                            collate_fn=lambda x: data_processing_DeepSpeech(x, transforms = None))
                            
    model_out_folder = os.path.join(exp_output_folder, 'trained_models')
        
    SPK_model_path = os.path.join(model_out_folder)
    model_path = os.path.join(SPK_model_path, 'CV' + str(CV) + '_DS.pt')
    model = SpeechRecognitionModel(n_cnn_layers, n_rnn_layers, rnn_dim, 41, in_dim, stride, dropout)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    beam_size = config['Testing_setup']['beam_size']
    data_session = config['Data_setup']['session']
    LM_weight = config['Testing_setup']['LM_weight']

    LM_path = os.path.join(args.buff_dir, data_session, 'data_CV', 'CV' + str(CV), 'LM.arpa')
    with open(LM_path, 'r') as f:
        num_grams, ngrams, backoffs = read_arpa(f)
    LM = BackoffNgramLM(ngrams, backoffs)

    pred = []
    label = []

    for batch_idx, _data in enumerate(test_loader):
        meg, labels, input_lengths, label_lengths = _data 
        
        x = meg.transpose(1,3) 
        output = model(x)  # (batch, time, n_class)

        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1) # (time, batch, n_class)   

        if prefix_BS == False:
            decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths) 
            pred.append(' '.join(decoded_preds[0]))
        else:
            _, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths) 
            decoded_preds_label = prefix_beam_search(output, LM, beam_size = beam_size, alpha=0.30)
            decoded_preds = PhoneTransform().int_to_text(decoded_preds_label[0][0])
            pred.append(' '.join(decoded_preds))
 
        label.append(' '.join(decoded_targets[0]))
        
    error = wer(pred, label)
    for i, prediction in enumerate(pred):
        print('Predicted:', prediction)
        print('Target:', label[i])
        print('')
       
    return error




if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_dir', default = 'conf/conf_spec.yaml')
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
            #te = open(os.path.join(data_path_CV, 'train_data.pkl'), 'rb')
            test_dataset = pickle.load(te)
        
            WER = test_MEG_ASR(CV, test_dataset, args.buff_dir, args)
            print('CV' + str(CV), '\t', file = r)
            print('WER = %0.4f' % WER, file = r)
    r.close()
