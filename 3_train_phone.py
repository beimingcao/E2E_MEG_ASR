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
from utils.utils import EarlyStopping, IterMeter, data_processing_DeepSpeech
import torch.nn.functional as F

import random
import numpy as np
import torchaudio

seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def train_MEG_ASR(CV, train_dataset, valid_dataset, exp_output_folder, args):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ### Dimension setup ###
    config = yaml.load(open(args.conf_dir, 'r'), Loader=yaml.FullLoader)

    ### Model setup ###
    n_cnn_layers = config['NN_setup']['n_cnn_layers']
    n_rnn_layers = config['NN_setup']['n_rnn_layers']    
    rnn_dim = config['NN_setup']['rnn_dim']
    stride = config['NN_setup']['stride']
    dropout = config['NN_setup']['dropout']
    
    ### Training setup ###
    learning_rate = config['Training setup']['learning_rate']
    batch_size = config['Training setup']['batch_size']
    epochs = config['Training setup']['epochs']
    early_stop = config['Training setup']['early_stop']
    patient = config['Training setup']['patient']
    train_out_folder = os.path.join(exp_output_folder, 'training')
    if not os.path.exists(train_out_folder):
        os.makedirs(train_out_folder)
    results = os.path.join(train_out_folder, 'CV' + str(CV) + '_train.txt')
    
    ### Model training ###
            
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: data_processing_DeepSpeech(x, transforms = None))                                
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: data_processing_DeepSpeech(x, transforms = None))
    ### **************** Make input dim flexible later                            
    model = SpeechRecognitionModel(n_cnn_layers, n_rnn_layers, rnn_dim, 41, 204, stride, dropout).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), learning_rate)
    criterion = torch.nn.CTCLoss(blank=40).to(device)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=int(len(train_loader)), epochs=epochs, anneal_strategy='linear')
    
    data_len = len(train_loader.dataset)
    if early_stop == True:
        print('Applying early stop.')
        early_stopping = EarlyStopping(patience=patient)
        
    iter_meter = IterMeter()
        
    with open(results, 'w') as r:    
        for epoch in range(epochs):
            model.train()
            loss_train = []
            for batch_idx, _data in enumerate(train_loader):
                meg, labels, input_lengths, label_lengths = _data 
                                   
                meg, labels = meg.to(device), labels.to(device)
  
                x = torch.mean(meg, 2).transpose(2, 3)               
                output = model(x)  # (batch, time, n_class)

                output = F.log_softmax(output, dim=2)
                output = output.transpose(0, 1) # (time, batch, n_class)

                loss = criterion(output, labels, input_lengths, label_lengths)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                iter_meter.step()
                
                loss_train.append(loss.detach().cpu().numpy())
            avg_loss_train = sum(loss_train)/len(loss_train)

            model.eval()
            loss_valid = []
            for batch_idx, _data in enumerate(valid_loader):  
                meg, labels, input_lengths, label_lengths = _data 
                meg, labels = meg.to(device), labels.to(device)           
                x = torch.mean(meg, 2).transpose(2, 3)    
                output = model(x)  # (batch, time, n_class)
                output = F.log_softmax(output, dim=2)
                output = output.transpose(0, 1) # (time, batch, n_class)
                loss = criterion(output, labels, input_lengths, label_lengths)    
                loss_valid.append(loss.detach().cpu().numpy())
            avg_loss_valid = sum(loss_valid)/len(loss_valid) 

            early_stopping(avg_loss_valid)
            if early_stopping.early_stop:
                break

            print('epoch %-3d \t train_loss = %0.5f \t valid_loss = %0.5f' % (epoch, avg_loss_train, avg_loss_valid))
            print('epoch %-3d \t train_loss = %0.5f \t valid_loss = %0.5f' % (epoch, avg_loss_train, avg_loss_valid), file = r)                           
                            
            model_out_folder = os.path.join(exp_output_folder, 'trained_models')
            if not os.path.exists(model_out_folder):
                os.makedirs(model_out_folder)
            if early_stopping.save_model == True:
                save_model(model, os.path.join(model_out_folder, 'CV' + str(CV) + '_DS.pt'))
    r.close()
    print('Training for CV' + str(CV) + ' is done.')       
           

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_dir', default = 'conf/conf.yaml')
    parser.add_argument('--buff_dir', default = 'current_exp')
    args = parser.parse_args()
    config = yaml.load(open(args.conf_dir, 'r'), Loader=yaml.FullLoader)
    data_session = config['Data_setup']['session']
    data_path = os.path.join(args.buff_dir, data_session, 'data_CV')
    CV_to_run = config['Data_setup']['CV_to_run']
   

    for CV in CV_to_run:
        data_path_CV = os.path.join(data_path, 'CV' + str(CV))

        tr = open(os.path.join(data_path_CV, 'train_data.pkl'), 'rb') 
        va = open(os.path.join(data_path_CV, 'valid_data.pkl'), 'rb')        
        train_dataset, valid_dataset = pickle.load(tr), pickle.load(va)
 
        train_MEG_ASR(CV, train_dataset, valid_dataset, args.buff_dir, args)   



