import os
import glob
import numpy as np
from scipy import ndimage
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
from utils.IO_func import read_file_list, load_binary_file, array_to_binary_file, load_Haskins_ATS_data
import librosa

class MEG_PHONE_ASR(Dataset):
    def __init__(self, MEG_path_list, LAB_list, idx_list, transforms=None):
        self.MEG_path_list = MEG_path_list
        self.LAB_list = LAB_list
        self.idx_list = idx_list
        self.transforms = transforms       
        self.data = []
        for idx in idx_list:
            MEG = torch.load(MEG_path_list[idx])
            Phone_seq = LAB_list[idx].strip().split(' ')
            Phone_seq_with_sil = ['SIL'] + Phone_seq + ['SIL']
            text_transform = PhoneTransform()
            label = torch.Tensor(text_transform.text_to_int(Phone_seq_with_sil))
            self.data.append((MEG, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        MEG, PHN = self.data[idx]     
        if self.transforms is not None:
            MEG, PHN = self.transforms(MEG), PHN         
        return (MEG, PHN)

def phn_file_parse(phn_path):
    
    import csv
    import numpy as np
    import re
    
    reader = csv.reader(open(phn_path))
    data_list = list(reader)
    
    phone_seq = []
    starts, ends = [], []
    for phone_start_end in data_list:       
        phone_tag = re.split(r'\t+', phone_start_end[0])
        phone_seq.append(phone_tag[0].upper())
        starts.append(float(phone_tag[1]))
        ends.append(float(phone_tag[2]))
        
    return phone_seq, starts, ends

class PhoneTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
        char_map_str = """
        SIL 0
        AA 1
        AE 2
        AH 3
        AO 4
        AW 5
        AY 6
        B 7
        CH 8
        D 9
        DH 10
        EH 11
        ER 12
        EY 13
        F 14
        G 15
        HH 16
        IH 17
        IY 18
        JH 19
        K 20
        L 21
        M 22
        N 23
        NG 24
        OW 25
        OY 26
        P 27
        R 28
        S 29
        SH 30
        T 31
        TH 32
        UH 33
        UW 34
        V 35
        W 36
        Y 37
        Z 38
        ZH 39
        """
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == 'SHH':
                ch = 16
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[int(i)])
        return string
