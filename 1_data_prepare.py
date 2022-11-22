import time
import yaml
import os
import glob
import numpy as np
import torch
from utils.IO_func import read_file_list, load_binary_file, array_to_binary_file, load_Haskins_SSR_data, save_phone_label, save_word_label
from shutil import copyfile
from utils.transforms import Transform_Compose, MEG_dim_selection, FixMissingValues, low_pass_filtering, MEG_framing


def data_processing(args):

    '''
    Load in data session involved, pre-processings like framing, dimension selection
    save them into binary files in the current_exp folder, 
    so that data loadin will be accelerated a lot.
    
    Transforms here are applied to all files, despite of training, validation, testing.
    Therefore, transforms like normalization will not be performed here.
    '''

    config_path = args.conf_dir       
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    data_path = config['Corpus']['path']
    data_session = config['Data_setup']['session']
    out_folder = os.path.join(args.buff_dir, data_session)
    sampling_rate = config['MEG_data']['sampling_rate']
    
    transforms = [FixMissingValues()] if config['MEG_data']['fix_missing_values'] == True else [] #
    if config['MEG_data']['dim_selection'] == True:
        sel_dim = config['MEG_data']['selected_dims']
        transforms.append(MEG_dim_selection(sel_dim))
        
    if config['MEG_data']['low_pass_filtering'] == True:
        cutoff_freq = config['MEG_data']['LP_cutoff_freq']
        transforms.append(low_pass_filtering(cutoff_freq, sampling_rate))
        
    if config['MEG_data']['framing'] == True:
        # (sample_num, MEG_dim) --> (frame_num, frame_len, MEG_dim)
        frame_len, frame_shift = config['MEG_data']['frame_len'], config['MEG_data']['frame_shift']
        drop_last = config['MEG_data']['drop_last']
        transforms.append(MEG_framing(frame_len, frame_shift, drop_last))
    
    transforms_all = Transform_Compose(transforms)
    
    data_session_path = os.path.join(data_path, data_session)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    # Please remove all "non-data" txt files
    data_file_path_list = glob.glob(data_session_path + '/*.txt')
    data_file_path_list.sort()
    for data_file_path in data_file_path_list:
        file_id = os.path.basename(data_file_path)
        data = np.loadtxt(data_file_path)
        data_transformed = transforms_all(data)
        data_Tensor = torch.from_numpy(data_transformed)
        file_name = file_id[:-4] + '.pt'
        out_dir = os.path.join(out_folder, file_name)
        print(f"Writing {file_name} into the folder {out_folder}")
        torch.save(data_Tensor, out_dir)
    



            
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_dir', default = 'conf/conf.yaml')
    parser.add_argument('--buff_dir', default = 'current_exp')

    args = parser.parse_args()
    data_processing(args)
