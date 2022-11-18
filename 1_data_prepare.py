import time
import yaml
import os
import torch
from utils.IO_func import read_file_list, load_binary_file, array_to_binary_file, load_Haskins_SSR_data, save_phone_label, save_word_label
from shutil import copyfile
from utils.transforms import Transform_Compose, MEG_dim_selection, FixMissingValues, low_pass_filtering
from scipy.io import wavfile


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
    out_folder = os.path.join(args.buff_dir, 'data')
    sampling_rate = config['MEG_data']['sampling_rate']
    
    transforms = [FixMissingValues()] if config['Transforms']['fix_missing_value'] == True else [] #
    if config['MEG_data']['dim_selection'] == True:
        sel_dim = config['MEG_data']['selected_dims']
        transforms.append(MEG_dim_selection(sel_dim))
        
    if config['MEG_data']['low_pass_filtering'] == True:
        cutoff_freq = config['MEG_data']['LP_cutoff_freq']
        transforms.append(low_pass_filtering(cutoff_freq, sampling_rate))
    


    transforms_all = Transform_Compose(transforms)
    
    for SPK in SPK_list:
        out_folder_SPK = os.path.join(out_folder, SPK)
        if not os.path.exists(out_folder_SPK):
            os.makedirs(out_folder_SPK)

        fileset_path_SPK = os.path.join(fileset_path, SPK)
        file_id_list = read_file_list(os.path.join(fileset_path_SPK, 'file_id_list.scp'))

        for file_id in file_id_list:
            data_path_spk = os.path.join(data_path, file_id[:3])
            mat_path = os.path.join(data_path_spk, 'data/'+ file_id + '.mat')
            EMA, fs_ema, wav, sent, phone_label, word_label, word_label_ms = load_Haskins_SSR_data(mat_path, file_id, sel_sensors, sel_dim)
            EMA = transforms_all(EMA) 
            WAV_out_dir = os.path.join(out_folder_SPK, file_id + '.wav')
            EMA_out_dir = os.path.join(out_folder_SPK, file_id + '.ema')
            PHO_out_dir = os.path.join(out_folder_SPK, file_id + '.phn')
            WRD_out_dir = os.path.join(out_folder_SPK, file_id + '.wrd')

            if file_id in bad_label_list:
                phone_label = word_label_ms
                sent = word_label

            wavfile.write(WAV_out_dir, 44100, wav )
            save_word_label(sent[0], WRD_out_dir)
            save_phone_label(phone_label, PHO_out_dir)
            array_to_binary_file(EMA, EMA_out_dir)
 
            
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_dir', default = 'conf/conf.yaml')
    parser.add_argument('--buff_dir', default = 'current_exp')

    args = parser.parse_args()
    data_processing(args)
