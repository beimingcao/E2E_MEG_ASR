import numpy as np
import librosa
import torch
import random
import torchaudio


class Transform_Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, X):
        for t in self.transforms:
            X = t(X)
        return X
        
class MEG_dim_selection(object):
    def __init__(self, dim_range, sensor_name_idx = None):
        self.dim_range = dim_range
        self.sensor_name_idx = sensor_name_idx
    def __call__(self, X):
        # When using start and end indices to select sensors
        if self.sensor_name_idx is None:
            if len(self.dim_range) % 2 != 0:            
                raise Exception("Nums of starts and ends don't match, please check!")
            sel = []   
            for i in range(len(self.dim_range)//2):
                start = self.dim_range[2*i]
                end = self.dim_range[2*i+1]            
                _X = X[:,start:end]           
                sel.append(_X)   
            X_sel = np.hstack(sel)
            return X_sel
        else:
            return X[:, self.sensor_name_idx]

class low_pass_filtering(object):
    def __init__(self, cutoff_freq, fs):
        
        self.fs = fs
        self.cutoff_freq = cutoff_freq
    def __call__(self, I):
        if self.cutoff_freq >= self.fs//2:
            raise Exception("Cutoff freq must be lower than half of the sampling rate, based on the Nyquist criterion")
        from scipy.signal import butter, lfilter
        cutoff_norm = self.cutoff_freq/(self.fs/2)
        b, a = butter(5, cutoff_norm, btype='low', analog=False)
        I_filtered_list = []
        for i in range(I.shape[1]):
            _I = lfilter(b, a, I[:,i])
            I_filtered_list.append(_I)
            I_filtered = np.vstack(I_filtered_list)

        return I_filtered.T
        
class MEG_framing(object):
    def __init__(self, frame_len, frame_shift, drop_last = True):
        
        self.frame_len = frame_len
        self.frame_shift = frame_shift
        self.drop_last = drop_last
    def __call__(self, I):
        frame_overlap = self.frame_len - self.frame_shift
        
        num_frames = np.abs(I.shape[0] - frame_overlap) // np.abs(self.frame_len - frame_overlap)
        rest_samples = np.abs(I.shape[0] - frame_overlap) % np.abs(self.frame_len - frame_overlap)
                
        if rest_samples != 0:
            pad_signal_length = int(self.frame_shift - rest_samples)
            z = np.zeros((pad_signal_length,I.shape[1]))
            pad_signal = np.append(I, z, axis = 0)
            num_frames += 1
        else:
            pad_signal = I

        idx1 = np.tile(np.arange(0, self.frame_len), (num_frames, 1))
        idx2 = np.tile(np.arange(0, num_frames * self.frame_shift, self.frame_shift),(self.frame_len, 1)).T
        indices = idx1 + idx2
        frames = pad_signal[indices.astype(np.int32, copy=False),:]
        
        if self.drop_last == True:
            frames = frames[:-1,:,:]
        
        return frames
        
class MEG_freq_feats(object):
    """
    Input: 2-D MEG data array (shape: (samples, channels/sensors))
    ---
    For more details about the first 5 argunebts, please see the signal.spectrogram() function of the scipy package:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
    
    fs:
        Sampling frequency in Hz (default: 1000)
    window:
        Window to use (default: ('tukey', 0.25))
    nperseg:
        Length of each segment (default: 128)
    noveral:
        Overlap between segments (default: None)
    nfft:
        Length of the FFT (default: None)
    log_transform:
        Apply log-transformation to the spectrogram? (default: True)
    freq_range:
        Frequency range to retain in the spectrogram (default: (0, 130), from 0 Hz to 130 Hz) 
    ---
    Output: 3-D array containing spectrograms of all channels (shape: (channels/sensors, frequency, time))
    """
    def __init__(self, fs = 1000, window = ('tukey', 0.25), 
                 nperseg = 128, noverlap = None, nfft = None,
                log_transform = True,
                freq_range = (0, 130)):
        self.fs = fs
        self.window = window
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft
        self.log_transform = log_transform
        self.freq_min, self.freq_max = freq_range
        
    def __call__(self, I):
        from scipy import signal
        
        all_specs = []
        
        # Loop through all channels/sensors
        for i in range(I.shape[1]):
            
            # Get the MEG timeseries of the current channel and its spectrogram
            X = I[:, i]
            f, t, spec = signal.spectrogram(X, fs = self.fs, window = self.window,
                                            nperseg = self.nperseg, noverlap = self.noverlap,
                                            nfft = self.nfft)

            # Apply log transformation if needed
            if self.log_transform: 
                spec = np.log(spec)
            
            if all_specs:
                spec = spec[idx_band, :]
                all_specs.append(spec)
            else:
                # Keep just the part of the spectrogram within the required frequency band
                # (idx_band needs to be computed only at the first iteration since it should be the 
                # same across all channels/sensors)
                idx_band = np.logical_and(f >= self.freq_min, f <= self.freq_max)
                spec = spec[idx_band, :]
                all_specs.append(spec)
        
        all_specs = np.array(all_specs)
        return all_specs



class MEG_MVN(object):
    def __init__(self, X_mean, X_std):
        self.X_mean = X_mean
        self.X_std = X_std
    def __call__(self, X):
        X_norm = (X - self.X_mean)/self.X_std
        return X_norm

############### EMA transformation #######################

class FixMissingValues(object):  

    def __call__(self, MEG):        
        from scipy.interpolate import interp1d
        
        MEG_fixed = np.zeros(MEG.shape)
        for i in range(MEG.shape[1]):
            xnew = np.arange(len(MEG[:,i]))
            zero_idx = np.where(np.isnan(MEG[:,i]))
            xold = np.delete(xnew,zero_idx)
            yold = np.delete(MEG[:,i], zero_idx)  
            f = interp1d(xold,yold)
            ynew = f(xnew)
            MEG_fixed[:,i] = f(xnew)

        return MEG_fixed

class apply_delta_deltadelta(object):
    # Adopted from nnmnkwii source code https://github.com/r9y9/nnmnkwii
    
    def delta(self, x, window):

        T, D = x.shape
        y = np.zeros_like(x)
        for d in range(D):
            y[:, d] = np.correlate(x[:, d], window, mode = "same")
        return y
    
    def apply_delta_windows(self, x, windows):

        T, D = x.shape
        assert len(windows) > 0
        combined_features = np.empty((T, D * len(windows)), dtype=x.dtype)
        for idx, (_, _, window) in enumerate(windows):
            combined_features[:, D * idx:D * idx + D] = self.delta(x, window)
        return combined_features
    
    def __call__(self, ema):
    
        windows = [(0, 0, np.array([1.0])), 
                   (1, 1, np.array([-0.5, 0.0, 0.5])),
                   (1, 1, np.array([1.0, -2.0, 1.0]))]
        
        ema_delta = self.apply_delta_windows(ema, windows)
        
        return ema_delta



############### ATS transformation #######################

class Fix_EMA_MissingValues_ATS(FixMissingValues):  
    def __call__(self, ema, wav):        
        return super().__call__(ema), wav

class apply_delta_deltadelta_EMA_ATS(apply_delta_deltadelta):
    def __call__(self, ema, wav):        
        return super().__call__(ema), wav

class apply_EMA_MVN(object):
    def __init__(self, X_mean, X_std):
        self.X_mean = X_mean
        self.X_std = X_std
    def __call__(self, X, Y):
        X_norm = (X - self.X_mean)/self.X_std
        return X_norm, Y

class apply_WAV_MVN(object):
    def __init__(self, Y_mean, Y_std):
        self.Y_mean = Y_mean
        self.Y_std = Y_std
    def __call__(self, X, Y):
        Y_norm = (Y - self.Y_mean)/self.Y_std
        return X, Y_norm

class apply_EMA_MinMax(object):
    def __init__(self, X_min, X_max):
        self.X_min = X_min
        self.X_max = X_max
    def __call__(self, X, Y):
        X_norm = (X - self.X_min)/(self.X_max - self.X_min)
        return X_norm, Y

class apply_WAV_MinMax(object):
    def __init__(self, Y_min, Y_max):
        self.Y_min = Y_min
        self.Y_max = Y_max
    def __call__(self, X, Y):
        Y_norm = (Y - self.Y_min)/(self.Y_max - self.Y_min)
        return X, Y_norm
        
####### Data augmentation transform ########

class meg_time_mask(object):
    def __init__(self, prob = 0.5, mask_num = 20):
        self.prob = prob
        self.mask_num = mask_num
           
    def __call__(self, meg):
        masking = torchaudio.transforms.TimeMasking(time_mask_param=self.mask_num)
        if random.random() < self.prob:
            for j in range(meg.shape[0]):
                meg[j,:,:] = masking(meg[j,:,:].unsqueeze(0)).squeeze(0)                
        return meg
        
        
class meg_freq_mask(object):
    def __init__(self, prob = 0.5, mask_num = 20):
        self.prob = prob
        self.mask_num = mask_num
           
    def __call__(self, meg):
        masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=self.mask_num)
        if random.random() < self.prob:
            for j in range(meg.shape[0]):
                meg[j,:,:] = masking(meg[j,:,:].unsqueeze(0)).squeeze(0)                      
        return meg
        
'''
class ema_wav_random_scale(object):
    def __init__(self, prob = 0.5, rates = [0.8, 0.9, 1.1, 1.2]):
        self.prob = prob
        self.rates = rates
    def __call__(self, ema, wav):
        from scipy import ndimage
        if random.random() < self.prob:
            rate = random.choice(self.rates)
            print(rate)
            ema_align = np.empty([int(ema.shape[0]*rate+0.5), ema.shape[1]])
            wav_align = np.empty([int(wav.shape[0]*rate+0.5), wav.shape[1]])   
            for i in range(ema.shape[1]):
                ema_align[:,i] = ndimage.zoom(ema[:,i], rate)
            for i in range(wav.shape[1]):
                wav_align[:,i] = ndimage.zoom(wav[:,i], rate)
        return ema_align, wav_align
'''
class ema_wav_random_scale(object):
    def __init__(self, prob = 0.5, rates = [0.8, 0.9, 1.1, 1.2]):
        self.prob = prob
        self.rates = rates
    def __call__(self, ema, wav):
        from scipy import ndimage
        if random.random() < self.prob:
            rate = random.choice(self.rates)
            ema_align = np.empty([ema.shape[0], int(ema.shape[1]*rate+0.5), ema.shape[2]])
            wav_align = np.empty([wav.shape[0], int(wav.shape[1]*rate+0.5), wav.shape[2]])   
            for i in range(ema.shape[2]):
                ema_align[:,:,i] = ndimage.zoom(ema[:,:,i], rate)
            for i in range(wav.shape[2]):
                wav_align[:,:,i] = ndimage.zoom(wav[:,:,i], rate)
            ema, wav = torch.from_numpy(ema_align), torch.from_numpy(wav_align)
        return ema, wav
        
class ema_wav_random_rotate(object):
    def __init__(self, prob = 0.5, angle_range = [-30, 30]):
        self.prob = prob
        self.angle_range = angle_range
        
    def rotation(self, EMA, angle):
        import math
        rotate_matrix = [[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]]
        EMA_rotated = np.zeros((EMA.shape[0], EMA.shape[1], EMA.shape[2]))

        for i in range(int((EMA.shape[2])/2)):

            sensor_2D = EMA[:,:,2*i:2*i+2]
            sensor_2D_rotated = np.dot(sensor_2D, rotate_matrix)
            EMA_rotated[:,:,2*i:2*i+2] = sensor_2D_rotated

        return EMA_rotated        
   
    def __call__(self, ema, wav):
        if random.random() < self.prob:
            angle = random.randint(self.angle_range[0], self.angle_range[1])
            ema = torch.from_numpy(self.rotation(ema, angle))
                        
        return ema, wav
        
class ema_sin_noise(object):
    def __init__(self, prob = 0.5, noise_energy_ratio = 0.1, noise_freq  = 40, fs=100):
        self.prob = prob
        self.noise_energy_ratio = noise_energy_ratio
        self.noise_freq = noise_freq
        self.fs = fs
        
    def noise_injection(self, ema, noise_energy_ratio, noise_freq, fs):
        for i in range(ema.shape[0]):
            x = np.arange(ema.shape[3])
            sin_noise = torch.outer(torch.Tensor(np.sin(2 * np.pi * noise_freq * x / fs)), torch.abs(ema.mean(3)[i,0]))
            ema[i,0] = ema[i,0] + sin_noise.T
        return ema
        
    def __call__(self, ema):
        if random.random() < self.prob:
            ema = self.noise_injection(ema, self.noise_energy_ratio, self.noise_freq, self.fs) 
        return ema

class ema_time_mask(object):
    def __init__(self, prob = 0.5, mask_num = 20):
        self.prob = prob
        self.mask_num = mask_num
           
    def __call__(self, ema):
        masking = torchaudio.transforms.TimeMasking(time_mask_param=self.mask_num)
        if random.random() < self.prob:
            for j in range(ema.shape[0]):
                ema[j,0,:,:] = masking(ema[j,0,:,:])                 
        return ema
        
        
class ema_freq_mask(object):
    def __init__(self, prob = 0.5, mask_num = 20):
        self.prob = prob
        self.mask_num = mask_num
           
    def __call__(self, ema):
        masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=self.mask_num)
        if random.random() < self.prob:
            for j in range(ema.shape[0]):
                ema[j,0,:,:] = masking(ema[j,0,:,:])                      
        return ema

class ema_random_rotate(object):
    def __init__(self, prob = 0.5, angle_range = [-30, 30]):
        self.prob = prob
        self.angle_range = angle_range
        
    def rotation(self, EMA, angle):
        import math

        rotate_matrix = np.matrix([[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]])
        EMA_rotated = np.zeros((EMA.shape[0], 1, EMA.shape[2], EMA.shape[3]))
        for j in range(EMA.shape[0]):
            for i in range(int((EMA.shape[2])/2)):
                sensor_2D = EMA[j,0,[2*i, 2*i+1],:]
                sensor_2D_rotated = np.dot(sensor_2D.T, rotate_matrix)
                EMA_rotated[j,0,[2*i, 2*i+1],:] = sensor_2D_rotated.T

        return EMA_rotated        
   
    def __call__(self, ema):
        if random.random() < self.prob:
            angle = random.randint(self.angle_range[0], self.angle_range[1])
            ema = torch.from_numpy(self.rotation(ema, angle))                        
        return ema

