#! /usr/bin/env/python

# Calculates acoustic features for ASR Applications

import numpy as np 
from scipy.fftpack import dct
import os 
from scipy.io import wavfile


def hz2mel(hz: float):
    """Converts a value in Hertz to Mels.

    :param hz: a value in Hz. This can also be a np array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * np.log10(1+hz/700.)


def mel2hz(mel: float):
    """Converts a value in Mels to Hertz.

    :param mel: a value in Mels. This can also be a np array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700*(10**(mel/2595.0)-1)


def preemphasis(signal: np.ndarray, k=0.95):
    """Performs preemphasis on the input signal.
    
    The pre-emphasis filter is represented by the difference equation
    p[n] = x[n] - k * x[n-1]

    :signal: The signal to filter.
    :param k: The preemphasis coefficient. 0 is no filter, default is 0.95.
    :returns: The filtered signal.
    """
    return np.append(signal[0], signal[1:] - k * signal[:-1])


def get_log_feats(fbank_feats: np.ndarray):
    """Computes the logarithm of the fbank feats.
    
    :param fbank_feats: Computed Filterbank features. 
    Make sure that these features do not have zero anywhere as you cannot then take the log.
    :returns log_fbank_feats
    """
    return np.log(fbank_feats)


def get_mel_bank(num_filters=80, lowfreq=0, highfreq=8000, nfft=512, sampling_rate=16000):
    """Computes the mel scaling triangular filterbank function.
    
    :param num_filters: Number of mel coefficients
    :param lowfreq: Low Band Edge where Mel Filterbanks start (0 Hz)
    :param highfreq: Highest Band Edge that the Mel Filterbanks go to (usually Sampling Rate // 2)
    :returns fbank: Mel filberbanks of shape (num_filters, NFFT//2+1)
    """

    ## Get center points evenly spaced on the Mel Frequency scale
    lower_mel_frequency = hz2mel(lowfreq)
    high_mel_frequency = hz2mel(highfreq)
    mel_centers = np.linspace(lower_mel_frequency, high_mel_frequency, num_filters + 2)
    
    ## Now convert the centers back to Frequency scale to get the filters using normal frequency scale
    freq_bins = np.floor(((nfft+1)*mel2hz(mel_centers)/sampling_rate))
    fbank = np.zeros([num_filters, nfft//2 + 1])
    for j in range(0, num_filters):
        for i in range(int(freq_bins[j]), int(freq_bins[j+1])):
            fbank[j,i] = (i - freq_bins[j]) / (freq_bins[j+1]-freq_bins[j])
        for i in range(int(freq_bins[j+1]), int(freq_bins[j+2])):
            fbank[j,i] = (freq_bins[j+2]-i) / (freq_bins[j+2]-freq_bins[j+1])
    return fbank


def lifter(cepstra: np.ndarray, L=22):
    """Applys a cepstral lifter the the matrix of cepstra. This has the effect of increasing the
    magnitude of the high frequency DCT coeffs.

    :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
    :param L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
    """
    if L > 0:
        _,ncoeff = np.shape(cepstra)
        n = np.arange(ncoeff)
        lift = 1 + (L/2.)*np.sin(np.pi*n/L)
        return lift * cepstra
    else:
        # values of L <= 0, do nothing
        return cepstra


## Below are two wrapper functions to compute LMF and MFCC

def compute_lmf_feats(
    raw_signal: np.ndarray,
    window_length,
    overlap_length,
    sampling_rate,
    preemph=True,
    mel_low_freq=0,
    mel_high_freq=8000,
    num_mel_filters=80
):
    """Computes log mel filterbank (LMF) features for a given signal.

    :param raw_signal: Input audio signal
    :param window_length: Window length in s
    :param overlap_length: Overlap length in s
    :param sampling_rate: Sampling Rate in Hz
    :param preemph: Do Pre-emphasis on audio signal
    :param mel_low_freq: Low Band Edge where Mel Filterbanks start (0 Hz)
    :param mel_high_freq: Highest Band Edge that the Mel Filterbanks go to (usually Sampling Rate // 2)
    :param num_mel_filters: Number of filters in the mel filterbank, which is usually 23/40/80
    :returns log_mel_fbanks: the log mel filterbank (LMF) features     
    """

    ## calculate the window length and overlap length in samples
    window_sample_length = int(window_length * sampling_rate)
    print("W=", window_sample_length)
    overlap_sample_length = int(overlap_length * sampling_rate)
    print("O=", overlap_sample_length)
    
    ## Step 0: Perform Pre-emphasis in the Time domain using the provided pre-emphasis function
    pre_emph_signal = preemphasis(raw_signal)
    print("size of pre-emphasis signal:", pre_emph_signal.shape)
    
    ## Step 1: Get the windowed signal (frames)
    framed_signal = frame_with_overlap(
        pre_emph_signal, window_sample_length, overlap_sample_length
    )
    print("T=", framed_signal.shape[0])
    print("win:", framed_signal.shape)
    ## Step 2: Compute the powerspectrum of the Framed Signal
    power_spectrum, nfft = compute_powerspec(framed_signal, window_sample_length)  
    print("pow:", power_spectrum.shape)
    ## Step 3: obtain the Mel Triangular Weighting Filterbanks with the given specifications
    melfilterbanks = get_mel_bank(
        num_mel_filters, mel_low_freq, mel_high_freq, nfft, sampling_rate
    )
    print("mel weighted filterbanks:", melfilterbanks.shape)
    ## Step 4: Get Mel Fbank features 
    mel_fbank_feats = get_mel_fbank_feat(power_spectrum, melfilterbanks)
    print("mel filterbanks features:", mel_fbank_feats.shape)
    ## Step 5: Get Log Fbank Features
    lmf_feats = get_log_feats(mel_fbank_feats)
    print("lmf feats:", lmf_feats.shape)
    
    return lmf_feats


def compute_mfcc_feats(
    raw_signal: np.ndarray,
    window_length,
    overlap_length,
    sampling_rate,
    preemph=True,
    mel_low_freq=0,
    mel_high_freq=8000,
    num_mel_filters=80,
    num_ceps=23,
    ceplifter=22
):
    """Computes MFCC features for a given signal.

    :param raw_signal: Input audio signal
    :param window_length: Window length in s
    :param overlap_length: Overlap length in s
    :param sampling_rate: Sampling Rate in Hz
    :param preemph: Do Pre-emphasis on audio signal
    :param mel_low_freq: Low Band Edge where Mel Filterbanks start (0 Hz)
    :param mel_high_freq: Highest Band Edge that the Mel Filterbanks go to (usually Sampling Rate // 2)
    :param num_mel_filters: Number of filters in the mel filterbank
    :param num_ceps: Number of cepstral coefficients considered from the DCT
    :param L: Last "L" coefficients to apply filtering in the cepstral domain on (Liftering)
    :returns mfcc_coeffs: Returns the mfcc_coefficients     
    """
    ## Step 6 : MFCC feats 
    
    ## Get the Log Mel Filterbank features
    lmel_feats = compute_lmf_feats(
        raw_signal, window_length, overlap_length, sampling_rate,
        preemph, mel_low_freq, mel_high_freq, num_mel_filters
    )
    
    ## Compute the DCT of the Log Mel Features
    mfcc_feats = dct(lmel_feats, type=2, axis=1, norm='ortho')[:,:num_ceps]
    
    ## Filtering in the Cepstral domain (called Liftering)
    liftered_mfcc = lifter(mfcc_feats, ceplifter)

    return liftered_mfcc


##-------------------------------------END OF PROVIDED UTILS. CODE TO BE FILLED STARTS BELOW-----------------------------

def frame_with_overlap(signal: np.ndarray, window_length, overlap_length):
    """Creates overlapping windows (frames) of the input signal.
    
    :param signal: 1-D audio signal of shape (num_samples,)
    :param window_length: window length (in samples)
    :param overlap_length: overlapping length (in samples)
    :returns: 2-D array of shape (num_frames, window_length) 
    """
    print("signal shape:", signal.shape)
    print("W=", window_length)
    print("O=", overlap_length)
    if len(signal) < window_length:
        padded_arr = np.pad(signal, (0, window_length - len(signal)), 'constant')
        return padded_arr
    
    overlapping_windows = signal[:window_length]
    start_window_i = window_length - overlap_length
    end_window_i = start_window_i + window_length
        
    while end_window_i < len(signal):
        window = signal[start_window_i : end_window_i]
        overlapping_windows = np.vstack((overlapping_windows, window))
        start_window_i += window_length
        start_window_i -= overlap_length
        end_window_i = start_window_i + window_length 
    # pad 0s on last window if last portion of array that window covers is less than W
    if end_window_i >= len(signal):
        unpadded_arr = signal[start_window_i:]
        padded_arr = np.pad(unpadded_arr, (0, window_length - len(unpadded_arr)), 'constant')
        overlapping_windows = np.vstack((overlapping_windows, padded_arr))
    print("num frames, window:", overlapping_windows.shape)
    return overlapping_windows


def compute_powerspec(framed_signal: np.ndarray, window_sample_length=160):
    """Computes the power spectrum from the framed signal.
    
    :param framed_signal: framed signal of shape (num_frames, window_sample_length)
    :param window_sample_length: Length of the window in samples 
    :returns power_spectrum: the Power Spectrum of the Short Time Fourier Transform of the signal.
    :returns nfft: the Fourier Transform dimension for the Fast Fourier Transform (always a power of two >= window length)
    If framed signal is a [T,w] matrix, output power_spec will be [T,(NFFT/2+1)]
    """
    
    ## Compute the number of FFT Points you need: nfft
    nfft = np.power(2, int(np.ceil(np.log2(window_sample_length))))
    
    ## Get the STFT
    stft = np.fft.rfft(framed_signal, n=nfft)
    
    ## Get the Magnitude of the STFT
    mag_stft = np.abs(stft)
    
    ## Get the Power Spectrum
    power_spectrum = np.divide(np.power(mag_stft, 2), nfft)
    
    return power_spectrum, nfft


def get_mel_fbank_feat(power_spec: np.ndarray, mel_filterbanks: np.ndarray, eps=1e-08):
    """Computes the Mel Filterbank features as the dot product of the Power-spectrum and the Mel Filter banks.

    :param power_spec: power Spectrum of the STFT Magnitude, shape: (T, (NFFT/2+1))
    :param mel_filterbanks: Mel Scale filterbank function, shape: (num_filters, NFFT//2+1)
    :param eps: Small value used where feature value is zero to make sure that the log is valid
    :returns mel_fbank: Mel Filterbank features of signal
    """
    '''
    mel_fbank = []
    for row in range(power_spec.shape[0]):
        dot = np.dot(mel_filterbanks, power_spec[row, :])
        mel_fbank.append(dot)
    mel_fbank = np.array(mel_fbank)
    for row in range(mel_fbank.shape[0]):
        mel_fbank[row, mel_fbank[row, :] == 0] = eps
    return mel_fbank
    '''
    mel_fbank = np.dot(power_spec, mel_filterbanks.T)
    mel_fbank += eps
    '''
    for row in range(mel_fbank.shape[0]):
        mel_fbank[row, mel_fbank[row, :] == 0] = eps
    '''
    return mel_fbank

if __name__ == "__main__":
    sampling_rate, audio = wavfile.read(os.path.join("example_data", "example_audio.wav"))
    feat_arrays = np.load(os.path.join("example_data", "example_feats.npz"))
    lmf_feats = feat_arrays["lmel"]
    mfcc_feats = feat_arrays["mfcc"]

    my_lmf_feats = compute_lmf_feats(
        raw_signal=audio,
        window_length=0.025,
        overlap_length=0.01,
        sampling_rate=sampling_rate,
        preemph=True,
        mel_low_freq=0,
        mel_high_freq=8000,
        num_mel_filters=80
    )
    my_mfcc_feats = compute_mfcc_feats(
        raw_signal=audio,
        window_length=0.025,
        overlap_length=0.01,
        sampling_rate=sampling_rate,
        preemph=True,
        mel_low_freq=0,
        mel_high_freq=8000,
        num_mel_filters=80
    )
    print(my_lmf_feats[0])
    print(lmf_feats[0])
    assert np.allclose(my_lmf_feats, lmf_feats), "LMF failed on the example audio."
    assert np.allclose(my_mfcc_feats, mfcc_feats), "MFCC failed on the example audio."

    print("---------- Success! ----------")