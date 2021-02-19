#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reshift: A pitch discretization effect

Manuel Planton 2020
"""

import numpy as np
import scipy.signal as signal
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import soundfile as sf

# define default samplerate of 44100Hz and not 22050Hz
from presets import Preset
import librosa as _librosa
import librosa.display as _display
_librosa.display = _display
librosa = Preset(_librosa)
librosa['sr'] = 44100

from utility import Filterbank

# -----------------------------------------------------------------------------

def reshift(x, sr, scale='chromatic'):
    # parameters
    
    # window size of the pitch analysis
    pitch_N = 1024
    librosa['n_fft'] = pitch_N
    # hop size of the pitch analysis
    pitch_hop_size = 512
    librosa['hop_length'] = pitch_hop_size
    
    
    # zero padding because librosa pYIN does this internally
    x = np.concatenate((x, np.zeros(pitch_N - (x.size % pitch_N) - 1)))
    
    fmin = 60
    fmax = 2000
    f_0, voiced_flag, voiced_probs = librosa.pyin(x, fmin=fmin, fmax=fmax,
                                                  sr=sr, frame_length=pitch_N)
    
    # append one zero for correct length with f_0
    x = np.concatenate((x, np.zeros(1)))
    
    f_out = freq_scale(f_0, scale)
    
    # pitch shifting factor rho
    rho = f_out / f_0
    
    y = pitch_shift_rollers(x, sr, rho, pitch_hop_size)
    return (y / np.abs(np.max(y))) * np.abs(np.max(x)) # normalization

# -----------------------------------------------------------------------------

def pitch_shift_rollers(x, fs, psr, N, order=2, n=100):
    """
    Time variant 'Rollers' pitch-shifting algorithm.
    x: input signal
    fs: sampling rate
    psr: list of pitch-shifting ratios for x
    N: pitch analysis block size of the pitch tracking algorithm
    order: filter order of the used filter bank
    n: number of filters in the filter bank
    """
    # format pitch shifting ratio
    # unvoiced parts cause no pitch shift
    psr = np.nan_to_num(psr, nan=1) # TODO: in pYIN machen
    
    
    ### UPSAMPLING ###
    L = N
    
    # Do the upsampling with the pitch shifting ratio
    psr_up = np.repeat(psr, L) # this on its own is most likely sufficient
    # optional smoothing
    # control signal smoothing filter length
    #l = 32
    #h_smooth = signal.windows.triang(l) / (l-1)
    #psr_up = signal.convolve(psr_up, h_smooth, mode="same")
    
    ## OR ##
    
    # Do upsampling by inserting zeros and linear interpolation
    #psr_up = np.zeros(x.size)
    #psr_up[::L] = psr
    #h_interp = signal.windows.triang(2 * L - 1)
    #psr_up = signal.convolve(psr_up, h_interp, mode="same")
    
    ### UPSAMPLING ###
    
    
    # divide input into frequency bands
    filt_bank = Filterbank(n, order, fs)
    x_filtered = filt_bank.filt(x)
    
    # frequency shifting in every band
    out_signals = []
    t = np.linspace(0, x_filtered[0].size/fs, x_filtered[0].size)
    for i in range(len(x_filtered)):
        # calculate time variant carrier frequencies for every block
        fc = filt_bank.fcs[i]
        f_shift = fc * psr_up - fc
        
        # integrate frequency function for FM of the frequency shifting carrier
        w_int = integrate.cumulative_trapezoid(2*np.pi*f_shift, t, initial=0)
        
        # frequency shifting with time variable carrier frequency
        carrier = np.exp(1j*w_int)
        band = (signal.hilbert(x_filtered[i]) * carrier).real
        out_signals.append(band)
    
    # add bands together
    y = np.zeros(out_signals[0].size)
    for sig in out_signals:
        y += sig
    
    return y

# -----------------------------------------------------------------------------

def freq_scale(f_0, scale='chromatic', tune=440):
    """
    The discrete frequency scaling function for the desired pitch.
    frequency scaling: chromatic or wholetone
    f_0...input frequency
    tune...tuning frequency
    return discrete tuned frequencies
    """
    # TODO: add new scales like major, minor, etc.
    if scale in ('chromatic', 'chrom', 'c'):
        n_tones = 12
    elif scale in ('wholetone', 'whole', 'w'):
        n_tones = 6
    elif scale in ('thirds', 'third', 't'):
        n_tones = 3
    elif scale in ('tritones', 'tritone', 'tri'):
        n_tones = 2
    
    tone = n_tones * np.log2(f_0/tune)
    discrete = np.round(tone)
    return tune * (2 ** (discrete / n_tones))

# -----------------------------------------------------------------------------

def _my_plot(sig, sr, title=''):
    """Plot waveform and spectrogram of signal sig with ramplerate sr."""
    plt.figure()
    t = np.linspace(0, sig.size/sr, sig.size)
    plt.plot(t, sig)
    plt.title(title)

    D = librosa.amplitude_to_db(np.abs(librosa.stft(sig)), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax)
    ax.set(title=title)
    fig.colorbar(img, ax=ax, format="%+2.f dB")

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def _test():
    # input signal
    
    dur = 5.12579
    fmin = 500
    fmax = 4*500
    sr = 44100
    #x = librosa.chirp(fmin, fmax, sr=sr, duration=dur)
    
    x, sr = librosa.load('../../samples/Toms_diner.wav')
    
    pos = 5
    dur = 20
    # x, sr = librosa.load("../../samples/ave-maria.wav", offset=pos, duration=dur)
    
    y = reshift(x, sr, scale='w')
    
    _my_plot(x, sr, "original signal")
    
    _my_plot(y, sr, "processed signal")
    
    sf.write("test.wav", y, sr)


if __name__ == "__main__":
    _test()
