#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reshift: A pitch discretization effect

Manuel Planton 2020
"""

# define default samplerate of 44100Hz and not 22050Hz
# and fft length and hop size
from presets import Preset
import librosa as _librosa
import librosa.display as _display
_librosa.display = _display
librosa = Preset(_librosa)

librosa['sr'] = 44100
librosa['n_fft'] = 4096
librosa_hop_len = 2048
librosa['hop_length'] = librosa_hop_len

# other needed modules
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# -----------------------------------------------------------------------------

def reshift(x, sr, scale='chromatic'):
    # parameters
    
    # window size of the pitch analysis
    pitch_N = 4096
    # hop size of the pitch analysis
    pitch_hop_size = 2048
    
    # analysis window size for pitch shifting
    shift_N = pitch_hop_size
    # pitch shifting analysis overlap factor
    overlap_factor = 2
    
    f_0, voiced_flag = pitch_track(x, sr, pitch_N, pitch_hop_size)
    
    f_out = freq_scale(f_0, scale)
    
    # pitch shifting factor rho
    rho = f_out / f_0
    
    # TODO: manage synchronisation between control rate (pitch estimates
    # window size and hop size) and
    # audio frame rate (pitch shifting window size and hop size)
    
    y = pitch_shift(x, sr, rho, pitch_hop_size, shift_N, overlap_factor)
    return y

# -----------------------------------------------------------------------------

def pitch_track(x, sr, window_size, hop_size):
    """
    Pitch tracking function to decouple the particular pitch shifting
    algorithm from the task of pitch shifting.
    """
    fmin = 60
    fmax = 2000
    f_0, voiced_flag, voiced_probs = librosa.pyin(x, fmin=fmin, fmax=fmax, sr=sr, frame_length=window_size)
    return f_0, voiced_flag

# -----------------------------------------------------------------------------

def pitch_shift(x, sr, rho, rho_N, N, overlap_factor):
    """
    General time varying pitch shifting function according to rho
    This function decouples various methods for pitch shifting from the rest
    of the algorithm.
    x...input signal to be pitch shifted
    sr...sample rate of x
    rho...time varying pitch-shifting factor
    rho_N...window size of the validity of the pitch-shifting factor
    N...analysis window size of pitch shifting
    overlap_factor...factor of window overlap for OLA
    """
    method = "ola"
    
    if method == "rosa":
        y_disc = pitch_shift_rosa(x, rho, rho_N)
    elif method == "ola":
        y_disc = pitch_shift_ola(x, sr, rho, rho_N, N, overlap_factor)
    return y_disc

# -----------------------------------------------------------------------------

def pitch_shift_rosa(x, rho, N):
    """
    This is the windowed no-overlap approach using librosa pitch_shift.
    This produces a lot of artifacts because the algorithm is not meant for
    time variable pitch shifting.
    """
    n_blocks = rho.size - 1
    w = np.hanning(N)
    y_disc = []
    # loop over blocks
    for i in range(n_blocks):
        r = rho[i]
        if np.isnan(r):
            # unvoiced
            shifted = x[i*N:(i+1)*N] * w # unprocessed
        else:
            # convert to semitones
            semitones = 12*np.log2(r)
            shifted = librosa.effects.pitch_shift(x[i*N:(i+1)*N], n_steps=semitones) * w
        y_disc.append(shifted)
    y_disc = np.concatenate(y_disc)
    return y_disc

# -----------------------------------------------------------------------------

def pitch_shift_ola(x, sr, rho, rho_N, N, overlap_factor):
    """
    Time varying pitch shifting using TSM by OLA and resampling.
    x...input signal to be pitch shifted
    sr...sample rate of x
    rho...time varying pitch-shifting factor
    rho_N...window size of the validity of the pitch-shifting factor
    N...analysis window size of pitch shifting
    overlap_factor...factor of window overlap for OLA
    """
    # format pitch-shifting factor to processing parameters
    rho_formated = np.repeat(rho, overlap_factor)
    
    y = np.zeros(N)
    w = np.hanning(N)
    Sa = int(N/overlap_factor) # analysis hop size
    n_blocks = rho_formated.size - (overlap_factor + 1) # skip last blocks
    for i in range(n_blocks):
        # pitch-shifting factor
        r = rho_formated[i]
        # time-scaling factor
        alpha = 1 / r 

        block = x[i*Sa : i*Sa+N] * w
        
        if not np.isnan(r): # voiced
            resampled_block = librosa.resample(block, sr, sr*alpha)
        else: # unvoiced (no pitch shifting)
            alpha = 1
            resampled_block = block
        
        Ss = int(alpha * Sa)
        head = y[:-(N-Ss)]
        overlap = y[-(N-Ss):] + resampled_block[:N-Ss]
        tail = resampled_block[N-Ss:]
        y = np.concatenate((head, overlap, tail))
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
    
    #x, sr = librosa.load('../../samples/Toms_diner.wav')
    #x, sr = librosa.load('../../samples/sweep_20Hz_20kHz_10s.wav')
    
    # pos = 5
    # dur = 10
    # x, sr = librosa.load("../../samples/ave-maria.wav", offset=pos, duration=dur)
    
    dur = 10
    fmin = 500
    fmax = 4*500
    sr = 44100
    x = librosa.chirp(fmin, fmax, sr=sr, duration=dur)
    
    
    y = reshift(x, sr, scale='w')
    
    
    _my_plot(x, sr, "original signal")
    
    _my_plot(y, sr, "processed signal")
    
    sf.write("test.wav", y, sr)


if __name__ == "__main__":
    _test()