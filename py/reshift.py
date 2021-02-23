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


from Filter import Filterbank


class Reshifter:
    """
    A pitch shifting and pitch discretization effect class.
    It uses pYIN to analyze the pitch of the signal and the 'Rollers'
    algorithm for pitch-shifting.
    """

    def __init__(self, sr=44100, a_N=1024, a_hop=512, a_fmin=60, a_fmax=2000,
                 a_res=0.1, filt_order=2, filt_num=100):
        """
        Constructor
        sr: sample rate of the signal to process
        a_N: pitch analysis block size
        a_hop: pitch analysis hop size
        a_fmin: pitch analysis minimum tracked frequency
        a_fmax: pitch analysis maximum tracked frequency
        filt_order: used filter order in filter bank for pitch shifting
        filt_num: used number of bands in filter bank for pitch shifting
        """
        self.sr = sr
        librosa['sr'] = sr
        
        self.a_N = a_N
        librosa['n_fft'] = a_N

        self.a_hop = a_hop
        librosa['hop_length'] = a_hop
        
        self.a_fmin = a_fmin
        self.a_fmax = a_fmax
        self.a_res = a_res
    
        self.filt_order = filt_order
        self.filt_num = filt_num
        self.filt_bank = Filterbank(filt_num, filt_order, sr)


    def freq_scale(self, f_0, scale='chromatic', tune=440):
        """
        The discrete frequency scaling function for the desired pitch.
        f_0: input frequency
        tune: tuning frequency
        
        Returns the discrete tuned frequencies.
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


    def discretize(self, x, scale='chromatic'):
        """
        Discretize the pitch of the signal 'x' to the given 'scale'.
        x: signal to be pitch discretized
        scale: the scale, the pitch should be discretized to
        
        scale choices:
            * 'chromatic', 'chrom', 'c'
            * 'wholetone', 'whole', 'w'
            * 'thirds', 'third', 't'
            * 'tritones', 'tritone', 'tri'
        
        Returns the pitch discretized signal according to the 'scale'.
        """
        # zero padding because librosa pYIN does this internally
        x = np.concatenate((x, np.zeros(self.a_hop - (x.size % self.a_hop) - 1)))
        
        f_0, voiced_flag, voiced_probs = librosa.pyin(x, fmin=self.a_fmin,
                                                      fmax=self.a_fmax,
                                                      sr=self.sr,
                                                      frame_length=self.a_N,
                                                      resolution=self.a_res)
        
        # append one zero for correct length with f_0
        x = np.concatenate((x, np.zeros(1)))
        
        f_out = self.freq_scale(f_0, scale)
        
        # pitch shifting factor rho
        psr = f_out / f_0
        
        y = self.pitch_shift(x, psr)
        return (y / np.abs(np.max(y))) * np.abs(np.max(x)) # normalization


    def pitch_shift(self, x, psr):
        """
        Time variant 'Rollers' pitch-shifting algorithm.
        x: input signal
        fs: sampling rate
        psr: list of pitch-shifting ratios for x
        N: pitch analysis block size of the pitch tracking algorithm
        order: filter order of the used filter bank
        n: number of filters in the filter bank
        
        Returns the pitch shifted signal according to 'psr'.
        """
        
        # fs, N, order=2, n=100
        fs = self.sr
        
        # format pitch shifting ratio
        # unvoiced parts cause no pitch shift
        psr = np.nan_to_num(psr, nan=1)
        
        
        ### UPSAMPLING from control rate to audio rate ###
        L = self.a_hop
        
        ## Do the upsampling of the pitch shifting ratio ##
        
        psr_up = np.repeat(psr, L) # this on its own is most likely sufficient
        # optional smoothing
        # control signal smoothing filter length
        #l = 32
        #h_smooth = signal.windows.triang(l) / (l-1)
        #psr_up = signal.convolve(psr_up, h_smooth, mode="same")
        
        ## OR ##
        
        ## Do upsampling by inserting zeros and linear interpolation ##
        
        #psr_up = np.zeros(x.size)
        #psr_up[::L] = psr
        #h_interp = signal.windows.triang(2 * L - 1)
        #psr_up = signal.convolve(psr_up, h_interp, mode="same")
        
        ### UPSAMPLING ###
        
        
        # divide input into frequency bands
        x_filtered = self.filt_bank.filt(x)
        
        # frequency shifting in every band
        out_signals = []
        t = np.linspace(0, x_filtered[0].size/fs, x_filtered[0].size)
        for i in range(len(x_filtered)):
            # calculate time variant carrier frequencies for every block
            fc = self.filt_bank.fcs[i]
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
    #x, sr = librosa.load("../../samples/ave-maria.wav", offset=pos, duration=dur)
    
    #y = reshift(x, sr, scale='w')
    
    ## new OOP variant
    reshifter = Reshifter(sr=sr, a_fmin=100, a_fmax=800, a_N=512, a_hop=512,
                          a_res=0.15)
    y = reshifter.discretize(x, scale='w')
    
    _my_plot(x, sr, "original signal")
    
    _my_plot(y, sr, "processed signal")
    
    sf.write("test.wav", y, sr)


if __name__ == "__main__":
    _test()
