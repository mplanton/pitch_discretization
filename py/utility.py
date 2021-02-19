#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility module

Manuel Planton 2021
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.fft import rfft, fft, rfftfreq, fftfreq

def plot_mag_spec(sig, fs, name="", negative=False, f_range=(20, 20000), db_range=(-40, 0), f_log=True):
    """
    Plot the magnitude spectrum of the signal 'sig'.
    sig:    signal to be analyzed
    fs:     sampling rate of sig
    name:   name of the plotted spectrum
    negative: show negative frequencies (default: False)
    f_range: define a frequency range for the spectrum (default: 20Hz to 20kHz)
    db_range: define a magnitude range in dB for the spectrum (default: 0dB to 100dB)
    f_log:  logarithmic or linear frequency axis? (default: True is logarithmic)
    """
    w = signal.hann(sig.size) # window
    if negative == False:
        # just positive frequencies
        freq = rfftfreq(sig.size, 1 / fs)
        mag = 20*np.log10(np.abs((1/sig.size)*rfft(sig*w)))
        if f_log == True:
            plt.semilogx(freq, mag)
        else:
            plt.plot(freq, mag)
        plt.xlim(f_range)
    else:
        # positive and negative frequencies
        freq = fftfreq(sig.size, 1 / fs)
        mag = 20*np.log10(np.abs((1/sig.size)*fft(sig*w)))
        if f_log == True:
            freq = freq[1:] # omit 0Hz
            mag = mag[1:]
        plt.plot(freq, mag)
        plt.xlim((-f_range[1], f_range[1]))
        if f_log == True:
            plt.xscale('symlog')
    plt.ylim(db_range)
    plt.xlabel("f [Hz]")
    plt.ylabel("amplitude [dB]")
    plt.title(name)
    plt.show()
