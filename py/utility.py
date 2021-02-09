#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


class Filterbank:
    """A  constant Q IIR butterworth filter bank for audio signal processing"""
    
    def __init__(self, n, order, fs, filt_type=""):
        """
        n: number of filters to use
        order: filter order
        fs: used sampling rate
        filt_type: choose 'third octave' or a general logarithmic spaced
                   filterbank
        """
        self.n = n
        self.order = order
        self.fs = fs
        
        if filt_type == "third octave":
            # third-octave filter bank
            if self.n > 28:
                self.n = 28
            
            freq_offset = 2
            k = np.arange(n + 2) - n // 2 - freq_offset

            # center frequencies are defined relative to a bandpass with
            # center frequency at 1kHz
            f_cs = np.power(2, k / 3) * 1000

            f_chs = [] # high cutoff frequencies
            f_cls = [] # low cutoff frequencies
            for k in range(1, f_cs.size-1):
                f_chs.append(np.sqrt(f_cs[k] * f_cs[k+1]))
                f_cls.append(np.sqrt(f_cs[k-1] * f_cs[k]))
        else:
            # general log spaced filter bank for audio range
            # 2**4 = 16Hz
            # 14.3 = 20171Hz
            f = np.logspace(4, 14.3, n*3, base=2)

            # low cutoff frequencies
            f_cls = f[::3]
            # center frequencies
            f_cs = f[1::3]
            # high cutoff frequencies
            f_chs = f[2::3]

        filters = []
        for k in range(f_cs.size-2):
            sos = self.butter_bp(f_cls[k], f_chs[k])
            filters.append(sos)
        
        self.fcs = f_cs[1:-1]
        self.filters = filters
    
    def butter_bp(self, lowcut, highcut):
        """
        Design a butterworth bandpass filter and return the 'sos' filter.
        lowcut: low cutoff frequency
        highcut: high cutoff frequency
        """
        f_nyq = 0.5 * self.fs
        low = lowcut / f_nyq
        high = highcut / f_nyq
        return signal.butter(self.order, [low, high], btype='band', output='sos')
    
    def plot_filters(self):
        """Plot the magnitude spectrum of all filter of the filter bank."""
        for sos in self.filters:
            w, h = signal.sosfreqz(sos, worN=10000)
            plt.semilogx((self.fs * 0.5 / np.pi) * w[1:], 20*np.log10(np.abs(h[1:])))
            plt.ylim((-100, 5))
            plt.xlim((10, 20000))
            plt.ylabel('H [dB]')
            plt.xlabel('f [Hz]')
            plt.title('third-octave filter bank')
        plt.show()
    
    def filt(self, in_sig):
        """Filter the in_sig and return an array of the filtered signals."""
        filtered_signals = []
        for sos in self.filters:
            filtered_signals.append(signal.sosfilt(sos, in_sig))
        return filtered_signals


if __name__ == "__main__":
    # test the Filterbank class
    filtb = Filterbank(28, 4, 44100)
    filtb.plot_filters()
    noise = np.random.normal(0, 1, 10*1024)
    filtered_noise = filtb.filt(noise)
    plot_mag_spec(filtered_noise[10], 44100, db_range=[-100, -30])