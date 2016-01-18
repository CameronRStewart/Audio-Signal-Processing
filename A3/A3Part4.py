import sys
sys.path.append('../../software/models/')
sys.path.append('/Users/cameronstewart/Code/AudioSignalProcessing/workspace')
from dftModel import dftAnal, dftSynth
from scipy.signal import get_window
import matplotlib.pyplot as plt
import numpy as np
import audioSignalUtils as asu

def suppressFreqDFTmodel(x, fs, N):
    """
    Inputs:
        x (numpy array) = input signal of length M (odd)
        fs (float) = sampling frequency (Hz)
        N (positive integer) = FFT size
    Outputs:
        The function should return a tuple (y, yfilt)
        y (numpy array) = Output of the dftSynth() without filtering (M samples long)
        yfilt (numpy array) = Output of the dftSynth() with filtering (M samples long)
    The first few lines of the code have been written for you, do not modify it. 
    """

    attenuated_freq = 70.0
    attenuation_value = -120 # DB

    M = len(x)
    w = get_window('hamming', M)
    outputScaleFactor = sum(w)
    
    mX1, pX1 = dftAnal(x, w, N)

    bin_index = asu.calcBinValue(fs, N, attenuated_freq)

    mX2 = mX1.copy()

    mX2[:(bin_index + 1)] = attenuation_value

    """
    compute the inverse dft of the spectrum
    y = DFT.dftSynth(mX, pX, w.size)*sum(w)
    """

    y = dftSynth(mX1, pX1, w.size)*outputScaleFactor
    yfilt = dftSynth(mX2, pX1, w.size)*outputScaleFactor

    return (y, yfilt)


