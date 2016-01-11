from scipy.fftpack import fft
import numpy as np
from fractions import gcd
import sys, os, math
sys.path.append('/Users/cameronstewart/Code/AudioSignalProcessing/workspace')
import audioSignalUtils as asu


def minimizeEnergySpreadDFT(x, fs, f1, f2):
    """
    Inputs:
        x (numpy array) = input signal 
        fs (float) = sampling frequency in Hz
        f1 (float) = frequency of the first sinusoid component in Hz
        f2 (float) = frequency of the second sinusoid component in Hz
    Output:
        The function should return 
        mX (numpy array) = The positive half of the DFT spectrum (in dB)
        of the M sample segment of x. 
        mX is (M/2)+1 samples long (M is to be computed)
    """

    px = asu.getPeriod(f1)
    py = asu.getPeriod(f2)
    M = asu.LCM((fs*px),(fs*py))

    hM1 = int(math.floor((M+1)/2))
    hM2 = int(math.floor(M/2))

    x1 = x[:M]

    fftbuffer = np.zeros(M)
    fftbuffer[:hM1] = x1[hM2:]
    fftbuffer[hM2:] = x1[:hM2]

    X = fft(fftbuffer)
    mX = 20*np.log10(abs(X))
    mX = mX[:hM2+1]
    return mX
