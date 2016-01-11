import sys
import math
import numpy as np
from scipy.fftpack import fft
sys.path.append('/Users/cameronstewart/Code/AudioSignalProcessing/workspace')
import audioSignalUtils as asu


def optimalZeropad(x, fs, f):
    """
    Inputs:
        x (numpy array) = input signal of length M
        fs (float) = sampling frequency in Hz
        f (float) = frequency of the sinusoid in Hz
    Output:
        The function should return
        mX (numpy array) = The positive half of the DFT spectrum of the N point DFT after zero-padding 
                        x appropriately (zero-padding length to be computed). mX is (N/2)+1 samples long
    """
    
    # Get number of samples per period of the input
    spp_x = asu.getSamplesPerPeriod(f, fs)

    M = len(x)

    if(M%spp_x != 0):
        # How many periods are there
        # scrape off the remainder and add a solid period.

        full_samples = math.floor(M/spp_x)
        W = int((full_samples + 1) * spp_x)
    else:
        W = M

    hM1 = int(math.floor((M+1)/2))
    hM2 = int(math.floor(M/2))

    fftbuffer = np.zeros(W)
    fftbuffer[:hM1] = x[hM2:]
    fftbuffer[W-hM2:] = x[:hM2]

    X = fft(fftbuffer)
    mX = 20*np.log10(abs(X))
    mX = mX[:(W/2)+1]
    return mX
