import numpy as np
from scipy.fftpack import fft, fftshift
from fractions import gcd
from operator import add
import math

def genComplexSine(k, N):
    """
    Inputs:
        k (integer) = frequency index of the complex sinusoid of the DFT
        N (integer) = length of complex sinusoid in samples
    Output:
        The function should return a numpy array
        cSine (numpy array) = The generated complex sinusoid (length N)
    """
    cSine = np.array([])
    # Produce array of time indecies
    n = np.arange(0, N)

    s = np.exp(1j * 2 * np.pi * k * n/N)
    cSine = np.append(cSine, np.conjugate(s))
    return cSine

def genSine(A, f, phi, fs, t):
    """
    Inputs:
        A (float) =  amplitude of the sinusoid
        f (float) = frequency of the sinusoid in Hz
        phi (float) = initial phase of the sinusoid in radians
        fs (float) = sampling frequency of the sinusoid in Hz
        t (float) =  duration of the sinusoid (is second)
    Output:
        The function should return a numpy array
        x (numpy array) = The generated sinusoid (use np.cos())
    """

    x = np.array([])

    # Generate individual sample times (array representing time index)    
    Fn = np.arange(0, t, 1.0/fs)
    x = A * np.cos (2 * np.pi * f * Fn + phi)
    return x

def genSineByPeriod(f, fs, periods):
    """
    Inputs:
        f (integer)       = frequency of sine wave
        fs (integer)       = sample rate in hertz
        periods (integer) = Number of periods to generate.
    Output:
        x (numpy array)  = sinusoid of specified period.
    """
    x = np.array([])
    period = getPeriod(f)
    length = period*periods
    x = genSine(1, f, 0, fs, length)
    return x

def genSineBySampleLength(f, fs, samples):
    """
    Inputs:
        f (integer)         = frequency of sine wave
        fs (integer)         = sample rate in hertz
        samples (integer)   = number of samples to generate
    Output:
        x (numpy array)     = sinusoid array with specified number of samples    
    """
    x = np.array([])
    t = samples / float(fs)
    x = genSine(1, f, 0, fs, t)
    return x

def genSineByTime(f, fs, time):
    """
    Inputs:
        f (integer)      = frequency of sine wave
        fs (integer)      = sample rate in hertz
        time (float)   = time length of sample
    Output:
        x (numpy array)  = sinusoid array of specified duration    
    """
    x = np.array([])
    x = genSine(1, f, 0, fs, time)
    return x

def genMultiFrequencySine(freq_array, fs, samples):
    """
    Inputs:
        freq_array (numpy array)   = array with each frequency withi which to compose this wave.
        fs (integer)                = sample rate in hertz (applies to all components)
        samples (integer)          = number of samples to generate
    Output:
        x (numpy array)  = sinusoid array of specified characteristics    
    """
    x = np.zeros(samples)
    for freq in freq_array:
        x = x + genSineBySampleLength(freq, fs, samples)

    return x


def DFT(x):
    """
    Input:
        x (numpy array) = input sequence of length N
    Output:
        The function should return a numpy array of length N
        X (numpy array) = The N point DFT of the input sequence x
    """
    N = len(x)
    kv = np.arange(0, N)
    X = np.array([])
    for k in kv:
        s = np.exp(1j * 2 * np.pi * k/N * kv)
        X = np.append(X, sum(x*np.conjugate(s)))

    return X

def IDFT(X):
    """
    Input:
        X (numpy array) = frequency spectrum (length N)
    Output:
        The function should return a numpy array of length N 
        x (numpy array) = The N point IDFT of the frequency spectrum X
    """
    N = len(X)
    x = np.array([])
    nv = np.arange(0, N)
    for n in nv:
        s = np.exp(1j * 2 * np.pi * n/N * nv)
        x = np.append(x, 1.0/N * sum(X*s))
    return x

def genMagSpec(x):
    """
    Input:
        x (numpy array) = input sequence of length N
    Output:
        The function should return a numpy array
        magX (numpy array) = The magnitude spectrum of the input sequence x
            (length N)
    """
    magX = np.absolute(DFT(x))
    return magX

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

def testRealEven(x):
    """
    Inputs:
        x (numpy array)= input signal of length M (M is odd)
    Output:
        The function should return a tuple (isRealEven, dftbuffer, X)
        isRealEven (boolean) = True if the input x is real and even, and False otherwise
        dftbuffer (numpy array, possibly complex) = The M point zero phase windowed version of x 
        X (numpy array, possibly complex) = The M point DFT of dftbuffer 
    """

    precision_tolerance = 6

    N = len(x)
    dftbuffer = zeroPhaseWindowBuffer(x, N)
    X = asu.DFT(dftbuffer)
    magX = np.absolute(X)



    real = True

    # Test Real
    for num in range(1, ((N/2) + 1)):
        if(round(magX[num], precision_tolerance) != round(magX[(N)-num], precision_tolerance)):
            real = False
            break

    # Test Even
    rounded_X = [round(elem, precision_tolerance) for elem in np.imag(X)]
    even = allZero(rounded_X)

    isRealEven = real and even

    return (isRealEven, dftbuffer, X)


def LCM(x,y):
    return (x*y)/gcd(x,y)

def getPeriod(frequency):
    return (1.0/frequency)

def getSamplesPerPeriod(frequency, sample_rate):
    period = getPeriod(frequency)
    return  period * sample_rate

def zeroPhaseWindowBuffer(x, dft_out_size):
    M = len(x)

    hM1 = math.floor((M+1)/2)
    hM2 = math.floor(M/2)
    dftbuffer = np.zeros(dft_out_size)
    dftbuffer[:hM1] = x[hM2:]
    dftbuffer[-hM2:] = x[:hM2]

    return dftbuffer

def allZero(arr):
    for element in arr:
        if element != 0:
            return False
    return True

def calcBinValue(fs, N, f):
    """
    Inputs:
        fs (float)  = sampling frequency (Hz)
        N (integer) = Size of fft/dft
        f (float)   = frequency at which to find bin
    Outputs:
        k (integer) = frequency bin
    """
    k = math.ceil(f*N / fs)
    return int(k)

def convertDecibels(x):
    """
    Input:
        x (numpy array) = sinusoid array (DFT/FFT) to convert to decibels
    Output:
        y (numpy array) = sinusoid array w/ amplitude converted to decibels
    """
    y = 20*np.log10(abs(x))
    return y


