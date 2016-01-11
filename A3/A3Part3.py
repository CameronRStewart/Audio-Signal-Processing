import sys
import numpy as np
sys.path.append('/Users/cameronstewart/Code/AudioSignalProcessing/workspace')
import audioSignalUtils as asu


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
    """
    real:
    |F(k)| = |F(N-k)|
    even:
    imag(F(k) = 0)
    """

    precision_tolerance = 6

    N = len(x)
    dftbuffer = asu.zeroPhaseWindowBuffer(x, N)
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
    even = asu.allZero(rounded_X)

    isRealEven = real and even

    return (isRealEven, dftbuffer, X)


