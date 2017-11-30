# Eli Saracino
# esaracin@bu.edu
# U55135975
#
#
# Uses k-means clustering on the measurement signal, X, to generate an
# appropriate mixing matrix with which to reconstruct our distinct signal
# sources.
#

import sys
import numpy as np
import scipy.signal as sig
import scipy.io.wavfile as wv
from sklearn.cluster import KMeans

def normalize(v):
    '''Given a vector v, returns the normalized form of that vector.'''
    norm = np.linalg.norm(v)    
    if(norm == 0):
        return v
    
    return v / norm

def main(argv):
    
    if(len(argv) != 1):
        print('Argument error:\n python soundFile.wav')
        sys.exit(2)

    # Read in and apply STFT to our audio signal, x.
    _, x = wv.read(argv[0])
    freqs, segTimes, stft = sig.stft(x)
    

    # Normalize the time-frequency representation of x
    normalized = np.apply_along_axis(normalize, 0, stft)


if __name__ == '__main__':
    main(sys.argv[1:])
