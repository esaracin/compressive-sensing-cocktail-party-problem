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
import glob
import numpy as np
import scipy
import scipy.sparse as sp
import scipy.signal as sig
import scipy.io.wavfile as wv
from sklearn.cluster import KMeans

def normalize(v):
    '''Given a vector v, returns the normalized form of that vector.'''
    norm = np.linalg.norm(v)    
    if(norm == 0):
        return v
    
    return v / norm

def learn_dictionary(N):
    '''Learns the Sparsifying Dictionary.''' 
    I = np.eye(N)
    D = scipy.fftpack.dct(I)
    return D

    #path = '../data/training/*'
    #for filename in glob.iglob(path):
    #    sampling_freq, x = wv.read(filename)

    #return D


def main(argv):    
    if(len(argv) != 2):
        print('Argument error:\n python soundFile.wav numSources')
        sys.exit(2)

    # Read in and apply STFT to our audio signal, x.
    samplingFreq, x = wv.read(argv[0])
    freqs, segTimes, stft = sig.stft(x, fs=samplingFreq)

    # Normalize the time-frequency representation of x
    normalized = np.apply_along_axis(normalize, 0, stft)
    normalized = normalized.flatten('F').T
    
    # Run k-means on column vectors with k = numSources
    kmeans = KMeans(init='k-means++', n_clusters=int(argv[1]))
    kmeans.fit_predict(normalized.reshape(-1, 1))
    A = kmeans.cluster_centers_.T

    # Use A to construct our mixing matrix, M
    l = 500
    for i in range(70):
        first_diag = np.zeros(l)
        second_diag = np.zeros(l)
    
        first_diag.fill(A[0][0])
        second_diag.fill(A[0][1])

        first = sp.diags(first_diag)
        second = sp.diags(second_diag)

        # Combine our matrices to construct M
        M = sp.hstack((first, second))
        D = learn_dictionary(M.shape[1]) 

        MD = M @ D


if __name__ == '__main__':
    main(sys.argv[1:])
