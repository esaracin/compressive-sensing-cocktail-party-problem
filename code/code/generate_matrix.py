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


def buildMatrix(A, t):
    D = np.zeros(shape=(1,2))
    for row in range(len(A)):
        for col in range(len(A[0])):
            curr = A[row][col]

            to_add = np.zeros((t,t))
            np.fill_diagonal(to_add, curr)
            D[row][col] = to_add


def main(argv):    
    if(len(argv) != 2):
        print('Argument error:\n python soundFile.wav numSources')
        sys.exit(2)

    # Read in and apply STFT to our audio signal, x.
    samplingFreq, x = wv.read(argv[0])
    freqs, segTimes, stft = sig.stft(x, fs=samplingFreq)
    print(freqs.shape)
    print(segTimes.shape)

    print(stft.shape)

    # Normalize the time-frequency representation of x
    normalized = np.apply_along_axis(normalize, 0, stft)

    # Run k-means on column vectors with k = numSources
    kmeans = KMeans(init='k-means++', n_clusters=int(argv[1]))
    kmeans.fit_predict(normalized)

    A = kmeans.cluster_centers_.T
    print(A.shape)

    A_ = buildMatrix(A, A.shape[0])


if __name__ == '__main__':
    main(sys.argv[1:])
