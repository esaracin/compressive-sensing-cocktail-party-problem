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
    if(len(argv) != 2):
        print('Argument error:\n python soundFile.wav numSources')
        sys.exit(2)

    # Read in and apply STFT to our audio signal, x.
    _, x = wv.read(argv[0])
    freqs, segTimes, stft = sig.stft(x)

    print(x.shape[0])

    # Normalize the time-frequency representation of x
    normalized = np.apply_along_axis(normalize, 0, stft)

    # Run k-means on column vectors with k = numSources
    kmeans = KMeans(init='k-means++', n_clusters=int(argv[1]))
    kmeans.fit_predict(normalized.T)
    A = kmeans.cluster_centers_

    print(A)
    print(A.shape)


if __name__ == '__main__':
    main(sys.argv[1:])
