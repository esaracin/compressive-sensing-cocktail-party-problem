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
import math
import logging
import numpy as np
import scipy
import scipy.sparse as sp
import scipy.signal as sig
import scipy.io.wavfile as wv
import sklearn
from sklearn.decomposition import FastICA, PCA
from sklearn.cluster import KMeans

def normalize(v):
    '''Given a vector v, returns the normalized form of that vector.'''
    norm = np.linalg.norm(v)    
    if(norm == 0):
        return v
    
    return v / norm

def estimateMixtureCoef(signal, sampFreq, numSources):
    freqs, segTimes, stft = sig.stft(signal, fs=sampFreq)
    
    # Normalize the time-frequency representation of x
    normalized = np.apply_along_axis(normalize, 0, stft)
    normalized = normalized.flatten('F').T
    
    # Run k-means on column vectors with k = numSources
    kmeans = KMeans(init='k-means++', n_clusters=numSources)
    kmeans.fit_predict(normalized.reshape(-1, 1))
    A = kmeans.cluster_centers_.T

    return A

def constructMixtureMatrix(a, window):
    first_diag = np.zeros(window)
    second_diag = np.zeros(window)

    first_diag.fill(a[0][0])
    second_diag.fill(a[0][1])

    first = sp.diags(first_diag)
    second = sp.diags(second_diag)

    M = sp.hstack((first, second))
    
    return M

def learn_dictionary(N):
    '''Learns the Sparsifying Dictionary.''' 
    I = np.eye(N)
    D = scipy.fftpack.dct(I)
    return D


def ICA_analysis(X, numSources, samplingFreq):
    '''Performs ICA to separate the mixed signal, X.''' 

    X = X.T
    X = np.nan_to_num(X)

    ica = FastICA(n_components=numSources)
    S_ = ica.fit_transform(X)  # Reconstruct signals

    print(S_.shape)
    print(S_)
    
    return S_


def main(argv):    
    if(len(argv) < 3):
        print('Argument error:\n python soundFile.wav numSources windowSize')
        sys.exit(2)

    soundFid = argv[0]
    numSources = int(argv[1])
    window = int(argv[2])

    # Read in and apply STFT to our audio signal, x.
    samplingFreq, x = wv.read(soundFid)

    # Simulate second measurement
    x_2 = (1.3 * x).T 
    X = np.column_stack((x.T, x_2))
    
    # Compute ICA to compare to results
    #ICA_analysis(X, numSources, samplingFreq)
    
    # Use a windowing procedure so not as to run out of memory
    x = x.reshape(-1, 1)

    A = estimateMixtureCoef(x, samplingFreq, numSources)
    M = constructMixtureMatrix(A, window)
    D = learn_dictionary(M.shape[1]) 
    MD = M@D
    
    idx = [x*window for x in range(math.floor(len(x)/window))]
    recoveredSources = [np.array([]) for sources in range(numSources)]

    logger.info("Beginning recovery for window:")
    for i in range(1,len(idx)):
        logger.debug("\t%d : %d" % (idx[i-1], idx[i]))
        omp = sklearn.linear_model.OrthogonalMatchingPursuit()
        omp.fit(MD, x[idx[i-1]:idx[i]])
        coef = D@omp.coef_.reshape(-1,1)
        sources = [coef[(source*window):(source+1)*window] for source in range(numSources)]
        for sourceIdx in range(numSources):
            recoveredSources[sourceIdx] = np.concatenate((recoveredSources[sourceIdx], sources[sourceIdx].flatten()))

    for sourceIdx in range(numSources):
        logger.info("Writing output_%d.wav" % (sourceIdx))
        logger.info("\tShape: {}".format(recoveredSources[sourceIdx].shape))
        wv.write('output_'+str(sourceIdx)+'.wav', samplingFreq, recoveredSources[sourceIdx])

    return 0

if __name__ == '__main__': 
    if(('-v' in sys.argv) or ('--verbose' in sys.argv)):
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    main(sys.argv[1:])
