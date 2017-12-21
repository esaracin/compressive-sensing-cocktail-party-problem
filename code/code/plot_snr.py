import sys
import logging
import numpy as np
import scipy.io.wavfile as wv
import matplotlib.pyplot as plt
from scipy.stats import signaltonoise

def main(argv):
    if(len(argv) < 1 or ('-h' in argv)):
        print('Argument error:\n python soundFile1.wav soundFile2.wav ... soundFileN.wav')
        sys.exit(2)

    soundFids = []
    for fid in argv:
        if(fid == '-v' or fid == '--verbose'):
            continue
        soundFids.append(fid)

    logger.info('Booting up plot')
    for fid in soundFids:
        samplingFreq, x = wv.read(soundFid)

        snr = signaltonoise(x)


if __name__ == '__main__':
    if(('-v' in sys.argv) or ('--verbose' in sys.argv)):
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    main(sys.argv[1:])
