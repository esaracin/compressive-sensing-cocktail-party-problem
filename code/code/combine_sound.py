import numpy as np
import scipy.io.wavfile as wv

import sys

def writeWave(rate, signal, destinationFile):
    wv.write(destinationFile, rate, signal)

    return 1

def combineAudio(fileName1, fileName2):
    rate1, audioSig1 = wv.read(fileName1)
    print('File 1: %f' % (rate1))
    rate2, audioSig2 = wv.read(fileName2)
    print('File 2: %f' % (rate2))

    minLength = min(audioSig1.shape[0], audioSig2.shape[0])
    print('Minimum length %d' % (minLength))

    return rate2, np.add(audioSig1[0:minLength], audioSig2[0:minLength])/2

def main(argv):
    if len(argv) < 3:
        print("Input argument error:")
        print("python combine_sound.py sound1 sound2 savedSound")
        sys.exit(2)

    # Bunch of inits
    soundFile1 = argv[0]
    soundFile2 = argv[1]
    destinationFile = argv[2]

    rate, sig = combineAudio(soundFile1, soundFile2)

    if(writeWave(rate, sig, destinationFile)):
        print('Saved %s' % (destinationFile))

    return 0

if __name__=="__main__":
    main(sys.argv[1:])
