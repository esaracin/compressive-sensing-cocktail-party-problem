import scipy.io.wavfile as wv

# Simple test script to see how best to load .wav files as vectors.

first, second = wv.read('./elisa/Downloads/disco_dancing.wav')
print(first, second)
