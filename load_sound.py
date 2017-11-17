import scipy.io.wavfile as wv


first, second = wv.read('./elisa/Downloads/disco_dancing.wav')
print(first, second)
