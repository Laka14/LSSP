
import numpy as np
from scipy import constants
import matplotlib.pyplot as plt

from scipy.io import wavfile
import wave
ograte, data = wavfile.read("dwsample1-wav.wav")
def changerate(originaldata, originalrate, targetrate):
  wavfile.write('sample.wav', targetrate, originaldata)
  t, newdata = wavfile.read('sample.wav')
  return newdata, targetrate
def quantize(input_data, nbits):
    quantization_step = 2 ** (16 - nbits)
    quantized_data = np.round(input_data / quantization_step) * quantization_step
    #wavfile.write(output_file, ograte, quantized_data.astype(np.int16))
    return quantized_data
samplingrates=[8000,16000,22050,41000]
quantizationbits=[4,8,16,32]

for rate in samplingrates:
  for bits in quantizationbits:
    #change sampling rate
    newdata, newrate = changerate(data,ograte,rate)

    #quantize the data
    quantizeddata=quantize(newdata, bits)

    #save the files
    outfilename=outfilename=str(rate)+str(bits)+'.wav'
    wavfile.write(outfilename, newrate, quantizeddata.astype(np.int16))
    plt.figure(rate+bits)
    x=np.linspace(-rate/2,rate/2,len(quantizeddata))
    plt.plot(x,np.fft.fftshift(np.fft.fft2(quantizeddata)))
    plt.show()
    