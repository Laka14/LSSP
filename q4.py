import numpy as np
import matplotlib.pyplot as plt

fs=100
t=np.arange(-0.5,0.51,1/fs)

f1=np.sin(2*np.pi*5*t)
f2=np.sin(2*np.pi*10*t)
f=f1+f2
plt.plot(t,f)
plt.show()

#q4 b

s=np.arange(-fs/2,fs/2+1,1)
ffreq=np.fft.fftshift((np.fft.fft(f)/np.sqrt(len(f))))
plt.plot(s,np.absolute(ffreq))
plt.plot(s,(ffreq))
plt.show()

#Q4 C 
def w(x):
  if np.abs(x)<6:
    return 1
  else:
    return 0

filt=np.zeros((len(s)))
ffilt=np.zeros((len(s)))

for i in range(0,fs+1):
  filt[i]=w(i-(fs/2))


ffilt=np.multiply(ffreq,filt)

plt.plot(s,filt)
#
plt.plot(s,ffilt)
plt.show()

#Q4 D
fre=np.fft.ifft(np.fft.ifftshift((ffilt)))

plt.plot(t,fre)
plt.show()
