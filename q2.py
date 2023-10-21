
# Q2 A
import numpy as np
import matplotlib.pyplot as plt

def f(x):
  return (1/a)*(np.exp((-np.pi*(x**2))/a**2))

x = np.linspace(-5,5,500)

arr=np.arange(1,5)

#for a in arr:
 # plt.plot(x,f(x))

#plt.show()

#Q2 F

L=5
B=5
t=1/B
f=1/L

def f(n):
  return (np.exp(-np.pi*((n)**2)))

def sys(B,L):
  nrange=np.arange(-L,L+1/B, 1/B)
  s=np.arange(-B,B+(1/L),1/L)
  y=np.fft.fftshift(np.absolute(np.fft.fft(f(nrange)))/np.sqrt(len(f(nrange))))
  return s,y

s,y = sys(B,L)
plt.plot(s,y)
#plt.show()

#Q2 G
for i in [5,10,20]:
  for j in [5,10,20]:
    s,y = sys(i,j)
    plt.plot(s,y,label=str(i)+'Hz,'+str(j)+'sec')

#plt.legend()
#plt.show()

#Q2 H) A

#B=5
#L=5

B=10
L=10

def w1(x):
  if np.abs(x)<=L/2:
    return (1-(2*np.abs(x)/L))
  else:
    return 0

nrange=np.arange(-L/2,L/2+(1/B),1/B)
s=np.arange(-B/2,B/2+(1/L),(1/L))

filt=np.zeros((len(s)))
ffilt=np.zeros((len(s)))

for i in range(0,B*L+1):
  filt[i]=w1(-L/2+(i/B))

ffilt=np.multiply(filt,f(nrange))

y=np.fft.fftshift(np.absolute(np.fft.fft(ffilt))/np.sqrt(len(ffilt)))

plt.plot(nrange,filt)
plt.plot(nrange,ffilt)
plt.plot(s,y)
plt.show()

#Q2 H) B
B=10
L=10

def w2(x):
  if np.abs(x)<=L/2:
    return (np.sin(2*np.pi*x/L))**2
  else:
    return 0

nrange=np.arange(-L/2,L/2+(1/B),1/B)
s=np.arange(-B/2,B/2+(1/L),(1/L))

filt=np.zeros((len(s)))
ffilt=np.zeros((len(s)))

for i in range(0,B*L+1):
  filt[i]=w2(-B/2+(i/L))

ffilt=np.multiply(filt,f(nrange))

y=np.fft.fftshift(np.absolute(np.fft.fft(ffilt))/np.sqrt(len(ffilt)))

ffilt=np.multiply(y,filt)

plt.plot(s,filt)
plt.plot(s,ffilt)
plt.plot(s,y, "g")

plt.show()