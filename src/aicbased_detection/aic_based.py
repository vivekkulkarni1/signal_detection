import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA

data = pd.read_csv("simulates_noise.csv", skiprows= 0)
X  = data.values

signal = X[:,1]

background_noise = signal[0:1500]

X = np.array([[0.763],[-0.124]])
A = np.eye(2)

vec = np.arange(2,len(signal))
vec1 = vec*0

i=3
err=[]
for i in vec:
    y = signal[i]
    e = y-0.763*signal[i-1]+0.124 * signal[i-2] 
    err.append(e)

parameters = 2 * 3 
aic = []

kk = np.arange(len(signal),1,-1)

for i in kk:
   k = np.asarray(err[0:i])
   kl = i*(np.log(2*np.pi)+1+np.log(np.sum(k**2)/i))
   aic.append(parameters+kl)


k1 = np.arange(2,2501)
print(kk.shape)
print(len(aic))
A = aic[::-1]
A = np.asarray(A)
np.save('simulated_aic.npy',A)
plt.plot(k1,A)
plt.show()

plt.figure()
plt.plot(signal)
plt.show()