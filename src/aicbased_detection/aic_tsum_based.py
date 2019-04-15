import numpy as np
X = np.array([0.2918,
    -0.3295,
    0.8546,
    -0.5962,
    1.1042,
    -0.5381,
    1.1731,
    -1.2153,
    1.0419,
    -1.5694,
    1.1610,
    -1.4891,
    1.5439,
    -1.3711,
    1.6795,
    -1.5151,
    1.5884,
    -1.6542,
    1.4734,
    -1.6288,
    1.4955,
    -1.4850,
    1.4004,
    -1.3038,
    1.3394,
    -1.2050,
    1.0921,
    -1.0800,
    0.9654,
    -0.8946,
    0.8220,
    -0.7448,
    0.6380,
    -0.5272,
    0.5125,
    -0.4434,
    0.3662,
    -0.3331,
    0.2555,
    -0.1915,
    0.1676,
    -0.1586,
    0.1043,
    -0.1036,
    0.0759,
    -0.0677])

X = np.array([X]).T
import pandas as pd
import matplotlib.pyplot as plt
signal = pd.read_csv('IUTSUM00BHZ.csv')
signal= signal.values
signal = np.diff(signal[:,0])
vec = np.arange(46,len(signal))
err=[]
for i in vec:
    y = signal[i]
    v = np.fliplr(np.array([signal[i-46:i]]))
    e = y - v@X
    e = e[0]
    err.append(e)
parameters = 2 *     47 
aic = []
kk = np.arange(len(err))+46
kk1 = np.fliplr(kk.reshape(1,-1))
kk1 = kk1[0,:]
aic = np.zeros(len(kk))
'''
for i in kk1:
    k = np.asarray(err[0:i])    
    kl = i*(np.log(2*np.pi)+1+np.log(k.T @ k/i))
    aic[i-46] = kl + parameters
    print(i)
A = np.fliplr(aic.reshape(1,-1))
A = A[0,:]
np.save('tsum_aic.npy',A)
print('done')
'''

A1 = np.load('tsum_aic.npy')
'''
plt.figure()
plt.plot(kk1,A1)
plt.show()
plt.figure()
plt.plot(kk1[0:-1],np.diff(A1))
plt.show()
'''
np.savetxt('tsum_aic.csv',A1)
np.savetxt('differenced_tsum.csv',signal)