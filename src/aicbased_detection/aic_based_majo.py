import numpy as np
X = np.array([1.2244,-1.5342,2.4906,-2.7203,3.0036,-3.9002,4.3475,-4.4766,5.1284,-5.2991,  
 5.4055,-5.5507,5.6493,-5.4631,5.4076,-5.1761,4.8783,-4.6325,4.2954,-3.9023,  
 3.5214,-3.1056,2.6938,-2.3135,1.8954,-1.5905,1.2463,-0.9514,0.6877,-0.5185,  
 0.3015,-0.1552,0.0334,0.0428,-0.1271,0.1497,-0.1592,0.1878,-0.1729,0.1260,   
-0.1064,0.0925,-0.0442,0.0265,-0.0383,0.0131]) 

X = np.array([X]).T
import pandas as pd
import matplotlib.pyplot as plt
signal = pd.read_csv('IUMAJO10BHZ.csv')
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
parameters = 2 *47 
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
np.save('majo_aic.npy',A)
print('done')
'''
A1 = np.load('majo_aic.npy')

plt.figure()
plt.plot(signal)
plt.figure()
plt.plot(kk1,A1)
plt.show()
plt.figure()
plt.plot(kk1[0:-1],np.diff(A1))
plt.show()
np.savetxt('majo_aic.csv',A1)
np.savetxt('differenced_majo.csv',signal)