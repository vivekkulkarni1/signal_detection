import numpy as np
import sys
import matplotlib.pyplot as plt

sn1 = float(sys.argv[1])

snr = sn1
def kalman(A,B,C,sigma,a,R):
    xhat = np.zeros([2,len(a)])
    xhat_prior = np.zeros([2,len(a)])
    xsmoothened = np.zeros([2,len(a)])
    P_prior = []
    Ppost = []
    xhat[:,0] = np.ones(2)
    P0 = np.eye(2)
    Ppost.append(P0)
    for i in range(len(a)-1):        
        xhat_prior[:,i] = A @ xhat[:,i]
        Ptemp = A @ P0 @ (A.T) + B @ B.T * sigma
        Ktemp = C @ Ptemp @ (C.T) + R
        i_Ktemp = np.linalg.inv(Ktemp)
        KGain = Ptemp @ C.T @ i_Ktemp
        ptr = C @ xhat_prior[:,i]
        xhat[:,i+1] = xhat_prior[:,i] + KGain @ (a[i]- ptr )
        P_prior.append(Ptemp)
        P0 = (np.eye(2) - KGain @ C) @ Ptemp
        Ppost.append(P0)

    xsmoothened[:,999] = xhat[:,999]
    for i in range(len(a)-2,0,-1):
        P0 = Ppost[i]
        Ptemp = P_prior[i-1]
        KK = P0 @ A.T @ np.linalg.inv(Ptemp)
        xsmoothened[:,i] = xhat[:,i]+KK @ (xsmoothened[:,i+1] - xhat_prior[:,i])
    return xsmoothened,xhat

fs = 1000
t = np.arange(0,1000)
t = t/fs

f0 = 65
omega = 2 * np.pi * f0
t0 = t[575]
#waveform
h = 50
a0 = 10
a = a0 * np.exp(-h*(t-t0))*np.sin(omega*(t-t0)) * (t>t0)

ar1 = 1.7326
ar2 = -0.8927
ma1 = -0.134

A = np.array([[-ar1, 1],[-ar2,0]])
B = np.array([[1],[ma1]])
C = np.array([[1,0]])
R = np.var(a)/ snr

data_noisy = a + np.random.normal(0,R,len(t))
sigma = 0.1414
xsmooth,x = kalman(A,B,C,sigma,data_noisy,R)

print((np.std(data_noisy)/R)**2)

plt.figure()
ax = plt.subplot(2,2,1)
plt.plot(a)
ax.set_title("Original simulated signal")
ax = plt.subplot(2,2,2)
plt.plot(data_noisy)
ST = 'Noisy signal with snr ' + str(snr)
ax.set_title(ST)
ax = plt.subplot(2,2,4)
plt.plot(xsmooth[0,:])
ax.set_title('Kalman Smoothened Signal')
ax = plt.subplot(2,2,3)
plt.plot(x[0,:])
ax.set_title('Kalman filtered signal')
plt.show()

plt.figure()
plt.subplot(211)
plt.plot(x[0,:]-data_noisy)
plt.subplot(212)
plt.plot(x[0,:])
plt.plot(data_noisy)
plt.show()