import numpy as np
import matplotlib.pyplot as plt 

a = np.load('aic_anmo_53.npy')
b = np.load('aic_majo_77.npy')
c = np.load('aic_tsum_65.npy')
plt.figure()
plt.subplot(211)
plt.plot(a)
plt.subplot(212)
plt.plot(np.diff(a))
plt.show()

plt.figure()
plt.subplot(211)
plt.plot(b[:-200])
plt.subplot(212)
plt.plot(np.diff(b[:-200]))
plt.show()

plt.figure()
plt.subplot(211)
plt.plot(c)
plt.subplot(212)
plt.plot(np.diff(c))
plt.show()
