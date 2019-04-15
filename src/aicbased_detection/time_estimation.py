import numpy as np
import os
import sys

import pandas as pd 
from pathlib import Path

name1 = str(Path(sys.argv[1]))
name2 = str(Path(sys.argv[2]))
name3 = str(Path(sys.argv[3]))

data_anmo = pd.read_csv(name1)
data_majo = pd.read_csv(name2)
data_tsum = pd.read_csv(name3)

params_anmo = 2*9
params_majo = 2*15
params_tsum = 2*12

aic_anmo = []
kk_anmo = np.arange(len(data_anmo))+8
kk1_anmo = np.fliplr(kk_anmo.reshape(1,-1))
kk1_anmo = kk1_anmo[0,:]
aic = np.zeros(len(kk_anmo))

for i in kk1_anmo:
    k = np.asarray(data_anmo[0:i])    
    kl = (i)*(np.log(2*np.pi)+1+np.log(k.T @ k/(i)))
    aic[i-8] = kl + params_anmo
#    print(i)
A = aic
np.save('aic_anmo_53.npy',A)
print('done')


aic_majo = []
kk_majo = np.arange(len(data_majo))+14
kk1_majo = np.fliplr(kk_anmo.reshape(1,-1))
kk1_majo = kk1_majo[0,:]
aic = np.zeros(len(kk_majo))

for i in kk1_majo:
    k = np.asarray(data_majo[0:i])    
    kl = (i)*(np.log(2*np.pi)+1+np.log(k.T @ k/(i)))
    aic[i-14] = kl + params_majo
#    print(i)
print(aic[-1])
A =aic
np.save('aic_majo_77.npy',A)
print('done')


aic_tsum = []
kk_tsum = np.arange(len(data_tsum))+11
kk1_tsum = np.fliplr(kk_tsum.reshape(1,-1))
kk1_tsum = kk1_tsum[0,:]
aic = np.zeros(len(kk_tsum))

for i in kk1_tsum:
    k = np.asarray(data_tsum[0:i+1])    
    kl = (i)*(np.log(2*np.pi)+1+np.log(k.T @ k/(i)))
    aic[i-11] = kl + params_tsum
#    print(i)
A = aic
np.save('aic_tsum_65.npy',A)
print('done')