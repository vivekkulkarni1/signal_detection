import numpy as np 
from obspy import read
import sys
from pathlib import Path
import matplotlib.pyplot as plt

fname = str(Path(sys.argv[1]))
st = read(fname)
dat_size = st.count()
dataset = []
st.plot()
'''
for i in range(dat_size):
    dataset.append(st[i].data)
    temp = st[i].stats
    temp1 = str(temp)+'\n'+ '\n'
    filname = temp['network']+temp['station']+temp['location']+temp['channel'] + '.csv'
    namm = temp['network']+temp['station']+temp['location']+'.txt'
    #with open(namm,'a') as f:
    #    f.writelines(temp1)
    #np.savetxt(filname,st[i].data)    
'''    