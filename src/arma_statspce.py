import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.representation import Representation
from statsmodels.tsa.statespace.representation import FrozenRepresentation 
from statsmodels.tsa.statespace.kalman_filter import FilterResults,KalmanFilter
#import statsmodels.api as sm
#consider 45000 data points

#TSUM dataset

def kalman_filter(A,B,C,sigma,a,R):
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
'''
    xsmoothened[:,999] = xhat[:,999]
    for i in range(len(a)-2,0,-1):
        P0 = Ppost[i]
        Ptemp = P_prior[i-1]
        KK = P0 @ A.T @ np.linalg.inv(Ptemp)
        xsmoothened[:,i] = xhat[:,i]+KK @ (xsmoothened[:,i+1] - xhat_prior[:,i])
    return xsmoothened,xhat
'''

dataset = pd.read_csv("../real_data/IUTSUM00BHZ.csv")
da = dataset.values

tsum = da[:,0]

#TSUM ARIMA(6,1,5)
tsum_diff = np.diff(tsum)

p = 6
q = 5

m = np.max((p,q+1))
#Harvey method
if(p==m):
    vec = np.array([[1.9147,-0.9110,-0.2802,0.0428,0.7674,-0.5431]])
else:
    vec = np.array([[1.9147,-0.9110,-0.2802,0.0428,0.7674,-0.5431,0,0,0,0]])
vec = vec.T
I = np.eye(m-1)
k = np.zeros((1,m-1))
A = np.concatenate((I,k))
A = np.concatenate((vec,A),1)
C = np.zeros((1,6))
C[0,0] = 1
B = np.array([[1,-1.7746,0.3896,1.4270,-1.3554,0.3152]])
B = B.T
R = 0
Q = 1059


rep = Representation(k_endog = 1,k_states = 6,k_posdef=1)
rep.transition = np.reshape(A,(6,6,1))
rep.obs_intercept = np.zeros((1,0))
rep.obs_cov = np.zeros((1,1,1))
rep.design = np.reshape(C,(1,6,1))
rep.state_intercept = np.zeros((6,1))
rep.selection = np.reshape(B,(6,1,1))
rep.state_cov = np.reshape(Q,(1,1,1))
rep.initialize_stationary()

frep = FrozenRepresentation(rep)
#fit the statespacemodel
#do the forecasting
print(type(rep))
print(type(rep.transition))

'''
kf = KalmanFilter(k_endog=1,k_states=6,k_posdef=1)
#filt = FilterResults()
print(type(kf))
'''