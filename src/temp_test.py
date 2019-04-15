import numpy as np
import statsmodels.tsa.statespace.kalman_filter 

from statsmodels.tsa.statespace.representation import Representation 

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


rep = Representation()
rep.transition = np.reshape(A,(6,6,1))
rep.obs_intercept = np.zeros((1,0))
rep.obs_cov = np.zeros((1,1,1))
rep.design = np.reshape(C,(1,6,1))
rep.state_intercept = np.zeros((6,1))
rep.selection = np.reshape(B,(6,1,1))
rep.state_cov = np.reshape(Q,(1,1,1))