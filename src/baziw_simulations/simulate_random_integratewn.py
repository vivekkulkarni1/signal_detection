#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 01:18:32 2018

@author: vivekkulkarni
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import arma_generate_sample 
#to generate the gauss markov stationary process -ARMA(1,1)

def kf3(d1,A,B,Q,R,fs,f0):
    t = np.linspace(0,1-1/fs,fs)
    C = np.array([[1,0,1]])
    xhat = np.zeros([3,len(d1)])
    xhat_prior = np.zeros([3,len(d1)])
    P_ = []
    Ppost = []
    xhat[:,0] = np.ones(3)
    P0 = np.eye(3)
    Ppost.append(P0)
    omega = 2*np.pi*f0
    for i in range(0,len(d1)-1):        
        xhat_prior[:,i] = A @ xhat[:,i]
        Ptemp = A @ P0 @ (A.T) + Q
        Ktemp = C @ Ptemp @ (C.T) + R
        i_Ktemp = np.linalg.inv(Ktemp)
        KGain = Ptemp @ C.T @ i_Ktemp
        ptr = C @ xhat_prior[:,i]
        xhat[:,i+1] = xhat_prior[:,i] + KGain @ (d1[i]- ptr )
        P_.append(Ptemp)
        P0 = (np.eye(3) - KGain @ C) @ Ptemp
        Ppost.append(P0)
        bb = 1*omega*np.cos(omega*(t[i+1]))/fs

        A = np.array([[1,bb , 0],[0,1,0],[0,0,0.75]])
    return xhat 

def stalta(dataset):
    LTA_window = 2000
    STA_window = 150
    temp1 = np.convolve((dataset)**2,np.ones(STA_window)/STA_window,mode='valid')
    temp2 = np.convolve((dataset)**2,np.ones(LTA_window)/LTA_window,mode='valid')
    temp1 = temp1[0:np.size(temp2)]
    return np.divide(temp1,temp2)

def modified_stalta(dataset):
    d = stalta(dataset)
    dtemp = dataset[0:len(d)]
    return abs(dtemp)*d**3

np.random.seed(1234)
mea1 = np.random.normal(0,0.5,1)
dat = np.random.normal(0,0.5,999)
dat1= np.concatenate((mea1,dat))
dat1 = np.cumsum(dat1)

dat = np.concatenate((dat1,np.mean(dat1)*np.ones(19000)))

fs = 20000
f0 = 700
f1 = 20

omega = 2*np.pi*f0
omega1 = 2* np.pi*f1

t = np.linspace(0,1-0.00005,20000)
t0 = t[2000]
t1 = t[10000]
B0 = 2
A0 = 4
h = 79
h1 = 50

A = A0 * np.exp(-h*(t-t0))*np.sin(omega*(t-t0))
A1 = np.concatenate((np.zeros(2000),A[2000:]))
B = B0 * np.exp(-h1*(t-t1))*np.sin(omega1*(t-t1))
B1 = np.concatenate((np.zeros(10000),B[10000:]))

arparams = np.array([0.75])
maparams = np.array([0.6])
ar = np.r_[1, -arparams] # add zero-lag and negate
ma = np.r_[1, maparams] 
gm_noise = arma_generate_sample(ar,ma,20000,sigma=0.1)
#plt.figure()
#plt.plot(t,gm_noise)

measurement_noise = np.random.normal(0,0.02,20000)
D = np.zeros([3,20000])
D[0,:] = A1+gm_noise + measurement_noise
D[1,:] = dat+gm_noise + measurement_noise
D[2,:] = B1+gm_noise + measurement_noise
l = ['P wave simulated','Random walk simulated','S wave simulated']
plt.figure()
for i in range(3):
    plt.subplot(3,1,i+1)
    plt.plot(t,D[i,:])
    plt.title(l[i])

A_1 = np.array([[1, 1*omega/fs, 0],[0,1,0],[0,0,0.75]])


B_1 = np.array([[0,0],[0.5,0],[0,0.6]])

Q = np.array([np.random.normal(0,0.01,20000),np.random.normal(0,0.02,20000),np.random.normal(0,0.03,20000)])
Q = np.matmul(Q,np.transpose(Q))/20000

OP = np.zeros([3,3,20000])
OP[0,:,:] = kf3(D[0,:],A_1,B_1,Q,np.var(measurement_noise),fs,f0)
OP[1,:,:] = kf3(D[1,:],A_1,B_1,Q,np.var(measurement_noise),fs,f0)
OP[2,:,:] = kf3(D[2,:],A_1,B_1,Q,np.var(measurement_noise),fs,f0)

plt.figure()
for i in range(3):
    plt.subplot(3,2,2*i+1)
    plt.plot(t,D[i,:])
    if(i==0):
        plt.title('Raw signal')
    plt.subplot(3,2,2*i+2)
    plt.plot(t,OP[i,0,:])
    if(i==0):
        plt.title('Filtered Signal')
#sta lta implementation - MER also
r = []
plt.figure()
for i in range(3):
    plt.subplot(3,2,2*i+1)
    plt.plot(t,OP[i,0,:])
    if(i==0):
        plt.title('Filtered Signal')
    plt.subplot(3,2,2*i+2)
    r1 = stalta(OP[i,0,:])
    plt.plot(t[0:len(r1)],r1)
    if(i==0):
        plt.title('STA/LTA')
    r.append(r1)    

mr=[]
plt.figure()
for i in range(3):
    plt.subplot(3,3,3*i+1)
    plt.plot(t,OP[i,0,:])
    plt.subplot(3,3,3*i+2)
    r1 = r[i]
    plt.plot(t[0:len(r1)],r1)
    tr1 = np.where(max(r1))
    tr1 = np.asarray(tr1)
    tr1 = tr1[0]
    plt.subplot(3,3,3*i+3)
    mr1 = modified_stalta(OP[i,0,:])
    mr.append(mr1)
    plt.plot(t[0:len(mr1)],mr1)