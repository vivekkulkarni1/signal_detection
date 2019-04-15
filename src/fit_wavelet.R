setwd('/media/vivekkulkarni/362480E21B181E95/Fodder/src')
fs = 1000
t = seq(0,99)/fs
A0 = 10
h = 50
t0 = t[0]
f0 = 65
A = A0 * exp(-h * (t)) * sin((t)*2* pi * f0) 
plot(t,A,type="l")
a = acf(A)
p = pacf(A)
mod1 = arima(A,order=c(2,0,1))
rs = mod1$residuals
plot(seq(1,100),rs,'p')
aresid = acf(rs)
presid = pacf(rs)