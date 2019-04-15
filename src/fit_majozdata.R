setwd("/media/vivekkulkarni/362480E21B181E95/Fodder/src")
d = read.csv('../real_data/IUMAJO10BHZ.csv')
d = as.numeric(unlist(d))
plot(as.ts(d))
d1 = as.ts(d[1:100])
d2 = as.ts(d[1:1000])
d3 = as.ts(d[1:5000])
d4 = as.ts(d[1:10000])
d5 = as.ts(d[1:25000])
d6 = as.ts(d[1:50000])
datas = as.ts(d[1:40000])
d7 = as.ts(d[50001:100000])
d8 = as.ts(d[1:100000])
fs = 40
plot(datas)
#fit d1
# plot(d1)
# a1 = acf(d1,60)
# p1 = pacf(d1,60)
# mod1 = arima(d1,order = c(4,1,3),method = 'CSS',seasonal = c(17,0,0))
# plot(mod1$residuals)
# ar1 = acf(mod1$residuals,50)
# pr1 = pacf(mod1$residuals,50)

#fit d2
# plot(d2)
# a2 = acf(d2)
# p2 = pacf(d2)
# mod2 = arima(d2,order = c(4,1,3),seasonal =c(20,0,0),method='CSS')
# k = fft(d2)
# k = abs(k)
# k = k[2:1000]
# freq = linspace(0,fs,1000)
# freq1 = freq[2:1000]
# plot(k,type='l')
# ar2 = acf(mod2$residuals,50)
# pr2 = pacf(mod2$residuals,50)

#fit d3
# a2 = acf(d2)
# p2 = pacf(d2)
# mod2 = arima(d2,order = c(7,1,1),seasonal =c(51,0,0),method='CSS') 
# ar2 = acf(mod2$residuals,500)
# pr2 = pacf(mod2$residuals,500)

#fit d_util


a = acf(datas,100)
p = pacf(datas,100)

d_util = diff(datas)
plot(d_util)

mod = ar(d_util)
mod1 = arima(d_util,order = c(7,0,7))
ar_res = acf(na.omit(mod$resid),400)
pr_res = pacf(na.omit(mod$resid),400)

ar1_res = acf(na.omit(mod1$residuals),400)
pr1_res = pacf(na.omit(mod1$residuals),400)