setwd("/media/vivekkulkarni/362480E21B181E95/Fodder/src")
d = read.csv('../real_data/ANMO_Zchannel.csv')
d = as.numeric(unlist(d))
plot(as.ts(d))

datas = as.ts(d[1:50000])
plot(datas)

a = acf(datas,100)
p = pacf(datas,100)

d = diff(d)
plot(d)