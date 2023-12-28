clear;
load('cell_curve.mat')
plot(soc,ocv,'k')
xlabel("电池组SoC值",FontSize=14)
ylabel("电池组开路电压值/V",FontSize=14)