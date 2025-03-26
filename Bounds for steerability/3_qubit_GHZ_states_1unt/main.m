%% GHZ states & 2 measurement (1-UNT)
clear
clc

oa=2; % number of outcomes
ma=2; % number of measurements
par = [0:0.1:0.3,0.33 1/3 0.34,0.4:0.1:1]; % visibility parameter of the target assemblage

S_lb = lower_bound(par,oa,ma);
S_ub = upper_bound(par,oa,ma);

figure(1)
plot(par,S_ub,'m','LineWidth',2)
hold on 
plot(par,S_lb,'b','LineWidth',1)
hold off
%% GHZ states & 3 measurement (1-UNT)
clear
clc

oa=2;
ma=3;
par = [0:0.1:0.2,0.26,0.2612,0.27 0.3:0.1:1];
S_lb = lower_bound(par,oa,ma);
S_ub = upper_bound(par,oa,ma);

figure(2)
plot(par,S_ub,'m','LineWidth',2)
hold on 
plot(par,S_lb,'b','LineWidth',1)
hold off