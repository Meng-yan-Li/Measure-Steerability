%% GHZ states & 2 measurement (2-UNT)
clear
clc

oa=2; % number of outcomes of A_1
ob=2; % number of outcomes of A_2
ma=2; % number of measurements of A_1
mb=2; % number of measurements of A_2
par = [0:0.1:0.6,0.66,2/3,0.67,0.7:0.1:1]; % visibility parameter of the target assemblage

S_ub = upper_bound(par,oa,ob,ma,mb);
S_lb = lower_bound(par,oa,ob,ma,mb);

figure(1)
plot(par,S_ub,'m','LineWidth',2)
hold on 
plot(par,S_lb,'b','LineWidth',1)
hold off
%% GHZ states & 3 measurement (2-UNT)
clear
clc

oa=2; % number of outcomes of A_1
ob=2; % number of outcomes of A_2
ma=3; % number of measurements of A_1
mb=3; % number of measurements of A_2
par = [0:0.1:0.4,0.42,0.4285,0.43,0.5:0.1:1];

S_ub = upper_bound(par,oa,ob,ma,mb);
S_lb = lower_bound(par,oa,ob,ma,mb);

figure(2)
plot(par,S_ub,'m','LineWidth',2)
hold on 
plot(par,S_lb,'b','LineWidth',1)
hold off