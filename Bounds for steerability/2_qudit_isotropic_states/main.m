%% Isotropic states & 2 measurement
clear
clc

d = 2;
par = [0:0.1:0.7,1/sqrt(2),0.71,0.8:0.1:1]; % visibility parameter of the target assemblage
% d = 3;
% par = [0:0.1:0.6,0.68,0.683012699901561,0.69,0.7:0.1:1];
% d = 4;
% par = [0:0.1:0.6,0.66,2/3,0.67,0.7:0.1:1];
% d = 5;
% par = [0:0.1:0.6,0.65,0.654508481556034,0.66,0.7:0.1:1];

S_lb = lower_bound(par,d);
S_ub = upper_bound(par,d);

figure(1)
plot(par,S_lb,'b','LineWidth',1)
hold on 
plot(par,S_ub,'m','LineWidth',2)
hold off

%% Isotropic states & 3 measurement
clear
clc

d = 2;
par = [0:0.1:0.5,0.57,1/sqrt(3),0.58,0.6:0.1:1]; % visibility parameter of the target assemblage
% d = 3;
% par = [0:0.1:0.5,0.56,0.568579017586038,0.57,0.6:0.1:1]; 
% d = 4;
% par = [0:0.1:0.5,0.55,5/9,0.56,0.6:0.1:1];
% d = 5;
% par = [0:0.1:0.5,0.53,0.539344661920183,0.54,0.6:0.1:1];

S_lb = lower_bound(par,d);
S_ub = upper_bound(par,d);

figure(2)
plot(par,S_lb,'b','LineWidth',1)
hold on 
plot(par,S_ub,'m','LineWidth',2)
hold off

%% Isotropic states & d+1 measurement
clear
clc

% d = 2;
% par = [0:0.1:0.5,0.57,1/sqrt(3),0.58,0.6:0.1:1]; % visibilities
% d = 3;
% par = [0:0.1:0.4,0.48,0.481762739304125,0.49,0.5:0.1:1];
% d = 4;
% par = [0:0.1:0.4,0.43,0.430940100483820,0.44,0.5:0.1:1];

S_lb = lower_bound(par,d);
S_ub = upper_bound(par,d);

figure(3)
plot(par,S_lb,'b','LineWidth',1)
hold on 
plot(par,S_ub,'m','LineWidth',2)
hold off