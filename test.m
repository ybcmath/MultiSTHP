% simulate multivariate STHPs
A1=importdata('A1.txt')'; %transpose due to different definition
mu1=importdata('mu1.txt')';
T = 2000;
X = 10;
Y = 10;
sigma1 = 0.3;
omega1 = 1;
y=simu_spetas(X,Y,T,mu1*10,A1,sigma1,omega1);
% Parametric estimation
H = [y.type, y.t, y.lon, y.lat];
[A,B,omega,sig,tau,aic,p,pb] = stestim(H);
% Nonparametric estimation
res = nphawkes(H,X,Y);