# Multivariate Spatiotemporal Hawkes Processes
**Spatiotemporal Hawkes processes** is a self-exciting point process model, and has found myriad applications in earthquake aftershock and crime forecasting. 

This Matlab package is developed for the following paper:
```
Baichuan Yuan, Hao Li, Andrea Bertozzi, P. Jeffrey Brantingham, and Mason Porter, 
Multivariate Spatiotemporal Hawkes Processes and Network Reconstruction, 
to appear in SIAM Journal on Mathematics of Data Science, 2019. 
```
Please cite this paper if you find the code useful.

## Multivariate model
A multivariate spatiotemporal Hawkes process (STHP) is characterized via its conditional intensity:
```
	\lambda_u(t,x,y) = \mu_u(x,y) + \sum_{t>t_i}  K_{u_iu} g(x-x_i,y-y_i,t-t_i)\,.
```
We estimate the triggering matrix K, background rate \mu_u and triggering kernel g from the observed data (X_i,y_i,t_i).
All these Matlab functions are documented. To get started, run the script `test.m` Please find the helper texts at the beginning of each M-file for more options.

## Basic usage

To estimate a multivariate STHP given observed point processes data H(i,:)=(u(i), t(i), x(i), y(i)) i = 1,...,N. Here u(i) is the subprocess for event i.
```
[A,B,omega,sig,tau,aic,p,pb] = stestim(H)
```
or:
```
res=nphawkes(H)
```
To simulate multivariate STHPs:
```
y=simu_spetas(X,Y,T,mu,A,sigma,omega);
```
Please refer to the documentation for more options. A summary of the functions in this package is listed below:

User functions (API):
* `nphawkes.m`: an EM-based nonparametric estimation for multivariate STHP
* `stestim.m`: an EM-based parametric estimation for multivariate STHP
* `simu_spetas.m`: a Poisson clustering process based simulations for multivariate STHP
* `tempestim.m`: an EM-based parametric estimation for multivariate THP

