function hk=log_mulHawkes(xx,H,deltat,M)
%computes the log-likelihood of a multidimensional Hawkes Process
%T: total time
%mu , w and A are the parameters
%t is the time series
%mk,dim
A=reshape(xx(1:M^2),M,M);
u=xx(M^2+1:M^2+M);
omega=xx(M^2+M+1);

N=length(H);
tdata=H(:,2)';
topics=H(:,1)';
T=tdata(end);

At=A';
P=omega*A(topics,topics).*triu(exp(-omega*deltat)); 
hk=-sum(log(u(topics)+sum(P)))+sum(sum(bsxfun(@etimes,1-exp(-omega*(T-tdata)),At(:,topics))))+T*sum(u);

