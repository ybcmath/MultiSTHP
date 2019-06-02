function l=log_likesp(xx,H,deltat,distance,M)%A11,A12,A21,A22,B11,B12,B21,B22,omega,sig,tau)%,H,x,y)

A=reshape(xx(1:M^2),M,M);
B=reshape(xx(M^2+1:2*M^2),M,M);
omega=xx(2*M^2+1);
sig=xx(2*M^2+2);
tau=xx(2*M^2+2);


N=length(H);
tdata=H(:,2)';
topics=H(:,1)';
T=tdata(end);

P=omega.*A(topics,topics).*triu(exp(-omega*deltat)).*triu(exp(-distance/(2*sig.^2)))/(2.*pi.*sig.^2); 
Pb=exp(-distance/(2.*tau.^2))/(2.*pi.*tau.^2.*T);  
P=P-diag(diag(P));
Pb=Pb-diag(diag(Pb));
  
Pb=B(topics,topics).*Pb;
temp=sum(Pb+P);
for i=1:N
   if temp(i)==0
       Pb(i,i)=1;
   end
end
       

l=-sum(sum(A(topics,:),2)'.*(1-exp(-omega.*(T-tdata))))-sum(sum(B(topics,:)))+sum(log(sum(P+Pb)));
l=-l; %negative log-likelihood
