function [pb,p] = Expcstepst(A,B,omega,sig,tau,H,N,deltat,distance)

  tdata=H(:,2)';
  topics=H(:,1)';
  T=tdata(end);


  P=omega*A(topics,topics).*triu(exp(-omega*deltat)).*exp(-distance/(2*sig^2))/(2*pi*sig^2); 
  Pb=B(topics,topics).*exp(-distance/(2*tau^2))/(2*pi*tau^2*T);
  
  P=P-diag(diag(P));
  Pb=Pb-diag(diag(Pb));
  P(deltat==0)=0;
  P(distance==0)=0;
  Pb(distance==0)=0;
  temp=sum(Pb+P);

for i=1:N
   if temp(i)==0
       Pb(i,i)=1;
   end
end
  
  deno=sum(P)+sum(Pb);
  
  p=bsxfun(@edivides,P,deno);
  pb=bsxfun(@edivides,Pb,deno);
  
  
end




