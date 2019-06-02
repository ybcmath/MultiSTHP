function [p,hk] = ExpcstepTemp(mu,A,omega,H,deltat)
  %calculate g(t_i-t_j)
  %first calculate t_i-t_j for j=1 up to i-1
  %The i's change going down the columns (first coordinates).
  tdata=H(:,2)';
  topics=H(:,1)';
  N=length(tdata);
  P=omega*A(topics,topics).*triu(exp(-omega*deltat),1); 
  %TODO: Each column is just multiplying by e^-\omega(t_{i+1}-t_{i}). They have
  %to belong to the same topic.
  P=P+diag(mu(topics));
  deno=sum(P);
  p=bsxfun(@times,triu(P),1./deno);
  diag_p = diag(p);
  p(deltat==0)=0;
  p = p + diag(diag_p);
  At=A';
  T=tdata(end);
  hk=sum(log(deno))-sum(sum(bsxfun(@etimes,1-exp(-omega*(T-tdata)),At(:,topics))))-T*sum(mu);
end






