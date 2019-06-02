function [u,A,w,lkh,p,para,aic] = tempestim(H)


% EM for temporal multivariate Hawkes process
% based on 'Topic time series analysis of microblogs', IMA Journal of Applied Mathematics, 2016.

%     Inputs
% H      - history of the processes H(i,:)=(I(i) , t(i)) i = 1,...,N, 

%     Outputs   -- para
%     u: 1*M vector 
%     A: M*M matrix A(i,j) to # of events in j trigger by i
%     w: time scale
% aic: scalar Akaike information criterion 
% p: N-by-N matrix triggering/background prob defined in EM
% lkh: neg log-likelihood
% Baichuan Yuan 6/2/2019 

N=length(H);
tdata=H(:,2)';
topics=H(:,1)';
T=H(end,2);% T      - length of the time window of the dataset

%definition: deltat(i,j)=t(j)-t(i)
deltat=triu(bsxfun(@minus,tdata,tdata(tril(ones(N))*ones(N))));


lb=zeros(1,M^2+M+1);   %MLE low bound
ub=ones(1,M^2+M+1);
mul_w=2*N/T;
ub(M^2+1+M)=mul_w;
para0=rand(1,M^2+1+M);


[para,fval,exitflag,output,lambda,grad,hessian]=fmincon(@(xx)log_mulHawkes(xx,H,deltat,M),para0,[],[],[],[],lb,ub);
% para=para0;
A=reshape(para(1:M^2),M,M);
u=para(M^2+1:M^2+M);
w=para(M^2+M+1);

  lastomega=inf;
  lastA=inf;
  lastu=inf;
  lastl=inf;
  
topic_ind=cell(1,M);
for i=1:M
	topic_ind{i}=(topics==i);
end
inv_t=(T-tdata);

for kk =1:1000
    
  
  % step 1: P matrix
  [p,lkh] = ExpcstepTemp(u,A,w,H,deltat);
  error=max(max(abs(lastA-A)))+abs(lastomega-w)+max(abs(lastu-u));
  fprintf('iter %d: error = %g, lkh = %g\n', kk, error, lkh);
  aic=2*(M^2+1+M)-2*lkh;
  if  error<.0001 || abs(lastl-lkh)<0.001
    kk
      break
  end
  
%   if isnan(sum(xx))
%       fprintf('NAN!!')
%       break
%   end
  
  
  % Step 2: M-step
  lastomega=w;
  lastA=A;
  lastu=u;
  lastl=lkh;
  %		Estimate u_kk A_kk and w_kk from the sampled data 
        %p is upper triangular, A is a matrix, omega is a constant, T is the max time, tdata is the times (row vector),
        %topics is the row vector of topics;
  At=A';
  diagp=diag(p);
  temp_omega1=sum(sum(p.*deltat));
  temp_omega2=sum(At(:,topics));
  temp_p=sum(sum(triu(p,1)));
  [w] = fminbnd(@(oo) abs(temp_p-(temp_omega1+sum(inv_t.*exp(-oo*inv_t).*temp_omega2))*oo),0,mul_w);
%     [w] = fminbnd(@(oo) abs(sum(sum(p-diag(diag(p))))/(sum(sum(p.*deltat))+sum(sum(bsxfun(@etimes,(T-tdata).*exp(-oo*(T-tdata)),At(:,topics)))))-oo),0,50*N/T);
%     w=temp_p/(temp_omega1+sum(inv_t.*exp(-w*inv_t).*temp_omega2));
	
    etotimes=exp(-w*inv_t);
    pnodiag=p-diag(diagp);
	
    u=zeros(1,M);
    for i=1:M
        u(i)=sum(diagp(topic_ind{i}))/T;
        for j=1:M
            A(i,j)=sum(sum(pnodiag(topic_ind{i}, topic_ind{j})))/(sum(topic_ind{i})-sum(etotimes(topic_ind{i})));
        end
    end

            
end