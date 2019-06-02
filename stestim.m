function [A,B,omega,sig,tau,aic,p,pb] = stestim(H)

% A parametric estimation for multivariate spatiotemporal Hawkes processes with exp(t) kernel in time and bivariate Gaussian(x) in space.
% based on 'Topic time series analysis of microblogs', IMA Journal of Applied Mathematics, 2016. used in our paper 'Multivariate Spatiotemporal Hawkes Processes and Network Reconstruction', to appear in SIAM Journal on Mathematics of Data Science, 2019. 

% Inputs
% H: N-by-4 
% observed point processes data H(i,:)=(u(i), t(i), x(i), y(i)) i = 1,...,N. Here u(i) is the subprocess for event i.

% Outputs   -- para
% u: 1-by-M vector   background rate
% A: M-by-M matrix   A(i,j) # of events in j trigger by i
% B: M-by-M matrix   Similar to A, for background
% w: scalar time scale
% sig=tau: scalar spatial scale
% aic: scalar Akaike information criterion 
% p, pd: N-by-N matrix triggering/background prob defined in EM

% Baichuan Yuan 6/2/2019 

N_iter = 500;
err_tol = .0001;

LAT=H(:,4);
LON=H(:,3);
N=length(H);

coord = 0; % coord = 1 if use Great-circle distance 
if coord
%     [X,Y] = deg2utm(LAT,LON);
%     X=(X-min(X))/(max(X)-min(X));
%     Y=(Y-min(Y))/(max(Y)-min(Y));
    rp_lat=repmat(LAT,1,length(LAT));
    rp_lon=repmat(LON,1,length(LON));
    [arclen] = distance(reshape(rp_lat,1,N^2),reshape(rp_lon,1,N^2),...
    reshape(rp_lat',1,N^2),reshape(rp_lon',1,N^2));
    distance1 = reshape(arclen,N,N);
else
    distance1 = dist2([LAT,LON],[LAT,LON]);
    distance1 = distance1 - spdiags(diag(distance1), 0, N, N);
end

tdata=H(:,2)';
topics=H(:,1)';
M = max(topics);
T=tdata(end);

deltat = triu(bsxfun(@minus, tdata, repmat(tdata', [1 N])));

mul_w=2*N/T;

% Using MLE to initalize
% lb=zeros(1,2*M^2+2);   %MLE low bound
% ub=ones(1,2*M^2+2);
% ub(2*M^2+1)=mul_w;
%para0=rand(1,2*M^2+2);
% para0(3)=0.75;
% para0(4)=0.6;
%[para,fval,exitflag,output,lambda,grad,hessian]=fmincon(@(xx)log_likesp(xx,H,deltat,distance1,M),para0,[],[],[],[],lb,ub);

% Default: Using Random initialization
para=rand(1,2*M^2+2);
A=reshape(para(1:M^2),M,M);
B=reshape(para(M^2+1:2*M^2),M,M);
omega=para(2*M^2+1);
sig=para(2*M^2+2);
tau=para(2*M^2+2);

%use previous initialization
lkh=0;

for kk =1:N_iter
  lasts=sig;
  lastomega=omega;
  lastA=A;
  lastB=B;
  lastt=tau;
  lastl=lkh;
    
  % step 1: E-step - get P matrix from the previous parameters
    
  [pb,p] = Expcstepst(A,B,omega,sig,tau,H,N,deltat,distance1);
    

  % Step 2: M-step - Estimate u, A, w and sigma from the data 
    
  % omega
  %definition: deltat(i,j)=t(j)-t(i)
  At=A';
%    [omega] = fminbnd(@(oo) abs(sum(sum(p))-(sum(sum(p.*deltat))+sum((T-tdata).*exp(-oo*(T-tdata)).*sum(At(:,topics))))*oo),0,mul_w);
% 	[omega,w_residual,eflag] = fzero(@(oo) sum(sum(p))-(sum(sum(p.*deltat))+sum((T-tdata).*exp(-oo*(T-tdata)).*sum(At(:,topics))))*oo,[0,5000*N/T]);
% 	if eflag~=1
% 		w_residual
% 		break
% 	end
% 	if errors>0.1
% 		errors
% 		break
% 	end
  omega=sum(sum(p))/(sum(sum(p.*deltat))+sum(sum(bsxfun(@etimes,(T-tdata).*exp(-omega*(T-tdata)),At(:,topics)))));

  %A
  etotimes=exp(-omega*(T-tdata));
  for i=1:length(unique(topics))
    for j=1:length(unique(topics))
      A(i,j)=sum(sum(p(topics==i, topics==j)))/sum(1-etotimes(topics==i));
    end
  end
  
  %B
  for i=1:length(unique(topics))
    for j=1:length(unique(topics))
      B(i,j)=sum(sum(pb(topics==i, topics==j)))/(sum(topics==i));
    end
  end   
  

  
  %sigma
  sig=sqrt(sum(sum(p.*distance1+pb.*distance1))/(2*sum(sum(p+pb))));
  
  %tau
  tau=sig;

  xx=[reshape(A,1,M^2),reshape(B,1,M^2),omega,sig];
  lkh=log_likesp(xx,H,deltat,distance1,M);
  error=max(max(abs(lastA-A)))+abs(lastomega-omega)+abs(lastt-tau)+abs(lasts-sig)+max(max(abs(lastB-B)));
  
  if mod(kk,50)==0
    fprintf('iter %d: error = %g, lkh = %g\n', kk, error, lkh);
  end
  
  aic=4*M^2+4+2*lkh;
  if  error< err_tol || abs(lastl-lkh)<err_tol
      break
  end
  
  if isnan(sum(xx))
      fprintf('NAN!!')
      break
  end
end

