function [res]=nphawkes(H)
% An nonparametric estimation for multivariate Hawkes process

% based on Baichuan Yuan, Hao Li, Andrea Bertozzi, P. Jeffrey Brantingham, and Mason Porter, 'Multivariate Spatiotemporal Hawkes Processes and Network Reconstruction', to appear in SIAM Journal on Mathematics of Data Science, 2019. 

% Inputs
% H: N-by-4 matrix
% observed point processes data H(i,:)=(u(i), t(i), x(i), y(i)) i = 1,...,N. Here u(i) is the subprocess for event i.
% X,Y: scalars spatial window [0,X]*[0,Y]

% Outputs   -- nonpara
% res.
% f: 1-by-nbins_r vector spatial triggering kernel
% g: 1-by-nbins_t vector temporal triggering kernel
% Z: nbins_x-by-nbins_y matrix   background rate over spatial grids
% K: M-by-M matrix   K(i,j) # of events in j trigger by i
% P: triggering/background prob defined in EM

% Baichuan Yuan 6/2/2019 

% Hyperparameters: spatial and temporal grid
time = H(:,2)';
topics = H(:,1)';
dim = max(topics);
T=time(end);
N=length(time);

np=50;
nbins_x=100;
nbins_y=100;
nbins_t=25;
nbins_r=30;
% np=20;
% nbins_x=30;
% nbins_y=30;
% nbins_t=20;
% nbins_r=30;
N_iter = 1000;
err_tol = 1e-04;
LAT=H(:,4);
LON=H(:,3);
coord = 0; % coord = 1 if use Great-circle distance 


% cal. distance
if coord
	rp_lat=repmat(LAT,1,length(LAT));
    rp_lon=repmat(LON,1,length(LON));
    [arclen] = distance(reshape(rp_lat,1,N^2),reshape(rp_lon,1,N^2),...
    reshape(rp_lat',1,N^2),reshape(rp_lon',1,N^2));
    distan = reshape(arclen,N,N);
	yr=[max(LON),min(LON)];
    xr=[max(LAT),min(LAT)];
else
	distan = real(sqrt(dist2([LAT,LON],[LAT,LON])));
	xr=[max(LON),min(LON)];
    yr=[max(LAT),min(LAT)];
end


%calculate spatial temporal difference 
distan= max(distan,1.0e-6);
distan = distan - spdiags(diag(distan), 0, N, N);

deltat = triu(bsxfun(@minus, time, repmat(time', [1 N])));
deltat= triu(max(deltat,1.0e-4),1);

d=ones(N,1)*0.02;
for i=1:N
    temp_dist=sort(distan(:,i));
    d(i)=max(temp_dist(np+1)+ 1e-08,0.02);
end


% log scale grid
delta_t  = 10.^linspace(min(1e-4, log10(min(min(deltat(deltat>0))))-1e-6),log10(max(max(deltat)))+1e-06,nbins_t+1);
delta_r  = 10.^linspace(floor(min(min(log10(distan(distan>0))))-1e-6), log10(max(max(distan))+0.5),nbins_r+1); 
%use max distance here instead of spatial window diagonal distance

delta_x  = linspace(min(xr),max(xr)+1e-08,nbins_x+1);
delta_y  = linspace(min(yr),max(yr)+1e-08,nbins_y+1);


% calculate grid index for each point
ind_t=zeros(N);
ind_r=zeros(N);
ind_x=zeros(N,1);
ind_y=zeros(N,1);
mid_x=zeros(nbins_x,1);
mid_y=zeros(nbins_y,1);
mid_r=zeros(nbins_r,1);
mid_t=zeros(nbins_t,1);
for i=1:nbins_t
    ind_t=ind_t+i*(deltat>=delta_t(i) & deltat <delta_t(i+1));
    mid_t(i)=(delta_t(i)+delta_t(i+1))*0.5;
end
for i=1:nbins_r
    ind_r=ind_r+i*(distan>=delta_r(i) & distan <delta_r(i+1));
	mid_r(i)=(delta_r(i)+delta_r(i+1))*0.5;
end

u_ind_t=triu(ind_t,1);
u_ind_t=u_ind_t(u_ind_t>0);
u_ind_r=triu(ind_r,1);
u_ind_r=u_ind_r(u_ind_r>0);

for i=1:nbins_x
    ind_x=ind_x+i*(H(:,3)>=delta_x(i) & H(:,3) <delta_x(i+1));
    mid_x(i)=(delta_x(i)+delta_x(i+1))*0.5;
end
for i=1:nbins_y
    ind_y=ind_y+i*(H(:,4)>=delta_y(i) & H(:,4) <delta_y(i+1));
    mid_y(i)=(delta_y(i)+delta_y(i+1))*0.5;
end
ind_bin_t=cell(1,nbins_t);
for i=1:nbins_t
    ind_bin_t{i}=(ind_t==i);
end
ind_bin_r=cell(1,nbins_r);
for i=1:nbins_r
    ind_bin_r{i}=(ind_r==i);
end

bin_r=diff(delta_r);
bin_t=diff(delta_t);
bin_x=diff(delta_x);
bin_y=diff(delta_y);


%% E-M algorithm
K=zeros(dim);
r_k=zeros(1,dim);
g=zeros(1,nbins_t);
h=zeros(1,nbins_r);
P=triu(rand(N));
deno=sum(P);
P=bsxfun(@times,P,1./deno);
%topic index
topic_ind=cell(1,dim);
for i=1:dim
	topic_ind{i}=(topics==i);
end

for kk=1:N_iter
    % M step
    Pnd=P-spdiags(diag(P), 0, N, N);
	%update K
    for i=1:dim
        for j=1:dim
            K(i,j)=sum(sum(Pnd(topic_ind{i}, topic_ind{j})))/sum(topic_ind{i});
        end
    end

	%update g and h
    sum_p=sum(sum(triu(P,1)));
    for i=1:nbins_t
        g(i)=sum(P(ind_bin_t{i}))/(sum_p*bin_t(i));
    end
    for i=1:nbins_r
        h(i)=sum(P(ind_bin_r{i}))/(sum_p*bin_r(i));
	end
    diag_p=diag(P);
    for i=1:dim
       r_k(i)=sum(diag_p(topic_ind{i}))/sum(diag_p);
    end


    % E step
    %cal background
    mu=zeros(nbins_x,nbins_y);
    Z=0;
    for i=1:nbins_x
        for j=1:nbins_y
            temp=(mid_x(i)-H(:,3)).^2+(mid_y(j)-H(:,4)).^2;
            mu(i,j)=dot(diag(P),exp(-temp(:)./(2*d(:).^2))./(2*pi*d(:).^2));
            Z=Z+mu(i,j)*abs(bin_x(i)*bin_y(j));
        end
    end
    mu=mu*sum(diag_p)/(T*Z);
	%update P
    Pnew=zeros(N);
    for j=1:N
        for i=1:j-1
            Pnew(i,j)=K(topics(i),topics(j))*g(ind_t(i,j))*h(ind_r(i,j))/(2*pi*mid_r(ind_r(i,j))); 
        end
        Pnew(j,j)=mu(ind_x(j),ind_y(j))*r_k(topics(j));
    end
    temp_diag=diag(Pnew);
	%remove near events in time
    Pnew(deltat<1.1e-6)=0;
    Pnew=Pnew+diag(temp_diag);
    deno=sum(Pnew);
    Pnew=bsxfun(@times,Pnew,1./deno);
	%determine convergence 
    err= max(max(abs(Pnew-P)));
	nb=sum(diag(Pnew));
	fprintf('iter %d: error = %g, # of background = %g\n', kk, err, nb);
	P=Pnew;
    if err<err_tol
        break
    end
end

%% output result
Pnd=P-spdiags(diag(P), 0, N, N);
for i=1:dim
	for j=1:dim
		theta_k(i,j)=sum(sum(Pnd(topics==i, topics==j)))/sum(sum(Pnd(topics==i, :)));
		var_k(i,j)=sum(sum(Pnd(topics==i, :)))*theta_k(i,j)*(1-theta_k(i,j))/(sum(topics==i))^2;
	end
end
res.delta_t=delta_t;
res.delta_r=delta_r;
res.f=h./(2*pi*mid_r');
res.g=g;
res.K=K;
res.P=P;
res.r=r_k;
res.mid_r=mid_r;
res.mid_t=mid_t;
res.var_k=var_k;
res.theta_k=theta_k;
res.Z=Z;

