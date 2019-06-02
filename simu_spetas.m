function y=simu_spetas(X,Y,T,mu,A,sig,omega)
%simulate multivariate Hawkes process with with exp(t) kernel in time and bivariate Gaussian(x) in space.
%output in chronological order
%keep only the ones in space window - to do
%here sig is var instead of std!

% Inputs  -- para
% A: M-by-M matrix   A(i,j) # of events in j trigger by i
% omega: scalar time scale
% sig: scalar spatial scale
% T: scalar time window [0,T]
% X,Y: scalars spatial window [0,X]*[0,Y]

% Outputs 
% simulated point processes data =(y.type, y.t, y.lon, y.lat) i = 1,...,y.n. Here u(i) is the subprocess for event i.
% Baichuan Yuan 6/2/2019 
%step1: background points
bmu=sum(mu);
n_types=length(mu);
n_start=poissrnd(bmu*T);
bp = struct('n',n_start,'lon',[],'lat',[],'t',[],'type',[]);
bp.lon=rand(1,bp.n)*X;
bp.lat=rand(1,bp.n)*Y;
bp.t=sort(rand(1,bp.n)*T);
bp.type=ones(1,bp.n);
bp.father=zeros(1,bp.n);
temp=rand(1,bp.n);

for i=1:n_types-1
%     ind=find(temp>mu(i)/bmu & temp<=mu(i+1)/bmu);
    bp.type((temp>sum(mu(1:i))/bmu & temp<=sum(mu(1:i+1))/bmu))=i+1;
end
flag=0;
if bp.n<0.5
    flag=2;
end
w=bp;

%step2: aftershocks
af_types=sum(A,2)';
while flag<1
    af=struct('n',[],'lon',[],'lat',[],'t',[],'type',[]);
    n2=poissrnd(af_types(w.type),1,w.n);
    af.n=sum(n2);
    af.type=[];
    af.lon=[];
    af.lat=[];
    af.t=[];
    af.father=[];

    for i=1:w.n
        if n2(i)>0.5
            b1=exprnd(1/omega,1,n2(i));
            af.t=[af.t,b1+w.t(i)];
            xy=mvnrnd([0,0],[sig,0;0,sig],n2(i));
            af.lon=[af.lon,xy(:,1)'+w.lon(i)];
            af.lat=[af.lat,xy(:,2)'+w.lat(i)];
            temp=rand(1,n2(i));
            temp_type=ones(1,n2(i));
            for j=1:n_types-1
                temp_type((temp>sum(A(w.type(i),1:j))/af_types(w.type(i)) & temp<=sum(A(w.type(i),1:j+1))/af_types(w.type(i))))=j+1;
            end
            af.type=[af.type,temp_type];
            af.father=[af.father,(i+bp.n-w.n)*ones(1,n2(i))];
        end
    end
    %combine
    if af.n>0.5
        bp.n=bp.n+af.n;
        bp.t=[bp.t,af.t];
        bp.lat=[bp.lat,af.lat];
        bp.lon=[bp.lon,af.lon];
        bp.type=[bp.type,af.type];
        bp.father=[bp.father,af.father];
        w=af;
        if min(af.t)>T
            flag=2;
        end
    end
    
    if af.n<0.5
        flag=2;
    end     
end
y=struct('n',[],'lon',[],'lat',[],'t',[],'type',[]);
[y.t,ind]=sort(bp.t);
y.id=[1:1:bp.n];
y.id=y.id(ind);
y.n=bp.n;
y.t=y.t';
y.lon=bp.lon(ind)';
y.lat=bp.lat(ind)';
y.type=bp.type(ind)';
y.father=bp.father(ind);
% y.father(y.father>0)=ind(y.father(y.father>0));

end   