close all; clear all;

load('wkspc_Nx_55.mat')
%figure(1)
%hold on
dipEnd=[];
Nxi=5:5:55;
for i=1:length(dipBmc)
    figure(i)
    
    dp=cell2mat(dipBmc(i));
    dipEnd(i)=dp(end);
    plot(dp)
    title(sprintf('Nx = %d',Nxi(i)))
    %clear dp
end
figure(nodes)
hold on
plot(dipEnd)

%% test beam disp
% disp = F*L^3 /(3EI)
Nelem=15;
q=100; % uniform
L=7.5;
x=1:1:Nelem;
%I=Iyy(1);
r_in = 0.0415*ones(Nelem,1); % nominal values of inner radii
thicc = 0.0085*ones(Nelem,1); % nominal values of annular thickness
r_out=r_in+thicc;
Iyy = (pi/4).*(r_out.^4-r_in.^4);
I=Iyy(1);
dx = ((q.*x.^2)/(24*E*I)).*(6*L^2-4*L.*x+x.^2);
tx = ((q.*x)/(6*E*I)).*(3*L^2-3*L.*x+x.^2);
%E=
Fdist=q*ones(Nelem,1);

%d1 = (F*L^3)/(3*E*I);
%theta1=(F*L^2)/(2*E*I);

%Foc=zeros(Nelem,1);
%Foc(end)=F;
%Iyy=I*ones(length(Iyy));
%Foc(end)=F;
 [u] = CalcBeamDisplacement(L, E, Iyy, Fdist, Nelem-1)


%{
F=100;
L=7.5;
%E=
I=Iyy(1);
d1 = (F*L^3)/(3*E*I);
theta1=(F*L^2)/(2*E*I);
Nelem=15;
Foc=zeros(Nelem,1);
%Foc(end)=F;
Iyy=I*ones(length(Iyy));
Foc(end)=F;
 [u] = CalcBeamDisplacement(L, E, Iyy, Foc, Nelem-1)
%}