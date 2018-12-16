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
plot(Nxi(1:end-1), dipEnd(1:end-1))
xlabel('Nx')
ylabel('Spar tip displacment (meters)')
grid on
%% test beam disp
% disp = F*L^3 /(3EI)
F=100;
L=7.5;
%E=
I=Iyy(1);
d1 = (F*L^3)/(3*E*I);
Nelem=15;
Foc=zeros(Nelem,1);
%Foc(end)=F;
Iyy=I*ones(length(Iyy));
Foc(end)=F;
 [u] = CalcBeamDisplacement(L, E, Iyy, Foc, Nelem-1)
