%% Project 4 Master Code
% By 6110
%% Inital Variables and Constraints
% computational values
% Change these for altering the analysis model
close all; clear all;
nodes =15; % number of nodes evaluated at
NCP=3;
NPP=4;
complex_step_size = 1e-60;
% model variables
mass =500; % kg
L = 7.5; % meter
E = 70*10^9; % Pa
Y=600 *10^6; % Yeild Stress Pa
rho = 1600.0;   % density (kg /m^3)
w = (2.5 * 9.8 *.5* 500); % 2.5 * plane weight, note that the 2.5 is put here to avoid as many calculations in the fmincon as possible

%% Create bounds
% Creates linear bounds
rdist=2.5*10^-3; % thickness, m
rmin=1*10^-2; % min radi m
rmax = 5*10^-2; % max radi m


A=zeros(nodes,nodes*2); % prealloc
for i=1:nodes
    A(i, i)=1; % lower node 
    A(i, i+nodes)=1; % upper node
    b(i)=rmax; % distance 
     
    lb(1,i)=rmin; %% lb for r_low
    ub(1,i)=rmax-rdist; %% ub for r_low
    lb(1,i+nodes)=rdist; % lb for thick
    ub(1,i+nodes)=rmax-rmin-rdist; %up for think
end

%% Initalize wing object
% provide the nominal spar values as initial parameter
r_in = 0.0415*ones(nodes,1); % nominal values of inner radii
thicc = 0.0085*ones(nodes,1); % nominal values of annular thickness
x0 = [r_in;thicc];            % concactenated design vector

wing=Spar(w,L,rho,E,Y,nodes,x0,A,b,lb,ub,NCP,NPP,complex_step_size);
%% Start fmincon
tic % start timer
[sprMss,sprGeo,exp,var,std,wts,pts,exitFlag, fmincOutput ] =fmincon_handle(wing);
fprintf('Nx=%d,Num Colc Pts = %d, Spar is %.5f kg, ',nodes,NCP, sprMss); % print mass
toc % output run time
makePlots(sprGeo,pts,wts,wing,exp,var,std); % create plots

