%% Spar Class
% By 6110
classdef Spar
    %SPAR Class for Wing Spar
    %   The purpose of useing a class is to simplyfy the number of
    %   variables sent in function handles in fmincon_handle.m, without
    %   using global variables that cause a significant slowdown of
    %   performance. The other reason was to give the Author experiance
    %   with object oreinted programming in MATLAB, which is needed for the
    %   Author's Neural Net
    properties
        w;
        L;
        x0;
        rho;
        Nx;
        E;
        Y;
        A;
        b;
        lb;
        ub;
        NcolPts;
        NpertPts;
    end
    methods
        function obj =Spar(w,L,rho,E,Y,Nx,x0,A,b,lb,ub,NCP,NPP) % method to initilizae values.
            obj.w=w; % plane weight
            obj.L=L; % spar length
            obj.rho=rho; % density
            obj.E=E; % Youngs mod
            obj.Y=Y; % yeild stress
            obj.Nx=Nx; % number of nodes
            obj.x0=x0; % inital geometry
            obj.A=A; % bound
            obj.b=b; % bound 
            obj.lb=lb; % lower bound
            obj.ub=ub; % upper bound
            obj.NcolPts=NCP; % Number of Collection points
            obj.NpertPts=NPP;
        end
    end
end

