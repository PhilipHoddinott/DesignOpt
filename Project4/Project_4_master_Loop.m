%% Project 4 Master Code
% By 6110
%% Inital Variables and Constraints
% computational values
% Change these for altering the analysis model
close all; clear all;
counter=1;
for nodes =65:10:355%55:5:250
%nodes =15; % number of nodes evaluated at
NCP=3;
NPP=4;
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

stepSize=1e-60;
wing=Spar(w,L,rho,E,Y,nodes,x0,A,b,lb,ub,NCP,NPP,stepSize);
%% Start fmincon
tic % start timer

[sprMss,sprGeo,exp,var,std,wts,pts,exitFlag, fmincOutput ] =fmincon_handle(wing);
[r_out, r_in, Iyy] = get_RO_RI_Iyy(sprGeo)
[sigmaExp, sigmaVar,dispBeam] = CalcStress(pts,wts,r_out,Iyy,wing)
dipBmc(counter)={dispBeam};
expBmc(counter)={sigmaExp};
varBmc(counter)={sigmaVar};


dipEnd=[];
for i=1:counter
    dp=cell2mat(dipBmc(i))
    dipEnd(i)=dp(end);
    clear dp
end
figure(counter)
hold on
plot(dipEnd)
st=sprintf('End Disp for Nx = %d',nodes);
title(st)
grid on
xlabel('Nodes')
ylabel('disp')
drawnow
counter=counter+1;
%{
%[r_out, r_in, Iyy] = get_RO_RI_Iyy(sprGeo)
[pts, wts] = get_pts_wts(wing)
[r_out, r_in, Iyy] = get_RO_RI_Iyy(x0)
[sigmaExp, sigmaVar,dispBeam] = CalcStress(pts,wts,r_out,Iyy,wing)
dipBmc(counter)={dispBeam};
expBmc(counter)={sigmaExp};
varBmc(counter)={sigmaVar};
std = sqrt(var - exp.*exp);
counter=counter+1;
%fprintf('Nx=%d,Num Colc Pts = %d, Spar is %.5f kg, ',nodes,NCP, sprMss); % print mass
%toc % output run time
%makePlots(sprGeo,pts,wts,wing,exp,var,std); % create plots
figure(nodes-1)
plot(dispBeam)
xlabel('span')
ylabel('disp')
grid on
%}
strSv=sprintf('wkspc_Nx_%d',nodes);
save(strSv)
toc
end
dipEnd=[];
for i=1:nodes-1
    dp=cell2mat(dipBmc(i))
    dipEnd(i)=dp(end);
    clear dp
end
figure(nodes)
hold on
plot(dipEnd)


    function [sigmaExp, sigmaVar,dispBeam] = CalcStress(pts,wts,r_out,Iyy,wing)
        sigmaExp = 0; % prealloc
        sigmaVar = 0; % prealloc
        for n=1:(wing.NcolPts^4)
            forceBeam = CalcForce(pts(n,:),wing); % get force on bream
            dispBeam = CalcBeamDisplacement(wing.L,wing.E,Iyy,forceBeam,wing.Nx-1); % get beam displacment  
            sigmaBeam = CalcBeamStress(wing.L,wing.E,r_out,dispBeam,wing.Nx-1); % get stress on beam
            sigmaExp = sigmaExp + wts(n)*sigmaBeam; % get mean
            sigmaVar = sigmaVar + wts(n)*(sigmaBeam).^2;% get var
        end

    end
    
    function [force] = CalcForce(pts,wing) % compute force at points
        delta_F = 0; % prealloc
        x = linspace(0,wing.L,wing.Nx); % create linspace
        forceNomX = (2*wing.w/wing.L)*(1-x./wing.L); % inital force
        for n=1:wing.NpertPts % for this code it is 4, but for other problems you might want more than 4 pts
            delta_F = delta_F + pts(n)*cos(((2*n-1).*pi.*x)/(2*wing.L)); % from project discription
        end
        force = forceNomX + delta_F; % return force
    end
    
    function [r_in,r_out]= getRoutRinFunc(desVar) % function to get r_out, r_in
        lnDV=length(desVar); % get length of designVariable
        r_in = desVar(1:lnDV/2); % get r_inner
        r_out = r_in+ desVar(lnDV/2+1:end); % get r_outer
    end
    
    function [r_out, r_in, Iyy] = get_RO_RI_Iyy(desVar) % function to get r_out, r_in, and Iyy
        [r_in,r_out]= getRoutRinFunc(desVar);
        Iyy = (pi/4).*(r_out.^4-r_in.^4);
    end

    
    function [pts, wts] = get_pts_wts(wing) %% This is imporatnt
        % This is done at the beginning, as it takes a long time to run
        % DO not use it in fmincon!!
        fNom0 = 2*wing.w/wing.L;
        mu = zeros(1,4);%
        
        sigma(1:4)=fNom0.*1./(10*(1:4));

        [xi, w] =GaussHermite(wing.NcolPts); % get points and weights
        pts = zeros((wing.NcolPts^4),4); % prealloc
        wts = zeros((wing.NcolPts^4),1); % prealloc

        i = 1;
        for i1=1:wing.NcolPts
            xi1 = sqrt(2)*sigma(1)*xi(i1) + mu(1); % first layer 
            for i2=1:wing.NcolPts
                xi2 = sqrt(2)*sigma(2)*xi(i2) + mu(2);  % second layer 
                for i3=1:wing.NcolPts
                    xi3 = sqrt(2)*sigma(3)*xi(i3) + mu(3);  % third layer 
                    for i4=1:wing.NcolPts
                        xi4 = sqrt(2)*sigma(4)*xi(i4) + mu(4);  % fourth layer 
                        pts(i,:) = [xi1, xi2, xi3, xi4]; % get points
                        wts(i) = w(i1)*w(i2)*w(i3)*w(i4); % get weights
                        i = i+1; % increment i
                    end
                end
            end
        end
        fprintf('pts and wts created\n');
    end
    
     function [x, w] = GaussHermite(n)
        % Function to determines the abscisas (x) and weights (w) for the
        % Gauss-Hermite quadrature of order n>1, on the interval [-INF, +INF].
        % works for n>=2
        % Credit to Geert Van Damme (geert@vandamme-iliano.be)
        % See referances section
        if n<2
            error('Warning Number of Collection points below 2');
        end

        i   = 1:n-1;
        a   = sqrt(i/2);
        CM  = diag(a,1) + diag(a,-1);

        % CM is such that det(xI-CM)=L_n(x), with L_n the Hermite polynomial
        % under consideration. Moreover, CM will be constructed in such a way
        % that it is symmetrical.
        [V, L]   = eig(CM);
        [x, ind] = sort(diag(L));
        V       = V(:,ind)';
        w       = sqrt(pi) * V(:,1).^2;
        w=w./sqrt(pi); % adjust weight
    end
    

