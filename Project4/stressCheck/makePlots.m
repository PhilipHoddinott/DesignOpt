%% makePlots Function
% By 6110
function [] = makePlots(sparGeo,~,~,wing,exp,var,~ )
%MAKEPLOTS Function to make plots
%   Makes colorful diagram and then a figure of stress and gemonetry
    r_low = sparGeo(1:wing.Nx);
    r_up = r_low + sparGeo(wing.Nx+1:2*wing.Nx);
    std = sqrt(var - exp.*exp);
    x = linspace(0,wing.L,wing.Nx)';
    hfig =  findobj('type','figure');
    nfig = length(hfig);
    cArr=[0.7098    0.8392    0.9686]; % color
    
    figure(nfig+1)
    hold on
    pup=area(x,r_up); % upper 
    pup.FaceColor=[.84 .84 .84]; % color
    plow=area(x,r_low); % inside area
    plow.FaceColor=cArr; % color
    pNup=area(x,-r_up); % lower 
    pNup.FaceColor=[.84 .84 .84]; % color
    pNlow=area(x,-r_low); % inside lower area 
    pNlow.FaceColor=cArr; % color
    ylabel(' y/z axis (meters)'); xlabel('x axis (meters)')
    ylim([-0.05 0.05])
    grid on
    % title('Spar Diagram')
    
    figure(nfig+2)
    hold on
    plot([x(1),x(end)],[600*1e6,600*1e6],'-k')
    plot(x,exp,'-om')
    plot(x,exp+6*std,'-sr');     plot(x,exp-6*std,'-sg')
    xlabel('x axis (meters)'); ylabel('Stress (N/m^2) '); xlim([ 0 7.5])
    grid on
    yyaxis right
    ylim([.01 .05])
    plot(x,r_low,'-+')
    plot(x,r_up,'b-x')
    % strT=sprintf('Spar shape vs Stress at %d nodes',wing.Nx); title(strT)
    legend('Maximum Stress','Mean Stress ($\mu$)','$\mu +6\sigma$','$\mu-6\sigma$','Lower Radius','Upper Radius','Location','best','Interpreter','latex')
    ylabel('y/z axis (meters)')
end

