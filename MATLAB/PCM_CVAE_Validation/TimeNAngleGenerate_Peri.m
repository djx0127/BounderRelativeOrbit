function [Td, OMGd,tp,xp] = TimeNAngleGenerate_Peri(Hz,k,alpha,DeltaE)
% Generate the crossing period and separation angle
% Input:
%     hz Angular momentum around z-axis
%     k  Magnitude of thrust acceleration
%     a  Pitch angle between thrust and z-axis
%     DeltaE  Energy deviation, determine the size of periodic orbit
%     rol,col(optional)  Obtain gradient colors
% Output:
%     crossing period ΔT and separation angle ΔΩ

% if nargin >= 6 && ~isempty(row) && ~isempty(col)
%     RGBColor = GetGradientColor(row, col); % Call function GetGradientColor.m
% end

if isempty(k) || isempty(alpha)
        error('k or alpha is empty!');
end
%% Initial parameter setting
% Integrator settings
RelTol = 3.e-06 ; AbsTol = 1.e-09; % lowest accuracy
MODEL = 'dynamics';
OPTIONS = odeset('RelTol',RelTol,'AbsTol',AbsTol,'Events','on');
RelTol = 3.e-14 ; AbsTol = 1.e-16; % high accuracy
OPTIONS2 = odeset('RelTol',RelTol,'AbsTol',AbsTol,'Events','on');

T_range= 1500;

[rou,z,rou0,z0,~]  = PeriorbitInitialState_DiffertialCorrection(Hz,k,alpha,DeltaE);
if isnan(rou) || isnan(z) || isnan(rou0) || isnan(z0) || isnan(T_range)
    Td = NaN;
    OMGd = NaN;
    return;
end

zp          =  z0; %%   Poincare section z=z0
x0po_peri   = [rou;z;0;0;0];
tspan       = linspace(0,T_range,1000);
Energy_peri = Hz^2/(2*rou^2)-1/sqrt(rou^2+z^2)-k*z*cos(alpha)-k*rou*sin(alpha)+1/2*(x0po_peri(3)^2+x0po_peri(4)^2);
[tp,xp,tep,xep] = ode113('dyna2_param',tspan,x0po_peri,OPTIONS2,Hz,k,alpha,zp);

rho_calc = xp(:,1); rho_e = xep(:,1);
z_calc   = xp(:,2); z_e   = xep(:,2);
phi_calc = xp(:,5); phi_e = xep(:,5);

Td =  tep(2)-tep(1);
OMGd =  phi_e(2)-phi_e(1)-2*pi;

%% Draw the distribution map
% X_i = rho_calc.*cos(phi_calc);
% Y_i = rho_calc.*sin(phi_calc);
% Z_i = z_calc;
% figure(1)
% plot3(X_i,Y_i,Z_i,'LineWidth',0.5);
% hold on; grid on;
% X_e = rho_e.*cos(phi_e);
% Y_e = rho_e.*sin(phi_e);
% Z_e = z_e;
% 
% figure(1)
% plot3(X_i,Y_i,Z_i,'LineWidth',0.5,'Color', [RGBColor.R,RGBColor.G,RGBColor.B] );
% hold on; 
% plot3(X_e,Y_e,Z_e,'k.','MarkerSize',12);
% grid on; box on;
% xlabel('\itx'); ylabel('\ity'); zlabel('\itz');
% set(gca,'FontSize',14); set(gca,'FontName','Times New Roman');
% set(gcf, 'Position', [100, 100, 500, 450]);
% 
% 
% figure(2)
% plot(rho_calc,z_calc,'LineWidth',1,'Color', [RGBColor.R,RGBColor.G,RGBColor.B] );
% hold on; grid on;
% plot(rho_e,z_e,'k.','MarkerSize',12);
% xlabel('\it\rho'); ylabel('\itz');
% set(gca,'FontSize',14); set(gca,'FontName','Times New Roman');
% set(gcf, 'Position', [100, 100, 500, 450]);
end