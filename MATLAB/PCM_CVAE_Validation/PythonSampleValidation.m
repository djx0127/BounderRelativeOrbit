%% ========================================================================
%  Formation Validation Script (PCM–CVAE)
%  ------------------------------------------------------------------------
%  This script evaluates and visualizes bounded formation generation results
%  using the semi-analytical PCM–CVAE method.
%
%  Part 1: Generate a reference periodic orbit sample (chief spacecraft)
%  Part 2: Load CVAE-generated samples (deputy spacecraft)
%  Part 3: Compare and visualize relative configurations
%  ------------------------------------------------------------------------
%  Author: Jixin Ding
%  Date:   25-10-16
% ========================================================================

clear;
close all;


% -------------------------------------------------------------------------
%  Reference data bounds (used for normalization)
% -------------------------------------------------------------------------
Tmax = 49.89999967;
Tmin = 20.69417536;
Omax = -0.01730248;
Omin = -0.51546563;


% -------------------------------------------------------------------------
%  Reference sample parameters (chief spacecraft)
% -------------------------------------------------------------------------
Hz_sample = 1.8;
kappa_sample = 0.0015;
alpha_sample = deg2rad(30);
DeltE_sample = 0.003;

% Generate reference periodic orbit
[Td_sample, OMGd_sample, tp_sample, xp_sample] = ...
    TimeNAngleGenerate_Peri(1.8,0.0015,deg2rad(30),0.003);
% Normalize time and angular difference
[Td_sample_norm] = normalize_to_01(Td_sample, Tmin, Tmax);
[OMGd_sample_norm] = normalize_to_01(OMGd_sample, Omin, Omax);
% [Td_sample, OMGd_sample, tp_sample, xp_sample] = TimeNAngleGenerate_Peri(1.8,0.0017,deg2rad(30),0.003)

% Extract chief spacecraft state
rho_sample = xp_sample(:,1); Vrho_sample = xp_sample(:,3);
z_sample   = xp_sample(:,2); Vz_sample   = xp_sample(:,4);
phi_sample = xp_sample(:,5);
theta_sample = atan(z_sample./rho_sample);
Xi_sample = rho_sample.*cos(phi_sample);
Yi_sample = rho_sample.*sin(phi_sample);
Zi_sample = z_sample;
Vx_sample = Vrho_sample.*cos(phi_sample);
Vy_sample = Vrho_sample.*sin(phi_sample);

% [X,Y,Z,Vx,Vy,Vz] = Cylindrical2Cartesian(rho,z,phi,Vrho,Vz);

% -------------------------------------------------------------------------
%  Rotation matrices (used for coordinate transformations)
% -------------------------------------------------------------------------
Lx = @(theta) [1,0,0; 0,cos(theta),sin(theta); 0,-sin(theta),cos(theta)];
Ly = @(theta) [cos(theta),0,-sin(theta); 0,1,0; sin(theta),0,cos(theta)];
Lz = @(theta) [cos(theta),sin(theta),0; -sin(theta),cos(theta),0; 0,0,1];


% -------------------------------------------------------------------------
%  Visualization settings
% -------------------------------------------------------------------------
colors = lines(10);
markers = {'o', '+', '*', 'x', 's', 'd', '^', 'v'};
h = zeros(10,1);

% -------------------------------------------------------------------------
%  Load CVAE-generated formation samples
% -------------------------------------------------------------------------
for jdx = 1:1
    filename = sprintf('Python/CVAE_Model/CVAEOutputData.mat');
    load(filename);

    % Extract input parameters from CVAE output
    kappa = double(X_samples(:,1));
    alpha = double(X_samples(:,2));
    Hz    = double(X_samples(:,3));
    DeltE = double(X_samples(:,4));
    
    % Initialize containers
    Td = zeros(length(kappa),1);
    OMGd = zeros(length(kappa),1);
    Hz_select = zeros(length(kappa),1);
    kappa_select = zeros(length(kappa),1);
    alpha_select = zeros(length(kappa),1);
    DeltE_select = zeros(length(kappa),1);
    Delta_Xi = zeros(length(kappa),1000);
    Delta_Yi = zeros(length(kappa),1000);
    Delta_Zi = zeros(length(kappa),1000);
    
    CNT = 1;
    idx_ergodic = 1:length(kappa);
    idx_formation = [917,1427,1753]; % Selected best-performing CVAE samples

    % ---------------------------------------------------------------------
    %  Formation generation and validation
    % ---------------------------------------------------------------------

    for idx = idx_formation
        fprintf("Processing sample #%d\n", idx);

        % Generate deputy periodic orbit
        [Td_temp, OMGd_temp,tp,xp] = TimeNAngleGenerate_Peri(Hz(idx),kappa(idx),alpha(idx),DeltE(idx));
        % [Td(idx), OMGd(idx),tp,xp] = TimeNAngleGenerate_Peri(Hz(idx),kappa(idx),alpha(idx),DeltE(idx));
        
        % Compute relative errors
        [Td_temp_norm] = normalize_to_01(Td_temp, Tmin, Tmax);
        [OMGd_temp_norm] = normalize_to_01(OMGd_temp, Omin, Omax);
        Sigma_T = abs(Td_temp-Td_sample)/abs(Td_sample);
        Sigma_O = abs(OMGd_temp-OMGd_sample)/abs(OMGd_sample);
        
        % Filter by tolerance
        if Sigma_T < 3e-4 && Sigma_O < 2e-3
            Td(CNT) = Td_temp;
            OMGd(CNT) = OMGd_temp;
            Hz_select(CNT) = Hz(idx);
            kappa_select(CNT) = kappa(idx);
            alpha_select(CNT) = alpha(idx);
            DeltE_select(CNT) = DeltE(idx);
            CNT = CNT + 1;

            % Relative position (chief - deputy)
            rho = xp(:,1); z = xp(:,2); phi = xp(:,5);
            Xi  = rho.*cos(phi); Yi = rho.*sin(phi); Zi = z;
            Delta_Xi(idx,:) = Xi_sample - Xi;
            Delta_Yi(idx,:) = Yi_sample - Yi;
            Delta_Zi(idx,:) = Zi_sample - Zi;

            % Transform relative motion into orbital frame
            State_o2 = zeros(3,length(tp));
            for ii = 1:length(tp)
                % State_o2(:,ii) = Lz(phi_calc1(ii))*[X_diff(ii);Y_diff(ii);Z_diff(ii)];
                State_o2(:,ii) = Ly(theta_sample(ii))*Lz(phi_sample(ii))*[Delta_Xi(idx,ii);Delta_Yi(idx,ii);Delta_Zi(idx,ii)];
            end
            Delta_Xo = State_o2(1,:)';
            Delta_Yo = State_o2(2,:)';
            Delta_Zo = State_o2(3,:)';

            % -------------------------------------------------------------
            %  Visualization
            % -------------------------------------------------------------
            % Chief–Deputy absolute orbits
            figure
%             subplot(1,2,1)
            h1 = plot3(Xi_sample,Yi_sample,Zi_sample,'r','LineWidth',1); 
            hold on; grid on; box on;
            h2 = plot3(Xi,Yi,Zi,'b','LineWidth',1);
            legend([h1,h2],'Chief s/c', 'Deputy s/c');
            xlabel('\itx_i'); ylabel('\ity_i'); zlabel('\itz_i');
            set(gca,'FontSize',14); set(gca,'FontName','Times New Roman');
            axis square; set(gcf, 'Position', [100, 100, 500, 450]);
%             subplot(1,2,2)
%             plot3(Delta_Xi(idx,:),Delta_Yi(idx,:),Delta_Zi(idx,:),'b','LineWidth',1);
%             hold on; grid on; box on; axis square; 
%             xlabel('\Delta\itx_i'); ylabel('\Delta\ity_i'); zlabel('\Delta\itz_i');
%             set(gca,'FontSize',14); set(gca,'FontName','Times New Roman');    

            % Relative orbit in ECI plots 
            figure
            subplot(2,2,1)
            plot3(Delta_Xo,Delta_Yo,Delta_Zo,'b','LineWidth',1);
            hold on; grid on; box on; axis square; 
            xlabel('\Delta\itx_o'); ylabel('\Delta\ity_o'); zlabel('\Delta\itz_o');
            set(gca,'FontSize',12); set(gca,'FontName','Times New Roman');   
            subplot(2,2,2)
            plot(Delta_Xo,Delta_Yo,'b','LineWidth',1);
            hold on; grid on; axis square; 
            xlabel('\Delta\itx_o'); ylabel('\Delta\ity_o');
            set(gca,'FontSize',12); set(gca,'FontName','Times New Roman');
            subplot(2,2,3)
            plot(Delta_Yo,Delta_Zo,'b','LineWidth',1);
            hold on; grid on; axis square; 
            xlabel('\Delta\ity_o'); ylabel('\Delta\itz_o');
            set(gca,'FontSize',12); set(gca,'FontName','Times New Roman');
            subplot(2,2,4)
            plot(Delta_Xo,Delta_Zo,'b','LineWidth',1);
            hold on; grid on; axis square; 
            xlabel('\Delta\itx_o');ylabel('\Delta\itz_o');
            set(gca,'FontSize',12); set(gca,'FontName','Times New Roman');
            set(gcf, 'Position', [100, 100, 500, 450]);
        end

    end
    % Clean up zero entries
    Td(Td == 0) = []; OMGd(OMGd == 0) = [];
    Hz_select(Hz_select == 0) = [];
    kappa_select(kappa_select == 0) = [];
    alpha_select(alpha_select == 0) = [];
    DeltE_select(DeltE_select == 0) = [];
    
    % Scatter plot comparison
    figure(101)
    h_scat = scatter(OMGd,Td,20,'Marker',markers{jdx},'MarkerEdgeColor','r','DisplayName',sprintf('VAE std = %d',jdx));
    hold on; grid on;
end
% -------------------------------------------------------------------------
%  Sample vs VAE result comparison
% -------------------------------------------------------------------------
figure(101)
h_samp = plot(OMGd_sample,Td_sample,'b*','MarkerSize',10,'DisplayName','Sample Point');
xlabel('\Delta\Omega'); ylabel('\Delta\itT');
legend([h_scat,h_samp],'VAE output','Sample value')
set(gca,'FontSize',14); set(gca,'FontName','Times New Roman');

% -------------------------------------------------------------------------
%  Parameter deviation visualization
% -------------------------------------------------------------------------
figure(102)
subplot(2,2,1)
plot(Hz_select,'r*','MarkerSize',8); hold on; grid on;
plot(Hz_select,'r:');
plot([0,length(Hz_select)], [Hz_sample,Hz_sample],'b-','LineWidth',1);
xlabel('Test Group'); ylabel('\itH_z');
set(gca,'FontSize',14); set(gca,'FontName','Times New Roman');

subplot(2,2,2)
plot(kappa_select,'r*','MarkerSize',8); hold on; grid on;
plot(kappa_select,'r:');
plot([0,length(kappa_select)], [kappa_sample,kappa_sample],'b-','LineWidth',1);
xlabel('Test Group'); ylabel('\it\kappa');
set(gca,'FontSize',14); set(gca,'FontName','Times New Roman');

subplot(2,2,3)
plot(alpha_select,'r*','MarkerSize',8); hold on; grid on;
plot(alpha_select,'r:');
plot([0,length(alpha_select)], [alpha_sample,alpha_sample],'b-','LineWidth',1);
xlabel('Test Group'); ylabel('\it\alpha');
set(gca,'FontSize',14); set(gca,'FontName','Times New Roman');

subplot(2,2,4)
h1 = plot(DeltE_select,'r*','MarkerSize',8); hold on; grid on;
plot(DeltE_select,'r:');
h2 = plot([0,length(DeltE_select)], [DeltE_sample,DeltE_sample],'b-','LineWidth',1);
xlabel('Test Group'); ylabel('\Delta\itE');
set(gca,'FontSize',14); set(gca,'FontName','Times New Roman');
legend([h1,h2],'Sample value','VAE output')

%% ========================================================================
%  Subfunctions
% ========================================================================

function Config_D = Evaluate_ConfigurationDispersion(DX,DY,DZ)
% Compute dispersion range in each axis
    Config_D.X = max(DX) - min(DX);
    Config_D.Y = max(DY) - min(DY);
    Config_D.Z = max(DZ) - min(DZ);
end

function [X,Y,Z,Vx,Vy,Vz] = Cylindrical2Cartesian(rho,z,phi,Vrho,Vz)
% Convert cylindrical to Cartesian coordinates
    X = rho.*cos(phi);
    Y = rho.*sin(phi);
    Z = z;
    Vx = Vrho.*cos(phi);
    Vy = Vrho.*sin(phi);
end

function normalized_data = normalize_to_01(data, data_min, data_max)
    % Normalize data into [0,1] range
    if any(data_max < data_min, 'all')
        error('Invalid input: data_max must be greater than data_min');
    end
    denominator = data_max - data_min;
    zero_mask = (denominator == 0);
    denominator(zero_mask) = 1;
    normalized_data = (data - data_min) ./ denominator;
    normalized_data(zero_mask) = 0.5;
    normalized_data = max(0, min(1, normalized_data));
end
