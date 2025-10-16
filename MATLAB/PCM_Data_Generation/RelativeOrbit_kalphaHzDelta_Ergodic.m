clear;
close all;

%% State parameters definition
% Test parameters
% Hz1 = 2.3452;         % Hz2 = 2.4495;
% DeltE1 = 0.01;        % DeltE2 = 0.005;

% Domain of state parameters
Hz0 = 1.5; Hz1 = 2.5;                      % z-axis angular momentum
DeltE0 = 0; DeltE1 = 0.005;                % Energy deviation
alpha0 = deg2rad(0); alpha1 = deg2rad(80); % Pitch angle
k0 = 0.001; k1 = 0.0035;                   % Acceleration magnitude

%% Call subfunction to generate feature parameter (ΔT,ΔΩ)
N = 50;  % 50 points per dimension

k = linspace(k0,k1,N)';
alpha = linspace(alpha0,alpha1,N)';
Hz = linspace(Hz0,Hz1,N)';
DeltE = linspace(DeltE0,DeltE1,N)';

results = zeros(N^4,6);

poolobj = gcp('nocreate'); % Checking parallel computing tool
if isempty(poolobj)
    poolobj = parpool(8);  % 8-core parallel computing
end
%% Parallel ergodic computing for 4D stateparameters
tic
parfor idx = 1:N^4
    % Convert indexes into 4D subscripts
    [a,b,c,d] = ind2sub([N,N,N,N], idx);

    % Obtain the current parameter
    current_k = k(a);
    current_alpha = alpha(b);
    current_Hz = Hz(c);
    current_DeltE = DeltE(d);
    [Td, OMGd] = TimeNAngleGenerate_Peri(current_Hz, current_k, current_alpha, current_DeltE);
    results(idx, :) = [current_k, current_alpha, current_Hz, current_DeltE, Td, OMGd];
    if mod(idx, ceil(N^4/100)) == 0
        fprintf('Progress: %.1f%%\n', idx/N^4 * 100);
    end
end
toc

% Dataset storage
save('PCM_Dataset\4D_traversal_results.mat', 'N', 'results'); 
disp('Data saved to 4D_traversal_results.mat');
X_4d = valid_data(:,1:4);
Y_2d = valid_data(:,5:6);
save('img\4D_traversal_results_XandY_New.mat', 'X_4d','Y_2d');

delete(poolobj)


%% Dataset visualization
% Retain valid data
valid_data = results(all(~isnan(results(:,5:6)), 2), :);

% 2D projection diagram
figure(1);
plot(valid_data(:,6), valid_data(:,5), 'b.','MarkerSize',8);
xlabel('\Delta\Omega'); ylabel('\Delta\itT');
title(sprintf('Valid results: %d/%d', size(valid_data,1), N^4));
grid on;
set(gca,'FontSize',16); set(gca,'FontName','Times New Roman');
set(gcf, 'Position', [100, 100, 500, 450]);
saveas(gcf,'img\4D_traversal_results.fig');

% 2D projection heat map
figure(2);
subplot(2,2,1)
scatter(valid_data(:,6), valid_data(:,5),5,valid_data(:,1),'filled');
shading interp; box on; grid on
custom_map = [linspace(0,1,256)', zeros(256,1), linspace(1,0,256)'];
colormap(custom_map);
zz1 = colorbar;
xlabel('\Delta\Omega'); ylabel('\Delta\itT'); ylabel(zz1, '\it\kappa');
set(gca,'FontSize',16); set(gca,'FontName','Times New Roman');
subplot(2,2,2)
scatter(valid_data(:,6), valid_data(:,5),5,valid_data(:,2),'filled');
shading interp; box on; grid on
custom_map = [linspace(0,1,256)', zeros(256,1), linspace(1,0,256)'];
colormap(custom_map);
zz2 = colorbar;
xlabel('\Delta\Omega'); ylabel('\Delta\itT'); ylabel(zz2, '\it\alpha');
set(gca,'FontSize',16); set(gca,'FontName','Times New Roman');
subplot(2,2,3)
scatter(valid_data(:,6), valid_data(:,5),5,valid_data(:,3),'filled');
shading interp; box on; grid on
custom_map = [linspace(0,1,256)', zeros(256,1), linspace(1,0,256)'];
colormap(custom_map);
zz3 = colorbar;
xlabel('\Delta\Omega'); ylabel('\Delta\itT'); ylabel(zz3, '\itH_z');
set(gca,'FontSize',16); set(gca,'FontName','Times New Roman');
subplot(2,2,4)
scatter(valid_data(:,6), valid_data(:,5),5,valid_data(:,4),'filled');
shading interp; box on; grid on
custom_map = [linspace(0,1,256)', zeros(256,1), linspace(1,0,256)'];
colormap(custom_map);
zz4 = colorbar;
xlabel('\Delta\Omega'); ylabel('\Delta\itT'); ylabel(zz4, '\Delta\itE');
set(gca,'FontSize',16); set(gca,'FontName','Times New Roman');

saveas(gcf,'PCM_IMG\DTandDOMG_Distribution_wrt4Dparam.fig');

