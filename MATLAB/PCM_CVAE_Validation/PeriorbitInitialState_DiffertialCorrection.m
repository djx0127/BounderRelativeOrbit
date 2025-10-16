function [rou,z,rou0,z0,T_range] = PeriorbitInitialState_DiffertialCorrection(Hz,k,alpha,DeltaE)
% PeriorbitInitialState_DiffertialCorrection
% --------------------------------------------------------------
% Purpose:
%   Computes the initial conditions of a periodic (hovering) orbit
%   near the equilibrium point by performing a differential correction
%   on the initial state around the equilibrium configuration.
%
% Inputs:
%   Hz      - Angular momentum
%   k       - Thrust magnitude
%   alpha   - Thrust direction (radians)
%   DeltaE  - Energy deviation from the equilibrium energy
%
% Outputs:
%   rou, z  - Corrected initial state of the periodic orbit
%   rou0, z0 - Equilibrium point coordinates
%   T_range - Estimated period range of the periodic orbit
%
% --------------------------------------------------------------

%% Integration and solver settings
RelTol = 3e-14;
AbsTol = 1e-16; % High accuracy for numerical integration
OPTIONS2 = odeset('RelTol', RelTol, 'AbsTol', AbsTol, 'Events', 'on');

%% Step 1: Solve for equilibrium point (rho0, z0) from zero root of f(rho)
PosilibpointSolve_fun = @(x) PosilibpointSolve_param(x,Hz,k,alpha);
rou0 = fsolve(PosilibpointSolve_fun,6,optimoptions('fsolve','TolX',1e-16,'Display','off'));
z0 = k *cos(alpha) * rou0^4 / (Hz^2 + rou0^3*k*sin(alpha));
PosiPoint = [rou0,z0];

%% Step 2: Compute Jacobian matrix and eigenvalues at equilibrium
omg = 0;
A11 = -3*Hz^2/rou0^4-1/(sqrt(rou0^2+z0^2))^3+3*rou0^2/(sqrt(rou0^2+z0^2))^5;
A12 = 3*rou0*z0/(sqrt(rou0^2+z0^2))^5;
A22 = -1/(sqrt(rou0^2+z0^2))^3+3*z0^2/(sqrt(rou0^2+z0^2))^5;
A = [A11,A12;A12,A22];
W = [0,omg;-omg,0];
C = [zeros(2),eye(2);A,W];

% Eigen-decomposition for the linearized system
[V,D] = eig(A);
Tsimu = 2*pi/sqrt(abs(D(1,1)));
correc = 1 ;
x0 = V*[correc;0];

%% Step 3: Compute energy at equilibrium and energy correction
Energy0 = Hz^2/(2*rou0^2)-1/sqrt(rou0^2+z0^2)-k*z0*cos(alpha)-k*rou0*sin(alpha);
Energy  = DeltaE + Energy0; 

% Solve for the correction velocity Vr to match the desired energy level
eqenergy_fun = @(p) eqenergy(p,Hz,k,alpha,rou0,z0,omg,Energy);
% Vr = fsolve(eqenergy_fun,2,optimoptions('fsolve','TolX',1e-16,'Display','off'));
Vr = 1;
x0po = [x0(1);x0(2);0;0]*Vr;
rou = rou0 + x0po(1);
z = z0 + x0po(2);

%% Step 4: Coordinate transformation matrices
Lx = @(theta) [1,0,0; 0,cos(theta),sin(theta); 0,-sin(theta),cos(theta)]';
Ly = @(theta) [cos(theta),0,-sin(theta); 0,1,0; sin(theta),0,cos(theta)]';
Lz = @(theta) [cos(theta),sin(theta),0; -sin(theta),cos(theta),0; 0,0,1]';

%% Step 5: Differential correction iteration
max_attemps = 6;
max_iterations = 1000;

for KK = 1:max_attemps
    tt = 0; xx = [reshape(eye(4,4),16,1);x0po]; 
    mr=1;nr=1;xx2=[];
    iteration = 0;
    %       while ~((tt>(Tsimu*4/5))*(xx(20)<0))%    
    % Integration loop until trajectory crosses symmetry plane
    while ~((tt > Tsimu*4/5) && (mr*nr < 0)) && iteration < max_iterations   
        mr=xx(20);
        varEqs_DiffertialCorrection_fun = @(t,PHI) varEqs_DiffertialCorrection(t,PHI,Hz,k,alpha,PosiPoint);
        % try
            [tt,xx] = runku45(varEqs_DiffertialCorrection_fun,tt,xx);
        % catch
        %     rou = NaN;
        %     z = NaN;
        %     T_range = NaN;
        %     return;
        % end
        nr = xx(20);
        iteration = iteration + 1;
        % xx2=[xx2;xx'];
    end
    % if iteration >= max_iterations
    %     rou = NaN;
    %     z = NaN;
    %     T_range = NaN;
    %     return;
    % end
    %    figure(2); hold on; plot(xx2(:,17)+rou0,xx2(:,18)+z0,'b','MarkerSize',8);     

    % Compute state transition matrix and correction
    FF = reshape(xx(1:16),4,4); % state transition matrix
    F11 = FF(1,1); F12 = FF(1,2); 
    F21 = FF(2,1); F22 = FF(2,2);
    F31 = FF(3,1); F32 = FF(3,2);

    rou = rou0 + xx(17); z = z0+ xx(18); r = sqrt(rou^2 + z^2);
    ax = Hz^2/rou^3 - rou/r^3 +k*sin(alpha);
    ay = -z/r^3  +k*cos(alpha);
    Urou = -(Hz^2/rou^3 - rou/r^3+k*sin(alpha));
    Uz  = -(-z/r^3  +  k*cos(alpha));
    C =  -Urou/Uz ;
    % Construct correction matrix
    VF = [F12*C+F11-1, xx(19);...
          F22*C+F21-C, xx(20);...
          F31+C*F32,   ax];

    temp = inv(VF'*VF)*VF'*[x0po(1)-xx(17);x0po(2)-xx(18);-xx(19)];
    drou = temp(1);  
    dz = C*drou;     
    % Update initial conditions
    x0po=x0po+[drou;dz;0;0]; 
end

%% Step 6: Final corrected periodic orbit state
rou = rou0 + x0po(1);
z = z0 + x0po(2);
omg_peri = Hz/(rou^2);
T_range = 5*2*pi/omg_peri;

%% Visualization of periodic orbit results, optional
% % Energy_Peri = Hz^2/(2*rou^2)-1/sqrt(rou^2+z^2)-k*z*cos(alpha)-k*rou*sin(alpha)+1/2*(x0po(3)^2+x0po(4)^2);  % 周期轨道的能量
% % 
% %% Compute periodic orbit
% % Integral computation
% x0po = [rou;z;x0po(3);x0po(4);0];
% tspan = [0,T_range];
% zp = z0
% % dyna2_fun = @(t,x) dyna2_param(t,x,Hz,k,alpha);
% [t1,x1,te2,xe2] = ode113('dyna2_param',tspan,x0po,OPTIONS2,Hz,k,alpha,zp) ;
% % Integrate and simultaneously find where the event function is zero. 
% % Output: te event time, ye solution at the event occurrence, ie trigger event index
% 
% % Result definition
% rho_calc = x1(:,1); z_calc   = x1(:,2); phi_calc = x1(:,5); 
% theta_calc = atan(rho_calc./z_calc);
% X_i = rho_calc.*cos(phi_calc);
% Y_i = rho_calc.*sin(phi_calc);
% Z_i = z_calc;
% rho_e = xe2(:,1); z_e   = xe2(:,2); phi_e = xe2(:,5);
% X_e = rho_e.*cos(phi_e);
% Y_e = rho_e.*sin(phi_e);
% Z_e = z_e;
% 
% figure(1)
% plot3(X_i,Y_i,Z_i); hold on; grid on;
% plot3(X_e,Y_e,Z_e,'r.','MarkerSize',8);
% 
% figure(2)
% plot(rho_calc,z_calc);
% hold on; grid on;
% plot(rho_e,z_e);
% % xlabel('\rho'); ylabel('z');
% % set(gca,'FontSize',14); set(gca,'FontName','Times New Roman');
% 
% figure(3)
% plot(t1,rho_calc)

end