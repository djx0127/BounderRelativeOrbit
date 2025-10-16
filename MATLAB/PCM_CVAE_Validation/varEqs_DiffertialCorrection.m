function PHIdot = varEqs_DiffertialCorrection(t,PHI,Hz,k,alpha,InitialState)
% varEqs_DiffertialCorrection
% -------------------------------------------------------------------------
% Purpose:
%   Defines the combined variational and dynamic equations used for
%   differential correction in periodic orbit computation.
%
% Description:
%   This function computes the time derivatives of both the state vector
%   and the state transition matrix (STM) Φ(t), which together form the
%   system of equations used in the differential correction process.
%   The STM evolves according to Φ̇ = DF·Φ, where DF is the Jacobian
%   matrix of the dynamic system.
%
% Inputs:
%   t            - Time variable (not explicitly used but required by ODE solver)
%   PHI          - Combined vector containing:
%                   [Φ(16 elements); x(4 elements)]
%   Hz           - Angular momentum
%   k            - Thrust magnitude
%   alpha        - Thrust direction (radians)
%   InitialState - Initial equilibrium point [rho₀, z₀]
%
% Outputs:
%   PHIdot       - Time derivatives of the combined vector:
%                  [Φ̇(16 elements); ẋ(4 elements)]
%
% Notes:
%   - The state vector x = [rho, z, drho/dt, dz/dt].
%   - The first 16 elements of PHI represent the flattened 4×4 state
%     transition matrix.
% -------------------------------------------------------------------------

% Extract equilibrium point
rou0 = InitialState(1);
z0 = InitialState(2);
omg = 0;

% Initialize derivative vector
PHIdot =zeros(20,1);
% Extract current state variables
x(1)=PHI(17); 
x(2)=PHI(18); 
x(3)=PHI(19); 
x(4)=PHI(20); 

% Reshape state transition matrix Φ (16 → 4×4)
phi = reshape(PHI(1:16),4,4);

% Compute absolute coordinates
rou = x(1) + rou0;
z =x(2) + z0;
r = sqrt(rou^2 + z^2);

% ------------------------
% Dynamic equations
% ------------------------
PHIdot(17) = x(3);
PHIdot(18) = x(4);
PHIdot(19) = Hz^2/rou^3 - rou/r^3 + omg*x(4) + k*sin(alpha);
PHIdot(20) = -z/r^3 - omg*x(3) +  k*cos(alpha);

% ------------------------
% Jacobian matrix (DF)
% ------------------------
A11=-3*Hz^2/rou^4-1/(sqrt(rou^2+z^2))^3+3*rou^2/(sqrt(rou^2+z^2))^5;
A12=3*rou*z/(sqrt(rou^2+z^2))^5;
A22=-1/(sqrt(rou^2+z^2))^3+3*z^2/(sqrt(rou^2+z^2))^5;
A=[A11,A12;A12,A22];
W=[0,0;0,0];
DF=[zeros(2,2), eye(2,2); % Jacobi matrix
    A,          W];

% ------------------------
% Variational equations
% ------------------------
phidot=DF*phi;  
PHIdot(1:16)=reshape(phidot,16,1);

end
