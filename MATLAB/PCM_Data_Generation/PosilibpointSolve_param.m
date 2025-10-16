function Gl = PosilibpointSolve_param(p,Hz,k,alpha)
% PosilibpointSolve_param
% --------------------------------------------------------------
% Purpose:
%   Given a parameter set (Hz, k, alpha), this function computes
%   the equilibrium condition f(rho) = 0 to determine the
%   equilibrium point of the system.
%
% Inputs:
%   p      - Candidate variable (rho₁)
%   Hz     - Angular momentum
%   k      - Thrust magnitude
%   alpha  - Thrust direction (radians)
%
% Output:
%   Gl     - Value of the nonlinear equilibrium function f(rho₁)
%
% Description:
%   The function defines the equilibrium equation derived from
%   the dynamic balance condition. The equilibrium point (rho₀)
%   is obtained by solving f(rho₀) = 0 using fsolve.
% --------------------------------------------------------------

rou1  = p;
% Nonlinear equation representing the equilibrium condition
Gl(1) = (1 + k^2*(cos(alpha))^2*rou1^6/(Hz^2 + k*sin(alpha)*rou1^3)^2)^3 ...
        - rou1^2/(Hz^2 + k*sin(alpha)*rou1^3)^2;
end
