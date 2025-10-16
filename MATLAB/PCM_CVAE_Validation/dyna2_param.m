function [out1,out2,out3] = dyna2_param(t,x,flag,Hz,k,alpha,zp)
% dyna2_param
% -------------------------------------------------------------------------
% Computes the dynamic equations or event conditions for the spacecraft
% motion model in cylindrical coordinates (ρ-z plane).
%
% Inputs:
%   t      - Time (s)
%   x      - State vector [ρ, z, ρ̇, ż, θ]
%   flag   - Optional event flag ('events')
%   Hz     - Angular momentum
%   k      - Thrust magnitude
%   alpha  - Thrust direction (radians)
%   zp     - z-plane value for the event condition
%
% Outputs:
%   If flag is empty:
%       out1 - State derivatives [ρ̇, ż, ρ̈, z̈, θ̇]
%   If flag = 'events':
%       out1 - Event function value (x(2) - zp)
%       out2 - isterminal flag (1 = stop integration)
%       out3 - direction flag (1 = detect zero crossing with positive slope)
% -------------------------------------------------------------------------

global T_range;


if nargin < 6 || isempty(flag) 
    % ----------------------------
    % Dynamic equations
    % ----------------------------
    r = x(1);
    z = x(2);
    dr = x(3);
    dz = x(4);
    theta = x(5);
    
    % Equations of motion (2D ρ-z version)
    d2r = Hz^2/r^3 - r/(r^2 + z^2)^(3/2) +  k*sin(alpha);
    d2z = -z/(r^2 + z^2)^(3/2) + k*cos(alpha) ;
    dtheta = Hz/r^2 ;

    % Return derivatives
    out1 = [dr; dz; d2r; d2z; dtheta];
else
    % ----------------------------
    % Event function definition
    % ----------------------------
    switch flag  % —— flag 的输入就是'events'
        case 'events'                           % Return [value,isterminal,direction].
            % Terminate integration when z = zp or time exceeds T_range
            if abs(t) > T_range
                isterminal = 1; % stop integration
            else
                isterminal = 0; % continue integration
            end
            direction = 1;      % detect only positive-slope crossings
            
            out1 = x(2)-zp ;    % event value
            out2 = isterminal ; % stop flag
            out3 = direction ;  % direction flag, +1 is incremental, -1 is decreasing
    end
end