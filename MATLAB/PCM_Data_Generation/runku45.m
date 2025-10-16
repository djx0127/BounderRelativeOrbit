function [t,xx]=runku45(hproc,t,xx)
% hproc is handle of function
% delt is time step
% t is time, xx is set of variables to be found
e=[0.5,0.5,1.0,1.0,0.5];
ta=t; xa=xx; xb=xx;
delt = 0.1;
for j=1:4
    xdot=feval(hproc,t,xb);
    t=ta+e(j)*delt;
    xb=xa+e(j)*xdot*delt;
    xx=xx+e(j+1)*xdot*delt/3;
end