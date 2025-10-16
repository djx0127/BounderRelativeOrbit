function vr = eqenergy(p,Hz,k,alpha,rou0,z0,omg,Energy)
A11 = -3*Hz^2/rou0^4-1/(sqrt(rou0^2+z0^2))^3+3*rou0^2/(sqrt(rou0^2+z0^2))^5;
A12 = 3*rou0*z0/(sqrt(rou0^2+z0^2))^5;
A22 = -1/(sqrt(rou0^2+z0^2))^3+3*z0^2/(sqrt(rou0^2+z0^2))^5;
A = [A11,A12;A12,A22];
W = [0,omg;-omg,0];
C = [zeros(2),eye(2);A,W];
[V,D] = eig(A);

correc = 1 ;
x0 = V*[correc;0];
x0po = [x0(1);x0(2);0;0]*p;
rou = rou0 + x0po(1) ;
z = z0 + x0po(2);

rou = rou0 + x0po(1);  z = z0+x0po(2);
vr = Hz^2/(2*rou^2)-1/sqrt(rou^2+z^2)-k*z*cos(alpha)-k*rou*sin(alpha)-Energy;
end