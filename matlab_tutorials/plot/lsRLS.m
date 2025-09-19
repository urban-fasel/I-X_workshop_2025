function out = lsRLS(A,b,c)

% level sets
N = 50;
lims = 2;
xP = linspace(-lims,lims,N);
yP = linspace(-lims,lims,N);
[X,Y] = meshgrid(xP,yP);
for i = 1:N
    for j = 1:N
        Z(i,j) = norm(A*[X(i,j);Y(i,j)]-b,2);
    end
end
out.X = X;
out.Y = Y;
out.Z = Z;
out.R = (X.^2+Y.^2).^0.5;
out.L1 = abs(X)+abs(Y);
out.lims = lims;


% solve constraint opt

lb = [-lims -lims];
ub = [lims lims];
x0 = [0.3;0.4];
% c = 0.5;

options = optimoptions('lsqnonlin','display','off');
xRR = lsqnonlin(@(x)myfun(x,A,b),x0,lb,ub,[],[],[],[],@(x)nlcon1(x,c),options);
xLASSO = lsqnonlin(@(x)myfun(x,A,b),x0,lb,ub,[],[],[],[],@(x)nlcon2(x,c),options);

out.xRR = xRR;
out.xLASSO = xLASSO;

end

function F = myfun(x,A,b)
    F = norm(A*x-b);
end

function [c,ceq] = nlcon1(x,constr)
    ceq = [];
    c = sqrt(x(1)^2 + x(2)^2) - constr;
end

function [c,ceq] = nlcon2(x,constr)
    ceq = [];
    c = abs(x(1)) + abs(x(2)) - constr;
end