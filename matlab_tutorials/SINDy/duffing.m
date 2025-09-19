function xdot = duffing(t,x,param)

delta = param(1); 
alpha = param(2); 
beta  = param(3);
gamma = param(4);
omega = param(5);

xdot = [-delta*x(1)-alpha*x(2)-beta*x(2)^3+gamma*cos(omega*t);
        x(1);
        ];
