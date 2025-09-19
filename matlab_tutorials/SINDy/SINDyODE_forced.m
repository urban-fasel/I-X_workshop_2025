function dy = SINDyODE_forced(t,y,param)

% add forcing
gamma = param.paramDuffing(4);
omega = param.paramDuffing(5);
u = gamma*cos(omega*t);
y = [y; u];

yPool = poolData(y',length(y),param.polyorder);
dy = (yPool*param.Xi)';