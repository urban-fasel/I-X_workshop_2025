%% Identify a SINDy model on synthetic Lorenz system data
%% 
% * Lorenz system dynamics: 
%% 
% $$\begin{array}{l}\dot{x} =\sigma \;\left(y-x\right)\\\dot{y} =x\left(\rho 
% -z\right)-y\\\dot{z} =\textrm{xy}-\beta \;z\end{array}$$
%% 
% * Generate noisy data: integrate Lorenz system ODE and add Gaussian white 
% noise
% * Compute derivatives: using finite difference
% * Build library
% * Compute sparse regression
% * Evaluate performance of SINDy model for forecasting

% addpath(genpath(pwd))

%% Generate Data

param = [10; 28; 8/3]; % Lorenz system parameters (sigma, rho, beta)
n = 3; % number of states
x0 = [-8; 8; 27];  % Initial condition
dt = 0.001; % time step
tspan = dt:dt:20; 

% Lorenz system right hand side
lorenz = @(t,x,param) ...
    [param(1)*(x(2)-x(1)); ...
     x(1)*(param(2)-x(3)) - x(2); ...
     x(1)*x(2) - param(3)*x(3)];

options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,3));
[t,x]=ode45(@(t,x) lorenz(t,x,param),tspan,x0,options);

% Add Gaussian white noise
rng(1)
sig = 0.01; 
x = x + sig*std(x(:))*randn(size(x)); 

plotLorenz(t,x) % plot data


%% Compute Derivative: finite difference

% fourth order central difference
dx = (1/(12*dt))*(-x(5:end,:)+8*x(4:end-1,:)-8*x(2:end-3,:)+x(1:end-4,:)); 

x = x(3:end-2,:); % cut tails
t = t(3:end-2,:);


%% Pool Data (i.e., build library of nonlinear time series)

polyorder = 3; % polynomials up to order 3
Theta = poolData(x,n,polyorder);
m = size(Theta,2); % size of library


%% Compute sparse regression: sequential thresholded least squares

lambda = 0.2; % lambda is our sparsification knob.
Xi = sparsifyDynamics(Theta,dx,lambda,n); % identify model coefficients
disp(poolDataLIST({'x','y','z'},Xi,n,polyorder)) % display SINDy model


%% Use SINDy model for prediction

paramSINDy.Xi = Xi;
paramSINDy.polyorder = polyorder;
[tSINDy,xSINDy]=ode45(@(t,x) SINDyODE(t,x,paramSINDy),tspan,x0,options);

plotSINDy(t,x,tSINDy(3:end-2,:),xSINDy(3:end-2,:))




%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% functions

% build library
function yout = poolData(yin,nVars,polyorder)
n = size(yin,1);

ind = 1;
% poly order 0
yout(:,ind) = ones(n,1);
ind = ind+1;

% poly order 1
for i=1:nVars
    yout(:,ind) = yin(:,i);
    ind = ind+1;
end

% poly order 2
if(polyorder>=2)    
    for i=1:nVars
        for j=i:nVars
            yout(:,ind) = yin(:,i).*yin(:,j);
            ind = ind+1;
        end
    end
end

% poly order 3
if(polyorder>=3)    
    for i=1:nVars
        for j=i:nVars
            for k=j:nVars
                yout(:,ind) = yin(:,i).*yin(:,j).*yin(:,k);
                ind = ind+1;
            end
        end
    end
end
end


% STLS
function Xi = sparsifyDynamics(Theta,dXdt,lambda,n)
% Copyright 2015, All Rights Reserved
% Code by Steven L. Brunton
% For Paper, "Discovering Governing Equations from Data: 
%        Sparse Identification of Nonlinear Dynamical Systems"
% by S. L. Brunton, J. L. Proctor, and J. N. Kutz

% compute Sparse regression: sequential least squares
Xi = Theta\dXdt;  % initial guess: Least-squares

% lambda is our sparsification knob.
for k=1:10
    smallinds = (abs(Xi)<lambda);   % find small coefficients
    Xi(smallinds)=0;                % and threshold
    for ind = 1:n                   % n is state dimension
        biginds = ~smallinds(:,ind);
        % Regress dynamics onto remaining terms to find sparse Xi
        Xi(biginds,ind) = Theta(:,biginds)\dXdt(:,ind); 
    end
end
end

% SINDy model
function dy = SINDyODE(t,y,param)
yPool = poolData(y',length(y),param.polyorder);
dy = (yPool*param.Xi)';
end


% display active library terms
function newout = poolDataLIST(yin,ahat,nVars,polyorder)
n = size(yin,1);

ind = 1;
% poly order 0
yout{ind,1} = ['1'];
ind = ind+1;

% poly order 1
for i=1:nVars
    yout(ind,1) = yin(i);
    ind = ind+1;
end

if(polyorder>=2)
    % poly order 2
    for i=1:nVars
        for j=i:nVars
            yout{ind,1} = [yin{i},yin{j}];
            ind = ind+1;
        end
    end
end

if(polyorder>=3)
    % poly order 3
    for i=1:nVars
        for j=i:nVars
            for k=j:nVars
                yout{ind,1} = [yin{i},yin{j},yin{k}];
                ind = ind+1;
            end
        end
    end
end

output = yout;
newout(1) = {''};
for k=1:length(yin)
    newout{1,1+k} = [yin{k},'dot'];
end
% newout = {'','xdot','ydot','udot'};
for k=1:size(ahat,1)
    newout(k+1,1) = output(k);
    for j=1:length(yin)
        newout{k+1,1+j} = ahat(k,j);
    end
end
% newout
end


% plot Lorenz
function plotLorenz(t,x)

figure
% set(gcf,'Position',[75 75 900 450])
set(gcf,'Position',[75 75 450 750])

nP = 5;

subplot(nP,1,1)
plot(t,x(:,1),"Color",'b')
ylabel('x')
set(gca,'XTickLabel',[]);
subplot(nP,1,2)
plot(t,x(:,2),"Color",'r')
ylabel('y')
set(gca,'XTickLabel',[]);
subplot(nP,1,3)
plot(t,x(:,3),"Color",'g')
ylabel('z')
xlabel('time')


subplot(nP,1,[4,5])
plot3(x(:,1),x(:,2),x(:,3),'k-','LineWidth',1), hold on
% plot3(x(k,1),x(k,2),x(k,3),'r.','LineWidth',2,'MarkerSize',10), hold off
axis([-40 40 -40 40 -10 50])
view(-140,20);
% axis off
% axis equal
xlabel('x')
ylabel('y')
zlabel('z')
end


% plot SINDy
function plotSINDy(t,x,t2,x2)

figure
set(gcf,'Position',[75 75 450 750])

if size(x,2) == 2
    nP = 9;
    
    subplot(nP,1,1:2)
    plot(t2,x2(:,1),':','Color',[0 0 0]+0.6,'LineWidth',2.5); hold on
    plot(t,x(:,1),"Color",'b','LineWidth',1.2); hold on
    % plot(t2,x2(:,1),'k--')
    % ylabel('x1')
    ylabel('$x_1$', 'Interpreter','latex')
    legend({'SINDy $x_1$','Data $x_1$'}, 'Interpreter','latex')
    set(gca,'XTickLabel',[]);
    title('time series', 'Interpreter','latex')
    ax = gca;
    ax.TickLabelInterpreter = "latex";
    % ylim([-1.2 1.2])

    subplot(nP,1,3:4)
    plot(t2,x2(:,2),':','Color',[0 0 0]+0.6,'LineWidth',2.5); hold on
    plot(t,x(:,2),"Color",'r','LineWidth',1.2); hold on
    % plot(t2,x2(:,2),'k--')
    ylabel('$x_2$', 'Interpreter','latex')
    legend({'SINDy $x_2$','Data $x_2$'}, 'Interpreter','latex')
    xlabel('time', 'Interpreter','latex')
    axis tight
    ax = gca;
    ax.TickLabelInterpreter = "latex";
    % ylim([0 3])
    
    subplot(nP,1,6:nP)
    plot(x2(:,2),x2(:,1),':','Color',[0 0 0]+0.6,'LineWidth',2.5); hold on
    plot(x(:,2),x(:,1),'b','LineWidth',0.5), hold on
    % plot(x2(:,2),x2(:,1),'k--','LineWidth',0.5), hold on
    xlabel('$x_1$', 'Interpreter','latex')
    ylabel('$x_2$', 'Interpreter','latex')
    axis tight
    title('phase space')
    ax = gca;
    ax.TickLabelInterpreter = "latex";

else
    nP = 5;

    subplot(nP,1,1)
    plot(t,x(:,1),"Color",'b','LineWidth',1.2); hold on
    plot(t2,x2(:,1),'k--')
    ylabel('x')
    legend({'Data','SINDy'})
    set(gca,'XTickLabel',[]);
    subplot(nP,1,2)
    plot(t,x(:,2),"Color",'r','LineWidth',1.2); hold on
    plot(t2,x2(:,2),'k--')
    ylabel('y')
    set(gca,'XTickLabel',[]);
    subplot(nP,1,3)
    plot(t,x(:,3),"Color",'g','LineWidth',1.2); hold on
    plot(t2,x2(:,3),'k--')
    ylabel('z')
    xlabel('time')
    
    
    subplot(nP,1,[4,5])
    plot3(x(:,1),x(:,2),x(:,3),'b','LineWidth',0.5), hold on
    plot3(x2(:,1),x2(:,2),x2(:,3),'k--','LineWidth',0.5), hold on
    % plot3(x(k,1),x(k,2),x(k,3),'r.','LineWidth',2,'MarkerSize',10), hold off
    axis([-40 40 -40 40 -10 50])
    view(-140,20);
    % axis off
    % axis equal
    xlabel('x')
    ylabel('y')
    zlabel('z')
end
end