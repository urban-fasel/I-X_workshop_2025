function plotRR(xL,maxMSE,c,in)

X = in.X;
Y = in.Y;
Z = in.Z;
R = in.R;
lims = in.lims;

th = -pi:0.001:pi;

zlevs = 0:0.25:maxMSE;
dx = 0.05; % delta x text
dP = 0.5; % delta ticks

figure
contour(X,Y,Z,zlevs); hold on
plot(xL(1),xL(2),'r*')
text(xL(1)+dx,xL(2),'OLS')
plot(sin(th)*c,cos(th)*c,'k')
fill(sin(th)*c,cos(th)*c,'k','FaceAlpha',0.1)
xlabel('$x_1$','Interpreter','latex')
ylabel('$x_2$','Interpreter','latex')

% colorbar('$||\\mathbf{b-Ax}||_2^2$','interpreter','latex');
cb = colorbar;
cb.Label.String = sprintf('$||\\mathbf{b-Ax}||_2^2$');
cb.Label.Interpreter = 'latex';
clim([0 2.5])
cb.Position = [.85 .58 .02 .3];
cb.Ticks = 0:0.5:2.5;

% arrow
th1 = 7/8*pi;
startpoint = [0 0];    
endpoint = [cos(th1)*c sin(th1)*c];   
headsize = 0.5;
v1 = headsize*(startpoint-endpoint)/2.5;
theta = 22.5*pi/180;
theta1 = -1*22.5*pi/180;
rotMatrix = [cos(theta)  -sin(theta) ; sin(theta)  cos(theta)];
rotMatrix1 = [cos(theta1)  -sin(theta1) ; sin(theta1)  cos(theta1)];
v2 = v1*rotMatrix;
v3 = v1*rotMatrix1;
x1 = endpoint;
x2 = x1 + v2;
x3 = x1 + v3;
hold on;
fill([x1(1) x2(1) x3(1)],[x1(2) x2(2) x3(2)],[0 0 0]);     % this fills the arrowhead (black)
plot([startpoint(1) endpoint(1)],[startpoint(2) endpoint(2)],'linewidth',1,'color',[0 0 0]);
dA = 0.4;
text(endpoint(1)*(1-dA),endpoint(2)*(1-dA)+0.1,sprintf('$$c$$'),'Interpreter', 'latex', 'fontsize', 12)

xline(0)
yline(0)
axis equal
xticks(-lims:dP:lims)
yticks(-lims:dP:lims)
xlim([-lims lims])
ylim([-lims lims])
grid on

% text(-1.6,-0.8,sprintf('$ \\mathrm{min} ||Ax-b||_2^2 \\ \\ \\mathrm{s.t.} \\ \\ ||x||_2 <$ %.2f', c),'Interpreter', 'latex', 'fontsize', 12)
title(sprintf('$ \\mathrm{Ridge:} \\  \\mathrm{argmin}_\\mathbf{x} ||\\mathbf{b-Ax}||_2^2 \\ \\ \\mathrm{s.t} \\ \\ ||\\mathbf{x}||_2^2 <$ %.2f', c),'Interpreter', 'latex', 'fontsize', 12)

minR = min(min(R(Z<maxMSE)));
if minR < c
    colorT = [0 0.7 0];
    colorT2 = [0.6 0.9 0.6];
    % plot(in.x(1),in.x(2),'-s','MarkerSize',10,...
    % 'MarkerEdgeColor','red',...
    % 'MarkerFaceColor',[1 .6 .6])
    plot(in.xRR(1),in.xRR(2),'-s','MarkerSize',10,...
    'MarkerEdgeColor',colorT,...
    'MarkerFaceColor',colorT2)
    % text(in.xRR(1)+0.1,in.xRR(2),'Ridge')

    % text(0.3,-1.2,sprintf('$ \\hat\\mathbf{x} = $[ %.2f; %.2f]', in.xRR(1),in.xRR(2)),'Interpreter', 'latex', 'fontsize', 12)

    text(in.xRR(1)+0.1,in.xRR(2),sprintf('$ \\hat\\mathbf{x} = $[ %.2f; %.2f]', in.xRR(1),in.xRR(2)),'Interpreter', 'latex', 'fontsize', 12)
else
    colorT = 'r';
end
text(-1.65,-1.2,sprintf('$ ||\\mathbf{b-Ax}||_2^2 $ = %.2f', maxMSE),'Interpreter', 'latex', 'fontsize', 12)
text(-1.65,-1.6,sprintf('$ ||\\mathbf{x}||_2^2 = $ %.2f', minR),'Interpreter', 'latex','Color',colorT, 'fontsize', 12)

