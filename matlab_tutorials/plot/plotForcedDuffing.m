function plotForcedDuffing(t,x,u)

np = 11;

figure;
% set(gcf,'Position',[75 75 450 350])
% axis equal;
% lims = 1;
% axis([-lims lims -lims lims]);

subplot(np,1,1:2)
% subplot(2,1,1)
plot(t,x(:,1),"Color",'b','LineWidth',1.2)
ylabel('$x_1$', 'Interpreter','latex')
title('time series', 'Interpreter','latex')
set(gca,'XTickLabel',[]);
ax = gca;
ax.TickLabelInterpreter = "latex";
% ylim([-1.2 1.2])

subplot(np,1,3:4)
% subplot(2,1,2)
plot(t,x(:,2),"Color",'r','LineWidth',1.2)
ylabel('$x_2$', 'Interpreter','latex')
% xlabel('time', 'Interpreter','latex')
set(gca,'XTickLabel',[]);
ax = gca;
ax.TickLabelInterpreter = "latex";
% ylim([0 3])

subplot(np,1,5:6)
% subplot(2,1,2)
plot(t,u,"Color",'g','LineWidth',1.2)
ylabel('forcing u', 'Interpreter','latex')
xlabel('time', 'Interpreter','latex')
ax = gca;
ax.TickLabelInterpreter = "latex";
% ylim([0 3])

subplot(np,1,8:np)
plot(x(:,2),x(:,1),'k')
title('phase space', 'Interpreter','latex')
xlabel('$x_2$', 'Interpreter','latex')
ylabel('$x_1$', 'Interpreter','latex')
ax = gca;
ax.TickLabelInterpreter = "latex";
