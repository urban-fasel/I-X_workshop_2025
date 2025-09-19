function plotSINDyPendulum(t,x,t2,x2)

figure
set(gcf,'Position',[75 75 450 350])

nP = 2;

subplot(nP,1,1)
plot(t2,x2(:,1),':','Color',[0 0 0]+0.6,'LineWidth',2.5); hold on
plot(t,x(:,1),"Color",'b','LineWidth',1.2); hold on
% plot(t2,x2(:,1),'k--')
ylabel('x')
legend({'SINDy','Data'})
set(gca,'XTickLabel',[]);

subplot(nP,1,2)
plot(t2,x2(:,2),':','Color',[0 0 0]+0.6,'LineWidth',2.5); hold on
plot(t,x(:,2),"Color",'r','LineWidth',1.2); hold on
% plot(t2,x2(:,2),'k--')
ylabel('xDot')
xlabel('time')
