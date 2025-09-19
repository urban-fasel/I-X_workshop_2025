function animateDuffing(t,x)

% generate animation

myVideo = VideoWriter('DuffingAnimation','MPEG-4'); % open video file
myVideo.FrameRate = 20; 
open(myVideo)

% Create figure
figure;
set(gcf,'Position',[75 75 450 350])
axis equal;
lims = 1;
axis([-lims lims -lims lims]);

set(gcf,'Color','w');

% Animation loop
numFrames = length(x); % number of frames
nF = 10; % plot every nF frame
for fr = 1:nF:numFrames
    subplot(2,1,1)
    p2 = plot(t,x(:,1),"Color",'b','LineWidth',1.2); hold on
    pp2 = plot(t(fr), x(fr,1), 'b*', 'linewidth', 6);
    ylim([-1.2 1.2])
    ylim([-3 3])
    ax = gca;
    ax.TickLabelInterpreter = "latex";
    ylabel('$x_1$', 'Interpreter','latex')
    set(gca,'XTickLabel',[]);

    subplot(2,1,2)
    % p3 = plot(t,180/pi*x(:,4),"Color",'r','LineWidth',1.2); hold on
    p3 = plot(t,x(:,2),"Color",'r','LineWidth',1.2); hold on
    pp3 = plot(t(fr), x(fr,2), 'r*', 'linewidth', 6);
    ylabel('$x_2$', 'Interpreter','latex')
    xlabel('time', 'Interpreter','latex')
    ylim([0 3])
    ylim([-3 3])
    ax = gca;
    ax.TickLabelInterpreter = "latex";

    % pause(0.05);

    frame = getframe(gcf); % get frame
    writeVideo(myVideo, frame);

    % delete(ll) 
    % delete(pp)
    delete(p2)
    delete(pp2)
    delete(p3)
    delete(pp3)
end

close(myVideo)
