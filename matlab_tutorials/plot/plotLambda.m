function plotLambda(Theta,dx,lambda,n,m)

XiPath = [];
sumPath = [];

dP = 0.2; % plot factor non active terms

for i = 1:length(lambda)
    Xi = sparsifyDynamics(Theta,dx,lambda(i),n); % identify model coefficients
    
    nonzero = Xi ~= 0;
    nonzero = double(nonzero(:));
    sumPath(i) = sum(nonzero);
    nonzero(nonzero==0)=dP;

    nn(i) = norm(Theta*Xi-dx);

    % nonzero = [nonzero,nonzero]';
    % nonzero = nonzero(:);
    nonzero = [nonzero;nonzero(1)];

    XiPath(:,i) = nonzero; 
end


figure
set(gcf,'Position',[0 0 400 200])

nat = 7; % number of active terms

% subplot(2,1,1)
semilogx(lambda,sumPath,'k'), hold on
semilogx([lambda(2) lambda(end)],[nat nat],'b--'), hold on
% semilogx(lambda(i),sumPath(i),'r*'), hold off
ylim([0 m*n])
xlabel('\lambda')
ylabel('number of active terms')

legend({'SINDy path', 'Lorenz system number of terms'})

