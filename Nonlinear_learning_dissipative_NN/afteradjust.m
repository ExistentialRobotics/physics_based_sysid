clear 
clc
load Results/weights_adjusted.mat
% can also test the model before weights adjustment 
load Results/weights_4164_3000_actv01_nolast.mat 
% b1,b2 for baseline; b1_retrained b2_trained for the final model 
load Results/biases_4164_3000_actv01_nolast.mat 
load Results/retrained_biases_4164_4000_actv01_nolast.mat

% b1 = zeros(16,1);
% b2 = [0;0;0;0];
% Be careful with unrolling and rolling! matlab and python are different
W1_adj = reshape(W1_adj,16,4);
W2_adj = reshape(W2_adj,4,16);
% For initially trained weights. The one from pytorch
W1_base = reshape(W1,4,16)';
W2_base = reshape(W2,16,4)';

tspan = [0 15];
y0 = [0.1;0.1;1;0];


% Survey mass spring damper
tspan = [0 15];
y0 = [0.1;0.1;1;0];

h = figure;
plot(y0(1),y0(2),'.','MarkerSize',15, 'HandleVisibility', 'off')
hold on

[t,y_true_MSD] = ode45(@(t,y) MSD_func(t,y), tspan, y0);
[t,y_base] = ode45(@(t,y) NN_func(t,y, W1_base, b1, W2_base,b2), tspan, y0);
[t,y_adj] = ode45(@(t,y) NN_func(t,y, W1_adj,b1,W2_adj,b2), tspan, y0);
[t,y_final] = ode45(@(t,y) NN_func(t,y, W1_adj,b1_retrained,W2_adj,b2_retrained), tspan, y0);


plot(y_true_MSD(:,1), y_true_MSD(:,2),'Color',[0.8627, 0.0784, 0.2353],'LineWidth',1.2);
hold on 
plot(y_base(:,1), y_base(:,2),'Color', [0.1804, 0.5451, 0.3412],'LineWidth',1.2);
plot(y_adj(:,1), y_adj(:,2),'Color', [0.2549, 0.4118, 0.8824],'LineWidth',1.2);
plot(y_final(:,1), y_final(:,2),'Color', [0.8549, 0.6471, 0.1255],'LineWidth',1.2);
hold off

legend('Ground Truth', 'Baseline', 'Model after Weights Perturbation', 'Final Model', ...
    'Location','best');

xlabel('x_1')
ylabel('x_2')
title('Mass Spring Damper Example')



function yout = NN_func(t, y, W1_,b1_,W2_,b2_)
    x0 = y;
    v1 = W1_*x0+b1_;
    v1(v1<=0) = 0.1*v1(v1<=0);
    x1 = v1;
    v2 = W2_*x1+b2_;
    yout = v2;
end




% for MSD example
function yout = MSD_func(t,y)

        x1 = y(1);
        x2 = y(2);
        u = y(3);
        a = x2;
        b = -x2-x1-x1^3+u;

        c = 0;
        d = 0;

        yout = [a;b;c;d];

end



