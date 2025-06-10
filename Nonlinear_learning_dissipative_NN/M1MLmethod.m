%% Data and Initializaion
clear 
clc
load Results/weights_4164_3000_actv01_nolast.mat
alpha = 0.1; % a in LeakyReLU function
beta = 1;
p = alpha*beta;
m = (alpha+beta)/2;


%% Set variables
d1 = 4;
d2 = 16;
d3 = 4;

W1ini = reshape(W1,[d1,d2])';
W2ini = reshape(W2,[d2,d3])';



W1=sdpvar(d2,d1,'full');
l1=sdpvar;
assign(W1,W1ini)

W2=sdpvar(d3,d2,'full');	
l2=sdpvar;
assign(W2,W2ini)

% For Survey MSD
l = sdpvar;


assign(W1,W1ini)
assign(l1,1)
assign(W2,W2ini)
assign(l2,1)

options = sdpsettings('solver', 'penlab', 'verbose', 1);


 


%% 
% passivity
q = 0;
Q = q*eye(2);
r = 0;
R = r*eye(2);
S = 0.5*eye(2);
P11 = [Q S'; S R];
P12 = 0*eye(d1);
P22 = -0.001;
P22 = (P22)*eye(d3);



% No activation function in the last layer
% For Survey MSD
ML = [P11+l1*l*p*(W1')*W1 -l1*l*m*W1' P12';... 
    -l1*l*m*W1 l1*l*eye(d2)+l2*l*(W2')*W2 -l2*l*1/2*W2';...
    P12 -l2*l*1/2*W2 P22+l2*l*eye(d3)];

% It helps if we do not allow q, r, l1, l2 go too far or too small (numerical issues)
% For MSD, passivity
Constraints = [ML >= 1e-5, l1 >= 1, l2 >= 1, l >= 1]; 


Objective1 = sum(sum((W1-W1ini).^2));
Objective2 =  sum(sum((W2-W2ini).^2));
% For Survey MSD, prevent l1 l2 l from going too far
Objective = 400*Objective1+400*Objective2+l1^2+l2^2+l^2;
sol = optimize(Constraints,Objective,options);





%% Output 
if sol.problem == 0
 % Extract and display value
 value(W1)
 value(W2)
else
 disp('Something went wrong!');
 sol.info
 yalmiperror(sol.problem)
end


disp(value(sqrt(Objective1)))
disp(value(sqrt(Objective2)))

NN_dim = [4,16,4];

W1_adj = reshape(value(W1), d2*d1, 1);
W2_adj = reshape(value(W2), d3*d2, 1);
save Results/Survey_MSD/weights_adjusted.mat W1_adj W2_adj NN_dim



