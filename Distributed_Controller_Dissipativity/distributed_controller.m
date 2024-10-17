% Clear previous variables
clear;
clc;

% Define system matrices for subsystems A, B, and C (corresponding to 1,2,
% and 3 in the paper)
% D_xx<-> H_ij   B_i^(1) = I_2
% E_i <-> B_i^(2)

% Subsystem A
A_A = [1 0; -2 -3];
B_A = [1; 2];
C_A = [4 5]; % Output matrix for A
E_A = [1; 0]; % Disturbance matrix for A

% Subsystem B
A_B = [0 2; -1 -3];
B_B = [2; -1];
C_B = [5.5 -2]; % Output matrix for B
E_B = [1; 0]; % Disturbance matrix for B

% Subsystem C
A_C = [0 3; -2 -1];
B_C = [1; 3];
C_C = [7 10]; % Output matrix for C
E_C = [1; 0]; % Disturbance matrix for C


% Dimensions
n = 2; % State dimension for each subsystem

% Coupling terms
D_AB = 0.4*eye(n);  % Coupling between A and B
D_BC = 0.4*eye(n);  % Coupling between B and C


% SDP decision variables
Q_A = [2 1;1 2]; %sdpvar(n, n);  % Positive definite matrix for A
Q_B = [3 0.5; 0.5 3]; %sdpvar(n, n);  % Positive definite matrix for B
Q_C = [4 1; 1 3]; %sdpvar(n, n);  % Positive definite matrix for C

% Set epsilon values to 10^-3
epsilon_A = 1e-3;
epsilon_B = 1e-3;
epsilon_C = 1e-3;

K_AA = sdpvar(1,n,'full');
K_AB = sdpvar(1,n,'full');
K_BB = sdpvar(1,n,'full');
K_BA = sdpvar(1,n,'full');
K_BC = sdpvar(1,n,'full');
K_CC = sdpvar(1,n,'full');
K_CB = sdpvar(1,n,'full');


% Solver settings
options = sdpsettings('solver', 'bmibnb' , 'verbose', 0);  


%%
S1 = -(A_A' * Q_A + Q_A * A_A) - ...
    ((-Q_A * D_AB + Q_A * E_A * K_AA) + (-Q_A * D_AB + Q_A * E_A * K_AA)') - epsilon_A * eye(n);
M1 = S1;


constraints1 = [Q_A >= 1e-5 * eye(n), M1 >= 1e-5 * eye(n), C_A' == Q_A * B_A];
objective1 = 0; 

% Solve the SDP problem
sol1 = optimize(constraints1, objective1, options);
disp(sol1)


%%
M1_v = value(M1);
S2 = -(A_B' * Q_B + Q_B * A_B) - ...
    ((-Q_B * (D_AB + D_BC) + Q_B * E_B * K_BB) + (-Q_B * (D_AB + D_BC) + Q_B * E_B * K_BB)') - epsilon_B * eye(n);
F2 = (D_AB' * Q_A + K_AB' * E_A' * Q_A + Q_B * D_AB + Q_B * E_B * K_BA) * inv(M1_v) * (D_AB' * Q_A + K_AB' * E_A' * Q_A + Q_B * D_AB + Q_B * E_B * K_BA)';
M2 = S2 - F2;

constraints2 = [Q_B >= 1e-5 * eye(n), M2 >= 1e-5 * eye(n), C_B' == Q_B * B_B];
objective2 = 0; 


% Solve the SDP problem
sol2 = optimize(constraints2, objective2, options);
disp(sol2)


%%
M2_v = value(M2);
S3 = -(A_C' * Q_C + Q_C * A_C) - ...
   ((-Q_C * D_BC + Q_C * E_C * K_CC) + (-Q_C * D_BC + Q_C * E_C * K_CC)') - epsilon_C * eye(n);
F3 = (D_BC' * Q_B + K_BC' * E_B' * Q_B + Q_C * D_BC + Q_C * E_C * K_CB) * inv(M2_v) * (D_BC' * Q_B + K_BC' * E_B' * Q_B + Q_C * D_BC + Q_C * E_C * K_CB)';
M3 = S3 - F3;

constraints3 = [Q_C >= 1e-5 * eye(n), M3 >= 1e-5 * eye(n), C_C' == Q_C * B_C];
objective3 = 0;


% Solve the SDP problem
sol3 = optimize(constraints3, objective3, options);
disp(sol3)


disp(value(K_AA))
disp(value(K_AB))
disp(value(K_BB))
disp(value(K_BA))
disp(value(K_BC))
disp(value(K_CC))
disp(value(K_CB))


%%
% Simulate the closed-loop system with noise and coupling terms using controller gains obtained from SDP
% Time span for simulation
tspan = [0 8];

% Initial conditions for the states of subsystems A, B, and C
x0_A = [1; 1];  % Initial state for subsystem A
x0_B = [1; -1];  % Initial state for subsystem B
x0_C = [-1; 1];  % Initial state for subsystem C

% Define the noise level (standard deviation of the noise)
noise_level = 0.1;


% Use ode45 to simulate the system with noise and coupling terms
[t, x] = ode45(@(t, x) closed_loop_dynamics_with_noise_and_coupling(t, x, value(K_AA), value(K_AB), value(K_BA), value(K_BB), value(K_BC), value(K_CC), value(K_CB), A_A, B_A, A_B, B_B, A_C, B_C, E_A, E_B, E_C, D_AB, D_BC, noise_level), tspan, [x0_A; x0_B; x0_C]);

% Plot the states of the subsystems over time
figure;
subplot(3,1,1);
plot(t, x(:, 1:2));  % Plot state of subsystem A
title('Subsystem A States with Noise and Coupling');
xlabel('Time');
ylabel('States');
legend('x1_A', 'x2_A');

subplot(3,1,2);
plot(t, x(:, 3:4));  % Plot state of subsystem B
title('Subsystem B States with Noise and Coupling');
xlabel('Time');
ylabel('States');
legend('x1_B', 'x2_B');

subplot(3,1,3);
plot(t, x(:, 5:6));  % Plot state of subsystem C
title('Subsystem C States with Noise and Coupling');
xlabel('Time');
ylabel('States');
legend('x1_C', 'x2_C');

% Display the final states
disp('Final states of the subsystems with noise and coupling:');
disp(['Subsystem A: ', num2str(x(end, 1:2))]);
disp(['Subsystem B: ', num2str(x(end, 3:4))]);
disp(['Subsystem C: ', num2str(x(end, 5:6))]);

% Function to represent the closed-loop dynamics with noise and coupling
function dxdt = closed_loop_dynamics_with_noise_and_coupling(t, x, K_AA, K_AB, K_BA, K_BB, K_BC, K_CC, K_CB, A_A, B_A, A_B, B_B, A_C, B_C, E_A, E_B, E_C, D_AB, D_BC, noise_level)
    % Extract the states of subsystems A, B, and C
    x_A = x(1:2); % State of subsystem A
    x_B = x(3:4); % State of subsystem B
    x_C = x(5:6); % State of subsystem C

    % Generate random noise for each subsystem
    w_A = noise_level * randn(1, 1);  % Noise for subsystem A
    w_B = noise_level * randn(1, 1);  % Noise for subsystem B
    w_C = noise_level * randn(1, 1);  % Noise for subsystem C

    % Closed-loop control inputs for each subsystem
    u_A = K_AA * x_A + K_AB * (x_B - x_A);  % Control input for A
    u_B = K_BB * x_B + K_BA * (x_A - x_B) + K_BC * (x_C - x_B);  % Control input for B
    u_C = K_CC * x_C + K_CB * (x_B - x_C);  % Control input for C


    % Closed-loop dynamics for each subsystem, including coupling and noise
    dx_A = A_A * x_A + B_A * u_A + D_AB * (x_B - x_A) + E_A * w_A;
    dx_B = A_B * x_B + B_B * u_B + D_AB * (x_A - x_B) + D_BC * (x_C - x_B) + E_B * w_B;
    dx_C = A_C * x_C + B_C * u_C + D_BC * (x_B - x_C) + E_C * w_C;

    % Combine the state derivatives into a single vector
    dxdt = [dx_A; dx_B; dx_C];
end