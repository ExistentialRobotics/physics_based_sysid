clear 
clc

rng(123)

% Define the circuit parameters
R = 1; % Resistance in ohms
L = 1; % Inductance in henries
C = 1; % Capacitance in farads

% Define the state-space matrices
A = [-R/L -1/L; 1/C 0];
B = [1/L; 0];
C = [1 0];
D = 0; 

sys_original = ss(A, B, C, D);

% Ensure D is symmetric for the check
D_sym = D + D';

% Set up the LMI problem using YALMIP
% Define symbolic variables
P = sdpvar(2,2); % P is a 2x2 symmetric matrix

% Define the LMI
LMI = [A'*P + P*A, P*B - C'; B'*P - C, -(D_sym)];

% Constraints for positive definite P and the LMI
constraints = [P >= 0, LMI <= 0];

% Set options for the solver
options = sdpsettings('solver', 'sedumi', 'verbose', 0);

% Solve the LMI
diagnostics = optimize(constraints, [], options);

% Check if the solution is feasible
if diagnostics.problem == 0
    disp('The LMI is feasible, so the system is positive real and dissipative.');
    P_value = value(P);
    disp('P matrix:');
    disp(P_value);
else
    disp('The LMI is not feasible, so the system is not positive real.');
    disp('Solver diagnostics:');
    disp(diagnostics.info);
end


%% Generate data and identify it
sys = ss(A, B, C, D);

% Display the state-space model
disp('Original State-Space Model:');
disp(sys);

% Define the time vector for simulation
t = 0:0.01:10;


u_sine = (exp(-0.1*t).*sin(2*pi*0.5*t))';


% Initial states
x0 = [0.1 0.2];

[y_sine, ~, ~] = lsim(sys, u_sine, t, x0);


% Add noise to the output data, std = 0.005
noise_level = 0.005;
y_sine_noisy = y_sine + noise_level * randn(size(y_sine));

% Plot the original and noisy output for one of the inputs 
figure;
plot(t, y_sine, 'b', t, y_sine_noisy, 'r');
title('System Output with Noise');
xlabel('Time (s)');
ylabel('Output');
legend('Original Output', 'Noisy Output');

% Prepare data for system identification
data_sine = iddata(y_sine_noisy, u_sine, 0.01);

% Combine the datasets
data_combined = data_sine;

% Identify the state-space model
sys_identified = ssest(data_combined, 2); % 2 is the order of the system

% Display the identified state-space model
disp('Identified State-Space Model:');
disp(sys_identified);

% Compare the original and identified system
figure;
compare(data_combined, sys, sys_identified);

%% Check if the system identified is dissipative
% Extract the state-space matrices from the identified system
A_identified = sys_identified.A;
B_identified = sys_identified.B;
C_identified = sys_identified.C;
D_identified = sys_identified.D;

% Ensure D is symmetric for the check
D_sym = D_identified + D_identified';

% Set up the LMI problem using YALMIP
% Define symbolic variables
P = sdpvar(size(A_identified, 1), size(A_identified, 2)); % P is a symmetric matrix

% Define the LMI
LMI = [A_identified'*P + P*A_identified, P*B_identified - C_identified'; B_identified'*P - C_identified, -D_sym];

% Constraints for positive definite P and the LMI
constraints = [P >= 0, LMI <= 0];

% Set options for the solver
options = sdpsettings('solver', 'sedumi', 'verbose', 0);

% Solve the LMI
diagnostics = optimize(constraints, [], options);

% Check if the solution is feasible
if diagnostics.problem == 0
    disp('The LMI is feasible, so the identified system is positive real and dissipative.');
    P_value = value(P);
    disp('P matrix:');
    disp(P_value);
else
    disp('The LMI is not feasible, so the identified system is not positive real.');
    disp('Solver diagnostics:');
    disp(diagnostics.info);
end



%% Perturb C_identified to make the system passive 
% Ensure D is symmetric for the check
D_sym = D_identified + D_identified';

% Define perturbation variable for C
Delta_C = sdpvar(size(C_identified, 1), size(C_identified, 2), 'full');

% Perturbed C matrix
C_perturbed = C_identified + Delta_C;

% Set up the LMI problem using YALMIP
% Define symbolic variables
P = sdpvar(size(A_identified, 1), size(A_identified, 2)); % P is a symmetric matrix

% Define the LMI with perturbed C
LMI = [A_identified'*P + P*A_identified, P*B_identified - C_perturbed'; B_identified'*P - C_perturbed, -D_sym];

% Constraints for positive definite P and the LMI
constraints = [P >= 0, LMI <= 0];

% Objective function to minimize the Frobenius norm of Delta C
objective = norm(Delta_C, 'fro');

% Set options for the solver
options = sdpsettings('solver', 'sedumi', 'verbose', 0);

% Solve the optimization problem
diagnostics = optimize(constraints, objective, options);

% Check if the solution is feasible and output the result
if diagnostics.problem == 0
    disp('The LMI is feasible, so the perturbed system is positive real and dissipative.');
    P_value = value(P);
    Delta_C_value = value(Delta_C);
    C_perturbed_value = value(C_perturbed);
    disp('P matrix:');
    disp(P_value);
    disp('Delta C (perturbation) matrix:');
    disp(Delta_C_value);
    disp('Perturbed C matrix:');
    disp(C_perturbed_value);
else
    disp('The LMI is not feasible, so the perturbed system is not positive real.');
    disp('Solver diagnostics:');
    disp(diagnostics.info);
end


% Check the performance after perturbation
sys_perturbed = ss(A_identified, B_identified, C_perturbed_value, D_sym);

% Simulate the system response for the original and perturbed systems
[y_original, ~, ~] = lsim(sys_original, u_sine, t);
[y_perturbed, ~, ~] = lsim(sys_perturbed, u_sine, t);

% Plot the original and perturbed system responses
figure;
plot(t, y_original, 'b', t, y_perturbed, 'r--');
title('System Response Comparison');
xlabel('Time (s)');
ylabel('Output');
legend('Original System', 'Perturbed System');



[~, fit_original, ~] = compare(data_sine, sys_original);
[~, fit_perturbed, ~] = compare(data_sine, sys_perturbed);

% Display the fit percentages
disp(['Fit percentage for the original system: ', num2str(fit_original), '%']);
disp(['Fit percentage for the perturbed system: ', num2str(fit_perturbed), '%']);














