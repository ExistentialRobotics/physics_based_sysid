function x = simulate_system(sys, u, x0, noise, delta) %system, input, initial condition, noise charaterization, noise bound
%simulation of discrete-time systems with noise

x = x0(:); %initial condition
N = length(u); %experiment length

%-------------------------system 1--------------------------- 
% Mass-spring-damper system
if (sys == 1)
    T = 0.5; %sampling time
    m = 1; %mass
    k = 1; %spring
    d = 1; %damper
    dynamic = @(x,u) [1 T; -k/m*T -d/m*T+1]*x+[0; T]*u; %system dynamics
end

%-------------------------noise-free--------------------------- 
if (noise==0)
    %simulation
    for j = 1:N
        x = [x dynamic(x(:,end),u(j))];
    end
end

%-------------------------noise--------------------------- 
if (noise == 1) %process noise bounded by infty-norm  
    noise_signal = 2*diag(delta)*(rand(length(x0),N+1)-0.5); %noise realization
    %simulation
    for j = 1:N
        x = [x dynamic(x(:,end),u(j))+noise_signal(:,j)];
    end
end

if (noise == 2) %process noise bounded by Euclidean-norm
    %noise realization
    noise_signal = 2*(rand(length(x0),N+1)-0.5); %random direction
    for i = 1:length(noise_signal(1,:))
        noise_signal(:,i) = noise_signal(:,i)/norm(noise_signal(:,i))*delta*rand(1); %bounded but random amplitude
    end
    %simulation
    for j = 1:N
        x = [x dynamic(x(:,end),u(j))+noise_signal(:,j)];
    end
end

end

