%% Simulation settings
t_end = 5.3;
sampling_freq = 100;
tspan = linspace(0, t_end, sampling_freq*t_end+1);
x_y_space = [-4, 4; -4, 4];
n1 = 3;
n2 = 4;
n3 = 10;
n_state = 2;

x1_1 = linspace(x_y_space(1,1), -2, n1);
x1_2 = linspace(-2, -1, n2);
x1_3 = linspace(-1, 1, n3);
x1_4 = linspace(1, 2, n2);
x1_5 = linspace(2, x_y_space(1,2), n1);
x1 = [x1_1(1:end-1), x1_2(1:end-1), x1_3(1:end-1), x1_4(1:end-1), x1_5];

x2_1 = linspace(x_y_space(2,1), -2, n1);
x2_2 = linspace(-2, -1, n2);
x2_3 = linspace(-1, 1, n3);
x2_4 = linspace(1, 2, n2);
x2_5 = linspace(2, x_y_space(2,2), n1);
x2 = [x2_1(1:end-1), x2_2(1:end-1), x2_3(1:end-1), x2_4(1:end-1), x2_5];

[X1,X2] = ndgrid(x1,x2);
initial_states = zeros(size(x1,2)*size(x2,2),n_state);
initial_states(:,1) = reshape(X1,[],1);
initial_states(:,2) = reshape(X2,[],1);