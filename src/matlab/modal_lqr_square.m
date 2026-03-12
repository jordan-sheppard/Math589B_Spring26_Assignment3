function modal_lqr_square()
% Modal LQR control for a vibrating square membrane.
% Unit square, Dirichlet boundary conditions, point actuator.

c = 1.0;
M = 6;
x0 = 0.37;
y0 = 0.61;
alpha = 1.0;
beta_v = 1.0;
R = 5e-2;

modes = zeros(M*M, 2);
lambda = zeros(M*M, 1);
beta = zeros(M*M, 1);
idx = 1;
for m = 1:M
    for n = 1:M
        modes(idx, :) = [m, n];
        lambda(idx) = pi^2 * (m^2 + n^2);
        beta(idx) = 2 * sin(m*pi*x0) * sin(n*pi*y0);
        idx = idx + 1;
    end
end

omega2 = c^2 * lambda;
N = M*M;
A = [zeros(N), eye(N); -diag(omega2), zeros(N)];
B = [zeros(N,1); beta];
Q = [alpha * diag(omega2), zeros(N); zeros(N), beta_v * eye(N)];
K = lqr(A, B, Q, R);

q0 = zeros(N,1);
p0 = zeros(N,1);
q0(mode_index(modes, 1, 1)) = 0.8;
q0(mode_index(modes, 2, 1)) = 0.3;
q0(mode_index(modes, 1, 3)) = -0.25;
X0 = [q0; p0];

f = @(t, X) (A - B*K) * X;
T = 6.0;
[t, X] = ode45(f, [0, T], X0);
X = X.';
q = X(1:N, :);
p = X(N+1:end, :);
energy = 0.5 * sum(p.^2 + omega2 .* q.^2, 1);
control = -(K * X);

fprintf('Initial energy: %.6e\n', energy(1));
fprintf('Final energy:   %.6e\n', energy(end));
fprintf('Max |control|:  %.6e\n', max(abs(control)));

figure;
plot(t, energy, 'LineWidth', 1.5);
xlabel('t');
ylabel('truncated energy');
title('Closed-loop energy decay');
grid on;

figure;
plot(t, control, 'LineWidth', 1.2);
xlabel('t');
ylabel('control input b(t)');
title('LQR control input');
grid on;

end

function idx = mode_index(modes, m, n)
    idx = find(modes(:,1) == m & modes(:,2) == n, 1);
end
