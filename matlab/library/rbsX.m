function [x, xt] = rbsX(A, alpha, kmax)

N = length(A);

lmb = eigs(A,1);
beta = alpha/lmb;

current_A = beta * A;
current_At = beta * A';

j = ones(N,1);

x = zeros(N, kmax);
x(:,1) = current_A * j;
xt = zeros(N, kmax);
xt(:,1) = current_At * j;

for k = 2:kmax
    current_A = beta * A * current_A;
    current_At = beta * A' * current_At;
    
    x(:,k) = current_A * j;
    xt(:,k) = current_At * j;
end