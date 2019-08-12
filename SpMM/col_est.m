clear;
close;

addpath ~/workspace/MATLAB_tools/

% A = fd3d(8,8,4,0,0,0,0);
% B = fd3d(16,16,1,0,0,0,0);
% n = size(A,1);
% M = n;
% K = n;
% N = n;

M = 210;
K = 200;
N = 190;
r = 100;

A = sprand(M, K, 0.1);
B = sprand(K, N, 0.1);
C = A * B;

R0 = zeros(r, M);
R1 = zeros(r, K);
R2 = zeros(r, N);

lambda = 1;
for i=1:M
    ri = rand(r,1);
    R0(:,i) = -log(1-ri) / lambda;
end

for i=1:K
    ri = R0(:,find(A(:,i)));
    R1(:,i) = min(ri,[],2);
end

for i=1:N
    ri = R1(:,find(B(:,i)));
    R2(:,i) = min(ri,[],2);
end

rz = ((r-1) / lambda) ./ sum(R2); 
rz = rz(:);

rz_ext = ones(1,M) * spones(C);
rz_ext = rz_ext(:);

plot(rz_ext,'--o');
hold on
plot(rz,'--x')