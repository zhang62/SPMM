clear;
close;

addpath ~/workspace/MATLAB_tools/

% laplacian
% A = fd3d(8,16,8,0,0,0,0);
% B = fd3d(32,32,1,0,0,0,0);
% n = size(A,1);
% M = n;
% K = n;
% N = n;

M = 2010;
K = 2000;
N = 1990;
A = sprand(M, K, 0.02);
B = sprand(K, N, 0.02);

r = 10;
lambda = 16;

C = A * B;
rz_ext = spones(C) * ones(N,1);

rz_min = zeros(M,1);
rz_max = zeros(M,1);
for i=1:M
    jmax = 0;
    jsum = 0;
    col = find(A(i,:));
    ncol = length(col);
    for j=1:ncol
        bi = B(col(j),:);
        jmax = max(jmax, nnz(bi));
        jsum = jsum + nnz(bi);
    end
    rz_min(i) = jmax;
    rz_max(i) = min(N,jsum);
end

R0 = zeros(r, N);
R1 = zeros(r, K);
R2 = zeros(r, M);

R0 = rand(r, N);
R0 = -log(1-R0) / lambda;
% for i=1:N
%     ri = rand(r,1);
%     R0(:,i) = -log(1-ri) / lambda;
% end

for i=1:K
    ri = R0(:,find(B(i,:)));
    if (isempty(ri))
        R1(:,i) = inf(size(ri,1),1);
    else
        R1(:,i) = min(ri,[],2);
    end
end

for i=1:M
    ri = R1(:,find(A(i,:)));
    if (isempty(ri))
        R2(:,i) = inf(size(ri,1),1);
    else
        R2(:,i) = min(ri,[],2);
    end
end

rz = ((r-1) / lambda) ./ sum(R2);
% increase it a bit ?
% rz = rz * 1.05;
%
rz = round(rz);
rz = rz(:);
% simple adjustments
rz = max(rz, rz_min);
rz = min(rz, rz_max);


[rz_ext_sorted, p] = sort(rz_ext);
plot(rz_ext_sorted,'o');
hold on
plot(rz(p),'x')
plot(rz_max(p),'g--')
plot(rz_min(p),'r--')

fprintf('NNZ(C): real %d, min %d, max %d, est %d\n', ...
    nnz(C), sum(rz_min), sum(rz_max), sum(rz));

