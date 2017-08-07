function out = ABsimilarityBlockMean(mat)

N = length(mat)/70;

out = zeros(N);

for i = 1:N
    idx_i = (i-1)*70+1:70*i;
    for j = 1:N
        idx_j = (j-1)*70+1:70*j;
        block = mat(idx_i, idx_j);
        out(i,j) = mean(block(:));
    end
end

end