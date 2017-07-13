function out = selectEdges(mat, type)

if strcmp(type, 'Undirected')
    idx = find(triu(mat));
elseif strcmp(type, 'Directed')
    idx = find(mat); % no need to use mat(:), find uses column vector by default
end

% idx = find(mat);

[i, j] = ind2sub(size(mat), idx);

out = [i, j, mat(idx)];

end