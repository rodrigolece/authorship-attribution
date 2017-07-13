function out = selectEdges(mat, type)

if type == 'Undirected'
    idx = find(triu(mat));
elseif type == 'Directed'
    idx = find(mat); % no need to use mat(:), find by default gives this behaviour
end

[i, j] = ind2sub(size(mat), idx);

out = [i, j, mat(idx)];

end