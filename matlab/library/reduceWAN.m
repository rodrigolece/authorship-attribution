function varargout = reduceWAN(mat)

if length(mat) ~= 211
    error('Error: please use 211 function words')
end

idx = idxAppearingFwords(mat);
out = mat(:, idx);
out = out(idx, :);

varargout{1} = out;

if nargout == 2
    varargout{2} = idx;
elseif nargout == 3
    varargout{2} = idx;
    edges = selectEdges(mat, 'Directed');
    varargout{3} = edges;
end    

end

function out = idxAppearingFwords(mat)

out = find(sum(mat,2));

end