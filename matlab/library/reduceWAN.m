function varargout = reduceWAN(mat)

if length(mat) ~= 211
    error('Error: please use 211 function words')
end

ids = idsAppearingFwords(mat);
out = mat(:, ids);
out = out(ids, :);

varargout{1} = out;

if nargout == 2
    varargout{2} = ids;
elseif nargout == 3
    varargout{2} = ids;
    edges = selectEdges(mat, 'Directed');
    varargout{3} = edges;
end    

end

function out = idsAppearingFwords(mat)

out = find(sum(mat,2));

end