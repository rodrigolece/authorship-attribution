function [idx_union, idx_intersect] = unionIntersectIdx(varargin)

if nargin < 2
    disp('Please input more than two sets of indices')
    return
end

idx_union = union(varargin{1}, varargin{2});
idx_intersect = intersect(varargin{1}, varargin{2});

if nargin == 2
    return
end

for i = 3:nargin
    idx_union = union(idx_union, varargin{i});
    idx_intersect = intersect(idx_intersect, varargin{i});
end

end