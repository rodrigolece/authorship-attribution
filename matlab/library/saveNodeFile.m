function saveNodeFile(file, nodes, book, varargin)

out = fopen(file, 'w');

num_partitions = nargin -3;
labels = 'Id,Label,Book';

for i = 1:num_partitions
    labels = [labels, sprintf(',Partition%i', i)];
end
labels = [labels, '\n'];

fprintf(out, labels);

for k=1:length(nodes)
    row = sprintf('%i,%s,%i', k, nodes{k}, book(k));
    for i = 1:num_partitions
        row = [row, sprintf(',%i', varargin{i}(k))];
    end
    row = [row, '\n'];
    fprintf(out, row);
end

fclose(out);
fprintf('\n%s written\n', file)