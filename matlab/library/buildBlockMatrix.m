function block_matrix = buildBlockMatrix(text_structs)

no_fwords = 0;

for text = text_structs
    no_fwords = no_fwords + length(text.ids);
end

block_matrix = zeros(no_fwords);
% annotated_labels = [];
% book_vector = [];

start = 1;
% count = 1;

for text = text_structs
    % Block matrix in the correct position
    fwords_in_text = length(text.ids);
    end_idx = start + fwords_in_text - 1;
    block_matrix(start:end_idx, start:end_idx) = text.WAN;
    start = start + fwords_in_text;
    
%     % labels
%     annotated_labels = [annotated_labels; ...
%         strcat(allfwordsnodes(text.ids, 2), sprintf('-%i', count)) ];
%     % vector with book number
%     book_vector = [book_vector; count*ones(fwords_in_text, 1)];
%     
%     count = count + 1;
end

end