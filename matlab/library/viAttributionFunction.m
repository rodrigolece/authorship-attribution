function count = viAttributionFunction(unknown_structs, candidate_structs,...
    author_no)

no_candidates = length(candidate_structs);
if no_candidates < 2
    error('Error: mumber of candidates be must at least 2')
end

count = 0;

for unknown = unknown_structs

    scores = zeros(no_candidates,1);

    for i = 1:no_candidates
        cand = candidate_structs(i);

        [~, vi_mat] = varinfo([cand.part, unknown.part]');
        vi_mat = vi_mat(1:6, 7:end);

%         scores(i) = trace(vi_mat);
        scores(i) = mean(vi_mat(:));
        if length(unknown_structs) == 1
            disp(num2str(scores(i)))
        end
    end

    [~, idx] = min(scores);
    if idx == author_no
        count = count + 1;
    end
    
end