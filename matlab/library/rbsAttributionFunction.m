function count = rbsAttributionFunction(unknown_structs, candidate_structs,...
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

        bm = buildBlockMatrix([unknown, cand]);
        Y = rbs(bm, 0.95);
        Y = 0.5*(Y + Y');
        od = Y(1:70, 71:end);

%         similarity = ABsimilarityBlockMean(Y);
%         scores(i) = similarity(1, 2);
        scores(i) = trace(od)/70;
    %     disp(num2str(scores(i)))
    end

    [~, idx] = max(scores);
    if idx == author_no
        count = count + 1;
    end
    
end

end



  
    
