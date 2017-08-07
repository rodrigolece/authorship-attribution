function out = rbsAttributionFunction(unknown, candidate_structs)

no_candidates = length(candidate_structs);
if no_candidates < 2
    error('Error: mumber of candidates be must at least 2')
end

scores = zeros(no_candidates,1);

for i =1:no_candidates
    cand = candidate_structs(i);
    
    bm = buildBlockMatrix([unknown, cand]);
    Y = rbs(bm, 0.95);
    Y = 0.5*(Y + Y');
    
    similarity = ABsimilarityBlockMean(Y);
    scores(i) = similarity(1, 2);
    disp(num2str(scores(i)))
end

[~, out] = max(scores);

end