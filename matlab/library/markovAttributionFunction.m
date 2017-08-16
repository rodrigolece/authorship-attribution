function count = markovAttributionFunction(unknown_structs, candidate_structs,...
    author_no)

no_candidates = length(candidate_structs);
if no_candidates < 2
    error('Error: mumber of candidates be must at least 2')
end
j = ones(70, 1);

count = 0;

for unknown = unknown_structs
    scores = zeros(no_candidates,1);

    d_u = diag(1./(unknown.WAN*j));
    markov_u = d_u * unknown.WAN;

    for i =1:no_candidates
        cand = candidate_structs(i);
        d_cand = diag(1./(cand.WAN*j));
        markov_cand = d_cand * cand.WAN;

        scores(i) = norm(markov_cand(:) - markov_u(:),1);
%         scores(i) = dot( cand.WAN(:), unknown.WAN(:) ) / ( norm(cand.WAN(:))*norm(unknown.WAN(:)) );
%         disp(num2str(scores(i)))
    end

    [~, idx] = min(scores);
    if idx == author_no
        count = count + 1;
    end
    
end

end