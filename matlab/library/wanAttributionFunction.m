function count = wanAttributionFunction(unknown_structs, candidate_structs,...
    author_no)

no_candidates = length(candidate_structs);
if no_candidates < 2
    error('Error: mumber of candidates be must at least 2')
end

count = 0;

for unknown = unknown_structs
    
    scores = zeros(no_candidates,1);

    for i =1:no_candidates
        cand = candidate_structs(i);

        scores(i) = norm(cand.WAN(:) - unknown.WAN(:),1)/70^2;
%         scores(i) = dot( cand.WAN(:), unknown.WAN(:) ) / ( norm(cand.WAN(:))*norm(unknown.WAN(:)) );
%         disp(num2str(scores(i)))
    end

    [~, idx] = min(scores);
    if idx == author_no
        count = count + 1;
    end
    
end

end