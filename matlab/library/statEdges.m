function varargout = statEdges(edges)

weights = edges(:,3);
mn = mean(weights);
med = median(weights);

if nargout == 0 % we plot the  histogram of weights

    h = histogram(weights);
    counts = h.Values;
    center_counts = h.BinEdges(1:end-1) + 0.5*h.BinWidth;
    close

    figure()
    loglog(center_counts, counts)
    title(['mean: ', num2str(mn, 4), '; median: ', num2str(med, 4)], ...
        'FontSize', 15)
elseif nargout == 2
    varargout{1} = mn;
    varargout{2} = med;
end

end

