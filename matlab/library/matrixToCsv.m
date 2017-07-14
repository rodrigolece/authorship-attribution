function matrixToCsv(filename, mat, type, idxFwords)

edges = selectEdges(mat, type);
% indices are relative to mat, but we want them relative to function words
edges(:,1) = idxFwords(edges(:,1));
edges(:,2) = idxFwords(edges(:,2));

save_edgefile(['../csv_files/', filename], edges, type)

end