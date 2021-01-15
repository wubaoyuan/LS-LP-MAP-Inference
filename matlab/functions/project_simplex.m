%% Projection a D-dimensional vector into the probability simplex

function X = project_simplex(Y)
[N,D] = size(Y);
X = sort(Y,2, 'descend');
Xtmp = (cumsum(X,2)-1)*diag(sparse(1./(1:D)));
X = max(bsxfun(@minus,Y,Xtmp(sub2ind([N,D],(1:N)',sum(X>Xtmp,2)))),0);
