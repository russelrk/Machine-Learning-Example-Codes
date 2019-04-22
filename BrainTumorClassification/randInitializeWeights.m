function W = randInitializeWeights(L_in, L_out)

% You need to return the following variables correctly 
W = zeros(L_out, 1 + L_in);




IEP = 1e-2;

W = rand(size(W))*IEP*2 - IEP;




% =========================================================================

end
