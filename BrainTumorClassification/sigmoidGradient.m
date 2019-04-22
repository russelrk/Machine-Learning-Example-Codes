function g = sigmoidGradient(z)

g = zeros(size(z));



sig = 1./(1+exp(z));
g = sig.*(1-sig);










% =============================================================




end
