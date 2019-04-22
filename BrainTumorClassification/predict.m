function p = predict(Theta1, Theta2, Theta3, X)

% Useful values
m = size(X, 1);
num_labels = size(Theta3, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
h3 = sigmoid([ones(m, 1) h2] * Theta3.');
p = round(h3);

% =========================================================================


end
