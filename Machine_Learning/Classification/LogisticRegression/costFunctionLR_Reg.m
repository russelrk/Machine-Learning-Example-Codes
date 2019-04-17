function [J, grad] = costFunctionLR_Reg(X, y, theta)

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));


h_theta = X*theta;
len = length(theta);
h_theta = sigmoid(X*theta);
J = (1/m) * (-y.' * log(h_theta) - (1-y).' * log(1-h_theta)) + (lambda/(2*m)) * theta(2:len).' * theta(2:len);

theta_1 = theta;
theta_1(1) = 0;
grad = (1/m)*X.'*(h_theta - y) + (lambda/m)*theta_1;

end
