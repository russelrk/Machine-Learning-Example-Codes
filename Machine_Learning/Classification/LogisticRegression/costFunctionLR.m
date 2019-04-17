function [jVal, gradient] = costFunctionLR(x, y, theta)

m = length(y);
h_theta = 1./(1+exp(-x*theta));
jVal = (1/m)*(-y.'*log(h_theta)-(1-y).'*log(1-h_theta));
gradient = (1/m)*x.'*(h_theta - y);

end
