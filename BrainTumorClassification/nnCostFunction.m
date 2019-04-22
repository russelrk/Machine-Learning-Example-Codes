function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
len1 = hidden_layer_size(1) * (input_layer_size + 1);
Theta1 = reshape(nn_params(1:len1), hidden_layer_size(1), (input_layer_size + 1));

len2 = len1+ hidden_layer_size(2) * (hidden_layer_size(1) + 1);
Theta2 = reshape(nn_params(1 + len1:len2), hidden_layer_size(2), (hidden_layer_size(1) + 1));

Theta3 = reshape(nn_params(1+len2:end), num_labels, (hidden_layer_size(2) + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));


    a1 = [ones(m, 1) X];
    z2 = Theta1*a1.';
    a2 = sigmoid(z2);
    
    a2 = [ones(m, 1) a2.'];
    z3 = Theta2*a2.';
    a3 = sigmoid(z3);
    
    a3 = [ones(m, 1) a3.'];
    z4 = Theta3*a3.';
    a4 = sigmoid(z4);
    
    y1 = zeros(num_labels, m);
    for i = 1:m
        y1(i) = y(i);
    end
    
    J = -(1/m)*(y1(:).' * log(a4(:)) + (1-y1(:)).' * log(1-a4(:)));
    
    t1 = Theta1(:, 2:size(Theta1, 2));
    t2 = Theta2(:, 2:size(Theta2, 2));
    t3 = Theta3(:, 2:size(Theta3, 2));
    
    t1 = t1(:);
    t2 = t2(:);
    t3 = t3(:);
    reg = lambda/(2*m) * (t1.' * t1 + t2.' * t2 + t3.' * t3);
    
    J = J+reg;

    
for i = 1:m,
    
    a1 = X(i, :);
    a1 = [1 a1];
    z2 = Theta1 * a1.';
    a2 = sigmoid(z2).';
    
    a2 = [1 a2];
    z3 = Theta2 * a2.';
    a3 = sigmoid(z3).';
    
    a3 = [1 a3];
    z4 = Theta3 * a3.';
    a4 = sigmoid(z4);
    
    del4 = a4-y1(:, i);
    
    del3 = (Theta3.'*del4) .* sigmoidGradient([1; z3]);

    del2 = (Theta2.'*del3(2:end)) .* sigmoidGradient([1; z2]);
    
    Theta3_grad = Theta3_grad + del4 * a3;
    Theta2_grad = Theta2_grad + del3(2:end) * a2;
    Theta1_grad = Theta1_grad + del2(2:end) * a1;
    
end

Theta1_grad = (1/m) * Theta1_grad;
Theta2_grad = (1/m) * Theta2_grad;
Theta3_grad = (1/m) * Theta3_grad;


Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ((lambda/m) * Theta1(:, 2:end)); % for j >= 1 

Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ((lambda/m) * Theta2(:, 2:end)); % for j >= 1

Theta3_grad(:, 2:end) = Theta3_grad(:, 2:end) + ((lambda/m) * Theta3(:, 2:end));


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:); Theta3_grad(:)];


end
