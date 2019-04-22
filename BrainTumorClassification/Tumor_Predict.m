%% Initialization
clear ; close all; clc


% Load the Spam Email dataset
% You will have X, y in your environment
load('Trainset.mat');

X = meas;
m = size(label, 1);
y = zeros(m, 1);

for i = 1:m
y(i) = isequal(label{i}, 'MALIGNANT');
end

fprintf('\nTraining Linear SVM (Spam Classification)\n')
fprintf('(this may take 1 to 2 minutes) ...\n')

C = 30;
sigma = 0.3;
model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
p = svmPredict(model, X);
fprintf('Training Accuracy: %f\n', mean(double(p == y)) * 100);


load('BenignSet.mat');
m = size(feat, 1);
pb = zeros(m,1);
pb = svmPredict(model, feat);
y = zeros(size(pb,1), 1);
fprintf('Accuracy on Bening Tumor Set: %f\n', mean(double(pb == y)) * 100);


load('MalignantSet.mat');
m = size(feat, 1);
pm = zeros(m,1);
pm = svmPredict(model, feat);
y = ones(size(pm,1), 1);
fprintf('Accuracy on Malignant Tumor Set: %f\n', mean(double(pm == y)) * 100);


%% Neural Network Implementation

load 'Trainset.mat';

X = meas;
m = size(label, 1);
y = zeros(m, 1);

for i = 1:m
y(i) = isequal(label{i}, 'MALIGNANT');
end


%% Setup the parameters you will use for this exercise
input_layer_size  = size(X, 2);%number of input features
hidden_layer_size = [10 5];   % number and size of hidden layers
num_labels = 1;          % 0 for benign and 1 for malignant labels



fprintf('\nInitializing Neural Network Parameters ...\n')


initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size(1));
initial_Theta2 = randInitializeWeights(hidden_layer_size(1), hidden_layer_size(2));
initial_Theta3 = randInitializeWeights(hidden_layer_size(2), num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:); initial_Theta3(:)];



%% ===================  Training NN ===================


fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 1000);

%  You should also try different values of lambda
lambda = 0.001;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
                                                                   % neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
len1 = hidden_layer_size(1) * (input_layer_size + 1);
Theta1 = reshape(nn_params(1:len1), hidden_layer_size(1), (input_layer_size + 1));

len2 = len1+ hidden_layer_size(2) * (hidden_layer_size(1) + 1);
Theta2 = reshape(nn_params(1 + len1:len2), hidden_layer_size(2), (hidden_layer_size(1) + 1));

Theta3 = reshape(nn_params(1+len2:end), num_labels, (hidden_layer_size(2) + 1));

fprintf('Program pause(1)d. Press enter to continue.\n');



pred = predict(Theta1, Theta2, Theta3, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);



load('BenignSet.mat');
m = size(feat, 1);
pbn = zeros(m,1);
pbn = predict(Theta1, Theta2, Theta3, feat);
y = zeros(size(pbn,1), 1);
fprintf('Neural Accuracy on Bening Tumor Set: %f\n', mean(double(pbn == y)) * 100);


load('MalignantSet.mat');
m = size(feat, 1);
pmn = zeros(m,1);
pmn = predict(Theta1, Theta2, Theta3, feat);
y = ones(size(pmn,1), 1);
fprintf('Neural net Accuracy on Malignant Tumor Set: %f\n', mean(double(pmn == y)) * 100);


figure(1)
plot([1:length(pb)], pb, 'bo', 'LineWidth', 3, 'MarkerSize', 10);hold on;
plot([1:length(pb)], pbn, 'rx', 'LineWidth', 3, 'MarkerSize', 10); hold off;
legend([{'SVM'}, {'NN'}]);

figure(2)
plot([1:length(pm)], pm, 'bo', 'LineWidth', 3, 'MarkerSize', 10);hold on;
plot([1:length(pmn)], pmn, 'rx', 'LineWidth', 3, 'MarkerSize', 10); hold off;
legend([{'SVM'}, {'NN'}]);

