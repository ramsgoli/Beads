
clear;
close all;
clc;

input_layer_size = 3;
num_labels =2;

load('subsets.mat'); %loads Xtrain, Xtest, ytrain, and ytest;

hidden_layer_vec = [1, 2, 4, 8, 16, 32, 50, 60, 70, 80, 90, 100];
%========== Train NN with various number of hidden layer units =========
lambda = 0.1;
train_pred_vec = zeros(length(hidden_layer_vec), 1);
cv_pred_vec = zeros(length(hidden_layer_vec), 1);


for i = 1:length(hidden_layer_vec),

  fprintf('Iteration %f of %f\n', i, length(hidden_layer_vec));
  
  hidden_layer_size = hidden_layer_vec(i);
  [theta] = trainNN(input_layer_size, hidden_layer_size, Xtrain, ytrain, lambda);
  
  Theta1 = reshape(theta(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
                 
  Theta2 = reshape(theta((1 + (hidden_layer_size * (input_layer_size + 1))):end), 
              num_labels, (hidden_layer_size + 1)); 
              
  train_pred_vec(i) = mean(double(predict(Theta1, Theta2, Xtrain) == ytrain)) * 100;
  cv_pred_vec(i) = mean(double(predict(Theta1, Theta2, Xval) == yval)) * 100;
  
end

fprintf('Hidden Layer Size\tTrain Prediction\tValidation Prediction\n');
for i = 1:length(hidden_layer_vec)
	fprintf(' %f\t\t%f\t\t%f\n', ...
            hidden_layer_vec(i), train_pred_vec(i), cv_pred_vec(i));
end


pause;