function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval, input_layer_size, hidden_layer_size)
    
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

for i = 1:length(lambda_vec)
  lambda = lambda_vec(i);
  [theta] = trainNN(input_layer_size, hidden_layer_size, X, y, lambda);
  
  error_train(i) = nnCostFunction(theta, input_layer_size, 
          hidden_layer_size, 2, X, y, 0);
  error_val(i) = nnCostFunction(theta, input_layer_size,
          hidden_layer_size, 2, Xval, yval, 0);
end


end