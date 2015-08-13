function [error_train error_val theta] = ...
    learningCurve(X, y, Xval, yval, lambda, input_layer_size, hidden_layer_size)

        
m = 30;
error_train = zeros(m, 1);
error_val = zeros(m,1);


for i = 1:m
  fprintf("training size number %f\n", i);
	[theta] = trainNN(input_layer_size, hidden_layer_size, X(1:i, :), y(1:i, :), lambda);
  
	error_train(i) = nnCostFunction(theta, input_layer_size, ...
          hidden_layer_size, 2, X(1:i, :), y(1:i), 0);
	error_val(i) = nnCostFunction(theta, input_layer_size, ...
          hidden_layer_size, 2, Xval(1:i, :), yval(1:i), 0);
end
