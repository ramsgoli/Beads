function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, Y, input_layer_size, hidden_layer_size)
    
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);
error_train_vec = zeros(10, 1);
error_val_vec = zeros(10,1);


for i = 1:length(lambda_vec)
  lambda = lambda_vec(i);
  fprintf('\niteration %f of %f\n', i, length(lambda_vec));
  
  for j = 1:5
    sel = randperm(825);
    Xtrain = X(sel(1:495),:);
    ytrain = Y(sel(1:495),:);
    Xval = X(sel(496:660),:);
    yval = Y(sel(496:660),:);
    
    [theta] = trainNN(input_layer_size, hidden_layer_size, Xtrain, ytrain, lambda);
    error_train_vec(j) = nnCostFunction(theta, input_layer_size, 
          hidden_layer_size, 2, Xtrain, ytrain, 0);
    error_val_vec(j) = nnCostFunction(theta, input_layer_size,
          hidden_layer_size, 2, Xval, yval, 0);
          
    end
    error_train(i) = mean(error_train_vec);
    error_val(i) = mean(error_val_vec);
end



end