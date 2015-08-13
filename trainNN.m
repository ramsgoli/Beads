function [nn_params] = trainNN(input_layer_size, hidden_layer_size, X, y, lambda)

num_labels = 2;
initial_Theta1 = debugInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = debugInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params = [initial_Theta1(:); initial_Theta2(:)];


options = optimset('MaxIter', 30);


costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

%Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
%                 hidden_layer_size, (input_layer_size + 1));
                 
%Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));