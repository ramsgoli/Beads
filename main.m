clear;
close all;
clc;

input_layer_size = 3; % number of features
hidden_layer_size = 50; %number of units in hidden layer of NN
num_labels = 2; %number of labels; money = 1; basket = 2;

%=================== Load Data ===================
%========== Money label = 1; Basket label = 2=====
%===== 427 money sets; 398 basket labels =========

X = load('data.txt');
Y = load('labels.txt');

m = size(X, 1);
fprintf("Data loaded. Press enter\n");
pause;

%=========== Set up Xtrain, test, and cv ========
%{
sel = randperm(825);
Xtrain = X(sel(1:495),:);
ytrain = Y(sel(1:495),:);
Xval = X(sel(496:660),:);
yval = Y(sel(496:660),:);
%}



%=================== Graph Data =================
%{
%Money beads
x = X(1:427,1); %diameter
y = X(1:427,2); %thickness
z = X(1:427,3); %aperture

%Basket beads
a = X(428:825, 1); %diameter
b = X(428:825, 2); %thickness
c = X(428:825, 3); %aperture

scatter3(x,y,z, 5, 'r', 'rx');
hold on;
scatter3(a,b,c, 5, 'b', 'rx');
xlabel('diameter');
ylabel('thickness');
zlabel('aperture');
%}
fprintf("press enter to initialize parameters\n");
pause;


%=========== Graph train and cv set ==============
load('subsets.mat');

%{
xtrainbasket = Xtrain(find(ytrain == 2),:);
xtrainmoney = Xtrain(find(ytrain == 1),:);

xvalbasket = Xval(find(yval == 2),:);
xvalmoney = Xval(find(yval == 1),:);

a = xtrainbasket(:, 1); %diameter
b = xtrainbasket(:, 2); %thickness
c = xtrainbasket(:, 3); %aperture

d = xtrainmoney(:, 1);
e = xtrainmoney(:, 2);
f = xtrainmoney(:, 3);

g = xvalbasket(:, 1);
h = xvalbasket(:, 2);
p = xvalbasket(:, 3);

j = xvalmoney(:, 1);
k = xvalmoney(:, 2);
l = xvalmoney(:, 3);

scatter3(a,b,c, 5, 'r', 'rx');
hold on;
scatter3(g,h,p, 5, 'b', 'rx');

xlabel('diameter');
ylabel('thickness');
zlabel('aperture');
fprintf("paused\n");
pause;
%}
%==============Initialize parameters =============

fprintf('initializing parameters . . .\n')

initial_Theta1 = debugInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = debugInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params = [initial_Theta1(:); initial_Theta2(:)];

%=============Train the Neural Network ===========

for i = 1:500

options = optimset('MaxIter', i);
lambda = .1;

costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, Xtrain, ytrain, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
               hidden_layer_size, (input_layer_size + 1));
                 
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));
pred_val = predict(Theta1, Theta2, Xval);
cv_accuracy =  mean(double(pred_val == yval)) * 100;
fprintf('\nCross-validation set accuracy: %f\n', cv_accuracy);

end
fprintf('finished training');
pause;

%============== Predict ==========================

pred_train = predict(Theta1, Theta2, Xtrain);
pred_val = predict(Theta1, Theta2, Xval);


%fprintf('\nTraining Set Accuracy: %f percent\n', mean(double(pred == Y)) * 100);
train_accuracy = mean(double(pred_train == ytrain)) * 100;
cv_accuracy =  mean(double(pred_val == yval)) * 100;

fprintf('\nTraining set accuracy: %f',train_accuracy);
fprintf('\nCross-validation set accuracy: %f\n', cv_accuracy);

pred_miss = find(pred_val != yval);

%============== Learning Curve ===================
%{
[error_train, error_val, theta] = ...
    learningCurve(Xtrain, ytrain, ...
                  Xval, yval, ...
                  lambda, input_layer_size, hidden_layer_size);
                  
   

Theta1 = reshape(theta(1:hidden_layer_size * (input_layer_size + 1)), ...
                hidden_layer_size, (input_layer_size + 1));
                 
Theta2 = reshape(theta((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));


%}
%=======================Validation Curve ============
%{
[lambda_vec, error_train, error_val] = ...
    validationCurveAv(X,Y, input_layer_size, 
                hidden_layer_size);

close all;
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

fprintf('lambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n', ...
            lambda_vec(i), error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;
%}          