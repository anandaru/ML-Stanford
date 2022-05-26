function [theta, J_history] = gradientDescentMultiNew(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta); % number of features + 1
derivatives = zeros(n, 1); % initialize fitting parameters
J_history = zeros(num_iters, 1);

cost = computeCostMulti(X, y, theta);
fprintf('cost = %f\n', cost);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    theta = theta - ((alpha/m) * (X' * ((X * theta) - y)));
    cost = computeCostMulti(X, y, theta);
    %fprintf('cost = %f\n', cost);



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = cost;
end
J_history
end
