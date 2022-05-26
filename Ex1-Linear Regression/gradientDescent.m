function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta); % number of features + 1
derivatives = zeros(n, 1); % initialize fitting parameters
J_history = zeros(num_iters, 1);

cost = computeCost(X, y, theta);
fprintf('cost = %f\n', cost);
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %initialize derivatives to zero
    for j = 1:n,
      derivatives(j) = 0 ;
    end
    %hypothesis
    y_pred = X * theta;
    
    # partial derivative of cost function wrt theta
    for j = 1:n,
        for k = 1:m,
          derivatives(j)= derivatives(j) + ((y_pred(k) - y(k)) * X(k,j))/m;
        end 
    end  
    # Find better theta to get lesser cost
    for j = 1:n,
      theta(j) = theta(j) - (alpha * derivatives(j));
    end 
    cost = computeCost(X, y, theta);
    fprintf('cost = %f\n', cost);



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = cost;

end

end
