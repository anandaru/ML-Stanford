function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
s = size(theta);
grad = zeros(s);
temp_theta = theta(2:s);
new_theta = [0;temp_theta];
a = sigmoid(X * theta);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

J = (1/m) * (-y' * (log(a)) - (1-y)' * log(1-a) )+ (lambda/(2 * m)) * new_theta' * new_theta; 
grad = (1/m) * (X' * (a - y)  + (lambda)* new_theta);

end
