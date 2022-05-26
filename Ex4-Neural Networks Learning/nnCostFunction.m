function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
p = zeros(size(y, 1), 1);  
Y=[];
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% -------------------------------------------------------------

% =========================================================================

%Feedforward
a1 = X; % 5000 * 400
% Add bias to a1
a1= [ones(m, 1) a1]; % 5000 * 401
% Activations
a2 = sigmoid(a1 * Theta1');% (5000 * 401) * (401 * 25)  
% Add bias to a2
a2= [ones(m, 1) a2]; %5000 * 26
% Activations
a3 = sigmoid(a2 * Theta2');% (5000 * 26) * (26 * 10)  

for i = 1:m
  Y = [Y (1:num_labels==y(i))'];
endfor

for i = 1:m
  for j = 1:num_labels
    J =  J + (-Y(j,i)* (log(a3(i,j))) - (1-Y(j,i)) * log(1-a3(i,j))) ;
  endfor
endfor

%Theta2 regularization (10*26)
%Theta2_nobias (10*25)
Theta2_nobias = Theta2(:,2:end);
for i = 1:num_labels
  for j = 1:hidden_layer_size
    Theta2_sq(i,j) =  Theta2_nobias(i,j) .^ 2 ;
  endfor
endfor
Theta2_sq_sum = sum(sum(Theta2_sq));

%Theta1 regularization (25*401)
%Theta1_nobias (25*400)
Theta1_nobias = Theta1(:,2:end);
for i = 1:hidden_layer_size
  for j = 1:input_layer_size
    Theta1_sq(i,j) =  Theta1_nobias(i,j) .^ 2 ;
  endfor
endfor
Theta1_sq_sum = sum(sum(Theta1_sq));

J = ((1/m) * J) + (lambda/ (2 *m)) * (Theta2_sq_sum + Theta1_sq_sum);

%----------BackProp--------------------
%Y  (10*5000)
%a3 (5000*10)
%a2 (5000*26)
%a1 (5000*401)
%Theta1_grad(:) (25*401)
%Theta2_grad(:) (10*26)
%D3(5000*10) - (10*25)
%D2(5000*25) - (25*400)

D3 = (a3 - Y'); %(5000*10)
D2 = Theta2_nobias' * D3' .* (sigmoidGradient(a1 * Theta1'))';% (25*10) (10*5000) .* (5000*25)
DL3 = D3' * a2; %(10*5000) * (5000*26)
DL2 = D2 * a1; %(25*5000) * (5000*401)

DL2(:,2:end) = (1/m) *DL2(:,2:end) + (lambda/m) * Theta1_nobias; %(25*400) + (25*400)
DL2(:,1) = (1/m) *DL2(:,1);

DL3(:,2:end) = (1/m) *DL3(:,2:end) + (lambda/m) * Theta2_nobias; %(10*25) + (25*10)
DL3(:,1) = (1/m) *DL3(:,1);

Theta1_grad(:) = DL2;
Theta2_grad(:) = DL3;

grad = [Theta1_grad(:) ; Theta2_grad(:)];
end