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


%Create the y matrix by mapping the digit to the spot where a 1 is marked in the vector




yArray = zeros(num_labels, m); 
for i=1:m,
  yArray(y(i),i)=1;
end

%Add the bias to X
X = [ones(m,1) X];

%a1 is just x
%a2 is the sigmoid of theta 1 times a1
a2 = sigmoid(Theta1 * X');

%Add ones to a2
a2 = [ones(m,1) a2'];

%Hypothesis is the sigmoid of theta 2 and a2
hyp = sigmoid(Theta2 * a2');

% Cost function equations inner sum
sum1 = sum(-yArray .* log(hyp) - (1-yArray) .* log(1-hyp));
%cost function for outer sum
J = (1/m) * sum(sum1);


%Don't regularize the bias
%Sets the first column of each theta to 0 in a temp value so that it doesn't change the costs
t1 = Theta1;
t1(:,1) = 0;
t2 = Theta2;
t2(:,1) = 0;

% J + regularization formula
J = J + ((lambda/(2*m)) * (sum(sum(t1.^2)) + sum(sum(t2.^2))));




%Back propagation


for t=1:m,
	
	%There is already bias added to X during the forward prop step instead of adding it to a1 twice
	a1 = X(t,:);
	%We now need the z terms so can't combine it here
	z2 = Theta1 * a1';
	a2 = sigmoid(z2);
	%a1 is just a vector so we need to add a 1 at the beginning
	a2 = [1 ; a2];
	z3 = Theta2 * a2;
	a3 = sigmoid(z3);
	
	%add bias for z2 (It's just a vector so just add 1
	z2 = [1; z2];
	
	%compute delta3 based off of a3 and the y array at column t, corresponding to training data m
	delta3 = a3-yArray(:,t);
	
	%compute delta 2 by using theta2, delta3, and z2
	delta2 = (Theta2' * delta3) .* sigmoidGradient(z2);
	
	%remove the bias layer in the delta
	delta2 = delta2(2:end);
	
	%accumulate the gradient of thetaX by delta(X+1) and aX
	Theta2_grad = Theta2_grad + delta3 * a2';
	Theta1_grad = Theta1_grad + delta2 * a1;
	
end
	
	%unregularized gradient on J = 0
	Theta1_grad(:, 1) = (1/m) .* Theta1_grad(:, 1);
	Theta2_grad(:, 1) = (1/m) .* Theta2_grad(:, 1);
	
	%Regularized gradient on J != 0
	Theta1_grad(:, 2:end) = (1/m) .* Theta1_grad(:, 2:end) + ((lambda/m) * Theta1(:, 2:end));
	Theta2_grad(:, 2:end) = (1/m) .* Theta2_grad(:, 2:end) + ((lambda/m) * Theta2(:, 2:end));




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
