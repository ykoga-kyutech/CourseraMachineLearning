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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1.3 Feedforward and cost function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Theta1 has size 25 x 401
% Theta2 has size 10 x 26
% y has size 5000 x 1

Y = zeros(m, num_labels); % [5000,10]

for i = 1:m
    Y(i,y(i)) = 1;
endfor

a1 = X;
z2 = [ones(m, 1), X] * Theta1';
a2 = sigmoid(z2);               % [5000, 25]
z3 = [ones(m, 1), a2] * Theta2';
h = sigmoid(z3);                % [5000, 10]

cost = -Y .* log(h) - (1 - Y) .* log(1 - h); % [5000,10]
J = (1/m) * sum(cost(:));       % scalar

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1.4 Regularized cost function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Theta1_nobias = Theta1(:, 2:end); %[25, 400]
Theta2_nobias = Theta2(:, 2:end); %[10, 25]
reg = lambda / (2*m) * (sumsq(Theta1_nobias(:)) + sumsq(Theta2_nobias(:))); % scalar
J = J + reg;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2.3 Backpropagation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Delta1 = Delta2 = 0.0;
for t=1:m
    % step 1 =========================
    a1 = [1; X(t,:)'];     % 401x1

    z2 = Theta1 * a1;      % 25x1
    a2 = [1; sigmoid(z2)]; % 26x1

    z3 = Theta2 * a2;      % 10x1
    a3 = sigmoid(z3);      % 10x1
    
    % step 2 ==========================
    % compute delta of the output layer3
    delta_layer_3 = a3 - Y(t,:)'; % [10, 1]
    
    % step 3 ==========================
    % compute delta of the layer2
    delta_layer_2 = (Theta2_nobias' * delta_layer_3) .* sigmoidGradient(z2); % 25x10 * 10x1 = [25, 1]
    
    % step 4 ==========================
    % Accumulate the gradient
    Delta2 = Delta2 + delta_layer_3 * a2'; % 10x1 * 1x26  = [10, 26]
    Delta1 = Delta1 + delta_layer_2 * a1'; % 25x1 * 1x401 = [25, 401]

endfor

% step 5 ==========================
% Accumulate the gradient
Theta1_grad = 1.0/m * Delta1;
Theta2_grad = 1.0/m * Delta2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2.5 Regularized Neural Networks
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ((lambda / m) * Theta1_nobias);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ((lambda / m) * Theta2_nobias);

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
