function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


%Test values as recommended in lectures
testValues = [0.01, 0.03, 0.1, 0.3,  1, 3, 10, 30];

results = [];

%Test c values
for loopC=1:length(testValues),
	%Test sigma values
    for loopSigma=1:length(testValues),
      
      %Test C
      testC = testValues(loopC);
	  %Test Sigma
      testSigma = testValues(loopSigma);
      
	  %Train the model with the same training as in ex6 but with different Sigmas and C's
      model= svmTrain(X, y, testC, @(x1, x2) gaussianKernel(x1, x2, testSigma)); 
	  %Use the svm to make predictions
      predictions = svmPredict(model, Xval);
      
	  %What is the test error? As listed above for computing the prediction error
      testError = mean(double(predictions ~= yval));
      
      results = [results; testC, testSigma, testError];
      
    end
end

%Get the index of the min in regards to the least error
[minError, minIndex] = min(results(:,3));


%Most optimal C
C = results(minIndex,1);
%Most optimal Sigma
sigma = results(minIndex,2);




% =========================================================================

end
