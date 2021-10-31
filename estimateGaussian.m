function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%
for i=1:n %for all the n features 
  
  %by definition the mu1 loads in the column of m examples of the 1st
  %feature x1
  %compute mu1,mu2,...mun
  mu(i,1) = (1/m)* sum(X(:,i)); %for mu1 sum of all x1 (1st feature) from 1 through m, column of X
  sqrDiff = (X(:,i) - mu(i,1)).^2; %diff btw column of x1 features and compured mean  
  sigma2(i,1) = (1/m)* sum(sqrDiff); 
    
end 
% =============================================================
end
