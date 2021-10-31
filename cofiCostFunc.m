function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X)); %1682x100
Theta_grad = zeros(size(Theta));%943x100

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features 
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%        X - 1682x100, nmx100
%        Theta- 943x100, nux100
%        Y- 1682x943, nm x nu, X * Theta'
%        R- 1682x943, nm x nu
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
sqrDiff = ((X * Theta')- Y).^2;

% using vectorized implementation to compute J
%use the R matrix to set selected entries to 0, since R only has elements with values either 0 or 1, 
%this has the effect of setting the elements of M to 0 only when the corresponding value
%in R is 0.
J = 1/2 * sum(sum(R.*sqrDiff)); %use R matrix to check if movie has been rated

%======================================================================================
% Computing X_grad and Theta_grad

for i=1:num_movies %to compute partial derivative w.r.t each element of X
    %idx changes every iteration based on THAT current ith movie selected
    idx = find(R(i,:)==1); %EACH ITERATION give list of all users who have rated a particular ith movie
    theta_temp = Theta(idx,:);% idx X 100 give ONLY parameters for the set of users that rated the ith movie
    Y_temp = Y(i,idx);%1 x idx, select that ith movie and select the idx users whom rated it
    
    % X_grad is ((1xnu)*(idx x 100)' - (1 x idx)) * (idx X 100) yielding 
    % 1 X 100 Row Vector
    X_grad(i,:) = (X(i,:)* theta_temp' - Y_temp)* theta_temp; %returns as a row vector
end 

for u=1:num_users %to compute partial derivatives w.r.t each element of Theta
    % index changes every iteration based on THAT current uth user selected
    index = find(R(:,u)==1);%list of movies rated by that particular uth user
    %Theta matrix remains unchanged since we are looping over no. of users
    Y_temp = Y(index,u); %index X 1, select the uth users and select the movies rated by THAT uth user
    X_temp = X(index,:); %index X 100, select only rated movies by THAT uth users
    
    
    % (index x 100) * (100 x 1) = (index X 1)'*(index X 100)
    % yielding 1 X 100 row vector
    Theta_grad(u,:) = (X_temp * Theta(u,:)' - Y_temp)'* X_temp;
end
% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
