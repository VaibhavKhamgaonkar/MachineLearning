%clear all
clear ; close all; clc

%Loading data 

data = load('ex1data2.txt');
X = data(:, 1:2);
X_backup = X;
y = data(:, 3);
m = length(y);


% Function to return the normalised data for X
  function[X_nor, mu, sigma] = featureNormalisation(X)
      
      mu = mean(X);
      sigma = std(X);
      X_nor = (X-mu)./sigma;
      size(X_nor)
      fprintf("Xnor value" );
      X_nor(1:10,:);
      
  end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Creating a normalised Input data
[X mu sigma]  = featureNormalisation(X);
  
%creating x0 column 
X = [ones(m,1), X];
theta = zeros(size(X,2), 1);
maxIter = 400;
alpha = 0.1;


% compute the cost function and find out the gradient descent.
  function[J, grad] = costFunction(X, y, m, theta)
      
      %J_hisotry = 0;
      h = X * theta;

      J = 1/(2*m) * (h-y)' * (h-y);  
      
      grad = (1/m) * (X' * (h-y));
  end  

%calculating cost
[J grad] = costFunction(X,y,m,theta);

%Printing the Cost 
fprintf('Cost value is %f\n' , J);


%gredient descent calculation
  function [theta, Jhistory] = GredientDescent(theta,X,y,m, alpha, maxIter)
    for iter = 1:maxIter 
      %h = X * theta;
      %temp = (alpha/m) * ( X' * (h-y) );
      %theta = theta - temp;
        theta_prev = theta;

      % number of features.
      p = size(X, 2);

      for j = 1:p
          % calculate dJ/d(theta_j)
          deriv = ((X*theta_prev - y)'*X(:, j))/m;

          % % update theta_j
          theta(j) = theta_prev(j)-(alpha*deriv);
       end
      
      
      %Calculating the J for the obtained theta
       Jhistory(iter) = costFunction(X,y,m,theta);
       
    end
  end

% 1st Method for calculating theta using Gredient descent.
[theta_gredient Jhistory] = GredientDescent(theta, X, y, m, alpha, maxIter);

% 2nd method : using advanced optimisation
% Setting Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

[theta_advancedOptim, J, exit_flag] = ...
	fminunc(@(t)(costFunction(X, y, m, t)), theta, options);



%plotting J history vs iterations for 400 iterations
plot(1:maxIter, Jhistory);
xlabel('no of iterations');
ylabel('cost value J'); 


%Predicton part 


price = [ 1 (1650 - mu(1))/sigma(1)  (3-mu(2))/sigma(2)] * theta_gredient;
fprintf(['Predicted price (using gradient descent):\n $%f\n'], price);

price = [ 1 (1650 - mu(1))/sigma(1)  (3-mu(2))/sigma(2)] * theta_advancedOptim;
fprintf(['Predicted price (using advanced Optimisation descent):\n $%f\n'], price);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% using Normal equation 

%% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

%initialising theta 
theta = zeros(size(X, 2), 1);


%creating a function to calculate the theta using Normal equation
  function[theta] = NormalEquation(X,y)
    theta = pinv(X' * X) * X' * y;
  end


% Calculate the parameters from the normal equation
theta = NormalEquation(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');


% Predicting the price for 1650 Sq feet house with 3 bedroom
% ====================== YOUR CODE HERE ======================
price = [1,1650,3]*theta; % You should change this

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price);