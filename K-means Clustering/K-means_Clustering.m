% Adding a meaningless statement to the top of the file
% 	to turn it into a script.
1;

function centroids = kMeansInitCentroids(X, K)
%KMEANSINITCENTROIDS This function initializes K centroids that are to be 
%used in K-Means on the dataset X
%   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
%   used with the K-Means on the dataset X
%

centroids = zeros(K, size(X, 2));

% Initialize the centroids to be random examples

% Randomly reorder the indices of examples
randidx = randperm(size(X, 1));

% Take the first K examples as centroids
centroids = X(randidx(1:K), :);

end

function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% Number of training examples
m = size(X,1);

idx = zeros(size(X,1), 1);


for i = 1:m

	% Getting i_th training example
	x_i = X(i,:)' ;
	
	% Subtract x_i from all af centroids and find index of minimum norm
	[min_value c_i] = min(vecnorm(centroids' - x_i)) ;
	idx(i) = c_i ;

end

end

function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. Each row of centroids
%   matrix is the mean of the data points assigned to it.

% Useful variables
[m n] = size(X);

centroids = zeros(K, n);

for k = 1:K

	idx_k = find (idx == k) ;
	centroids(k,:) = sum(X(idx_k,:)) / length(idx_k) ;

end

end

function [centroids, idx] = runkMeans(X, initial_centroids,max_iters)
%RUNKMEANS runs the K-Means algorithm on data matrix X, where each row of X
%is a single example
%   [centroids, idx] = RUNKMEANS(X, initial_centroids, max_iters) runs
%   the K-Means algorithm on data matrix X, where each row of X
%   is a single example. It uses initial_centroids used as the
%   initial centroids. max_iters specifies the total number of interactions 
%   of K-Means to execute. runkMeans returns centroids, a Kxn matrix of the
%   computed centroids and idx, a m x 1 vector of centroid
%   assignments (i.e. each entry in range [1..K])
%

% Initialize values
[m n] = size(X);
K = size(initial_centroids, 1);
centroids = initial_centroids;
previous_centroids = centroids;
idx = zeros(m, 1);

% Run K-Means
for i=1:max_iters
    
    % Output progress
    fprintf('K-Means iteration %d/%d...\n', i, max_iters);
    
    % For each example in X, assign it to the closest centroid
    idx = findClosestCentroids(X, centroids);
    
    % Given the memberships, compute new centroids
    centroids = computeCentroids(X, idx, K);
end

end

%% Initialization
clear ; close all; clc


%% =================== Part 1: K-Means Clustering ======================
%  In this part, we run the K-Means algorithm on the example dataset
%  we have provided. 
%
fprintf('\nRunning K-Means clustering on example dataset.\n\n');

% Load an example dataset
load('data1.mat');

% Settings for running K-Means
K = 3;
max_iters = 10;
initial_centroids = kMeansInitCentroids(X, K);

% Run K-Means algorithm
[centroids, idx] = runkMeans(X, initial_centroids, max_iters);
fprintf('\nK-Means Done.\n\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============= Part 2: K-Means Clustering on Pixels ===============
%  In this part, we use K-Means to compress an image. To do this,
%  we first run K-Means on the colors of the pixels in the image and
%  then we map each pixel onto its closest centroid.

fprintf('\nRunning K-Means clustering on pixels from an image.\n\n');

%  Load an image of a bird
A = double(imread('bird_small.png'));

A = A / 255; % Divide by 255 so that all values are in the range 0 - 1

% Size of the image
img_size = size(A);

% Reshape the image into an Nx3 matrix where N = number of pixels.
% Each row will contain the Red, Green and Blue pixel values
% This gives us our dataset matrix X that we will use K-Means on.
X = reshape(A, img_size(1) * img_size(2), 3);

% Run K-Means algorithm on this data
% We should try different values of K and max_iters here
K = 16; 
max_iters = 10;

% When using K-Means, it is important the initialize the centroids randomly. 
initial_centroids = kMeansInitCentroids(X, K);

% Run K-Means
[centroids, idx] = runkMeans(X, initial_centroids, max_iters);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================= Part 3: Image Compression ======================
%  In this part, we use the clusters of K-Means to compress an image.

fprintf('\nApplying K-Means to compress an image.\n\n');

% Find closest cluster members
idx = findClosestCentroids(X, centroids);

% Essentially, now we have represented the image X as in terms of the
% indices in idx. 

% We can now recover the image from the indices (idx) by mapping each pixel
% (specified by its index in idx) to the centroid value
X_recovered = centroids(idx,:);

% Reshape the recovered image into proper dimensions
X_recovered = reshape(X_recovered, img_size(1), img_size(2), 3);

% Display the original image 
subplot(1, 2, 1);
imagesc(A); 
title('Original');

% Display compressed image side by side
subplot(1, 2, 2);
imagesc(X_recovered)
title(sprintf('Compressed, with %d colors.', K));

fprintf('Program paused. Press enter to continue.\n');
pause;

% =============================================================

