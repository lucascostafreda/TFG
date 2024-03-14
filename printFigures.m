% Load the data
load('gradient_vector.mat', 'totalGradient'); % Loads the totalGradient matrix
load('accuracy2_data.mat', 'accuracy2'); % Loads the accuracy2 matrix

% Calculate moving averages with a window of 3
movingAvgAccuracy = movmean(accuracy2, [2 0]); % Current and 2 past
movingAvgGradient = movmean(totalGradient, [2 0]); % Current and 2 past

% Assuming 'iteration' is the length of your saved matrices
iteration = length(accuracy2); % or use length(totalGradient) as appropriate

% Plotting accuracy with moving average line
fig1 = figure;
hold on;
xlabel('Iteration');
ylabel('Current Accuracy');
title('Real-Time Accuracy Plot (RM_NE)');
plot(1:iteration, accuracy2, 'bo-'); % Plot accuracy data
plot(1:iteration, movingAvgAccuracy, 'r-', 'LineWidth', 2); % Plot moving average line
grid on;
legend('Accuracy', 'Moving Avg Accuracy');

% Plotting gradient with moving average line
fig2 = figure;
hold on;
xlabel('Iteration');
ylabel('Current Gradient');
title('Real-Time Gradient Plot (RM_NE)');
plot(10:iteration, totalGradient(10:end), 'ro-'); % Plot gradient data, skipping the first nine as before
plot(10:iteration, movingAvgGradient(10:end), 'b-', 'LineWidth', 2); % Plot moving average line
grid on;
legend('Gradient', 'Moving Avg Gradient');
