function comparePIConfigurations()
    % Load deviation data and compute/display the difference
    [deviationDifference1, deviationDifference2] = computeDeviationDifference();
    disp('Difference between deviation vectors:');
    disp(deviationDifference1);
    disp(deviationDifference2);
    
    % Load accuracy data and plot for comparison
    plotAccuracyComparison();
end

function [deviationDifference1, deviationDifference2]  = computeDeviationDifference()
    % Load the deviation vectors from the saved files
    data1 = load('/gpfs/workdir/costafrelu/deviation_PI_1.mat');
    data2 = load('/gpfs/workdir/costafrelu/deviation_PI_2.mat');
    data3 = load('/gpfs/workdir/costafrelu/deviation_PI_3.mat');
    

    % Extract the deviation vectors from the loaded data
    deviation1 = data1.average_deviation;
    deviation2 = data2.average_deviation;
    deviation3 = data3.average_deviation;

    disp(deviation1);
    disp(deviation2);
    disp(deviation3);
    % Compute the difference between the two deviation vectors
    deviationDifference1 = deviation2 - deviation1;
    deviationDifference2 = deviation3-deviation1;
end

function plotAccuracyComparison()
    % Load the accuracy vectors from the saved files
    data1 = load('/gpfs/workdir/costafrelu/Accuracy_PI_1.mat');
    data2 = load('/gpfs/workdir/costafrelu/Accuracy_PI_2.mat');
    data3 = load('/gpfs/workdir/costafrelu/Accuracy_PI_3.mat');

    % Extract the accuracy vectors from the loaded data
    accuracy1 = data1.average_accuracy;
    accuracy2 = data2.average_accuracy;
    accuracy3 = data3.average_accuracy;

    % Create a new figure for the plot
    figure;

    % Plot the first accuracy vector
    plot(accuracy1, 'o-', 'DisplayName', 'Accuracy PI Config 1');
    hold on; % Hold on to plot the second vector on the same graph

    % Plot the second accuracy vector
    plot(accuracy2, 'x-', 'DisplayName', 'Accuracy PI Config 2');
    
    % Plot the third accuracy vector
    plot(accuracy3, 'd-', 'DisplayName', 'Accuracy PI Config 3');

    % Enhance the plot
    xlabel('Iteration');
    ylabel('Accuracy');
    title('Accuracy Vectors Comparison');
    legend show; % Display a legend to identify the vectors
    grid on; % Optional: Add a grid for easier comparison
end
