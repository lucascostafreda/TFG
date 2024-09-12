
%%%%%%%%%%%%%%%% GRAFICA %%%%%%%%%%%%%%%%%%

<<<<<<< HEAD
function comparePIConfigurations(ruche, iteration, average, numDatasets)
    % Determine base directory
=======
function comparePIConfigurations(ruche, iteration, average, numDatasets, plotAveraged)
      % Determine base directory
>>>>>>> 9c6ccc124 (Reinicializando el repositorio)
    if ruche
        baseDir = '/gpfs/workdir/costafrelu/temporaryMat/';
    else
        baseDir = fullfile('..', 'workdir', 'temporaryMat');
    end

    % Construct the directory path
    directory_tempDir = sprintf('i_%d_avg_%d', iteration, average);
    fullPath = fullfile(baseDir, directory_tempDir);

<<<<<<< HEAD
    % Load deviation data
    deviations = loadDeviations(fullPath, numDatasets, 'Deviation');

    % Compute differences from the last deviation
    deviationDifferences = computeDeviationDifferences(deviations);

    % Plot all deviations
    plotDeviations(deviations, 'Deviation');
    % 
    % % Plot the differences
    plotDeviations(deviationDifferences, 'Difference with Last Deviation');

    % Load and plot accuracy data
    plotAccuracyComparison(fullPath, numDatasets);
end


function deviations = loadDeviations(fullPath, numFiles, mode)
    deviations = cell(1, numFiles);
    for i = 1:numFiles
        data = load(fullfile(fullPath, sprintf('Temp_%s_PI_%d.mat', mode, i)));
        deviations{i} = data.average_deviation;
    end
end



function deviationDifferences = computeDeviationDifferences(deviations)
    lastDeviation = deviations{end};  % The last deviation is used for comparison
    deviationDifferences = cell(1, numel(deviations) - 1);
    for i = 1:numel(deviations) - 1
        deviationDifferences{i} = deviations{i} - lastDeviation;
    end
end


function plotDeviations(deviations, titleSuffix)
    figure;
    hold on;
    colors = lines(numel(deviations));  % Ensure enough colors are available
    markers = {'o', 's', '^', 'v', '>', '<', 'p', 'h', '*', '+'};  % List of markers for different lines
    for i = 1:numel(deviations)
        markerIndex = mod(i-1, length(markers)) + 1;  % Cycle through markers if there are more lines than markers
        plot(deviations{i}, 'Color', colors(i, :), 'Marker', markers{markerIndex}, 'DisplayName', sprintf('%s %d', titleSuffix, i));
    end
    xlabel('Iteration');
    ylabel([titleSuffix, ' Value']);
    title([titleSuffix, ' Comparison']);
    legend show;
    grid on;
end
    


function plotAccuracyComparison(fullPath, numAccuracies)
    % Load the accuracy vectors from the saved files
    accuracies = cell(1, numAccuracies);
    for i = 1:numAccuracies
        data = load(fullfile(fullPath, sprintf('Temp_Accuracy_PI_%d.mat', i)));
        accuracies{i} = data.average_accuracy;
    end

=======
    if plotAveraged
        % Plot averaged deviations
        averagedDeviations = loadAveragedVectors(fullPath, 'Deviation');
        plotDeviations(averagedDeviations, 'Averaged Deviation');

        % Plot averaged accuracies
        averagedAccuracies = loadAveragedVectors(fullPath, 'Accuracy');
        plotAccuracyComparison(averagedAccuracies, 'Averaged Accuracy');
    else
        % Load deviation data
        deviations = loadDeviations(fullPath, numDatasets, 'Deviation');
        % Plot all deviations
        plotDeviations(deviations, 'Deviation');

        % Load and plot accuracy data
        accuracies = loadDeviations(fullPath, numDatasets, 'Accuracy');
        plotAccuracyComparison(accuracies, 'Accuracy');
    end
end

function deviations = loadDeviations(fullPath, numFiles, mode)
    deviations = cell(1, numFiles);
    for i = 1:numFiles
        filePath = fullfile(fullPath, sprintf('Temp_%s_PI_%d.mat', mode, i));
        data = load(filePath);
        
        % Check if data is already a vector
        if isvector(data)
            deviations{i} = data;
        else
            % List all variables in the loaded file
            vars = fieldnames(data);
            
            % Check if 'average_deviation' exists, otherwise use the first variable
            deviations{i} = data.(vars{1});
        end
    end
end

function averagedVectors = loadAveragedVectors(fullPath, mode)
    averagedVectors = cell(1, 8);  % Assuming 8 averaged vectors for both modes
    for i = 1:8
        if strcmp(mode, 'Deviation')
            fileName = sprintf('AveragedModel_Dev_%d.mat', i);
        else
            fileName = sprintf('AveragedModel_Acc_%d.mat', i);
        end
        filePath = fullfile(fullPath, fileName);
        data = load(filePath);
        vars = fieldnames(data);
        averagedVectors{i} = data.(vars{1});
    end
end

function plotDeviations(deviations, titleSuffix)
    figure;
    hold on;

    % Define custom colors as RGB triplets
    customColors = [
        1, 0, 0;  % Red
        0, 1, 0;  % Green
        0, 0, 1;  % Blue
        1, 1, 0;  % Yellow
        0, 1, 1;  % Cyan
        1, 0, 1;  % Magenta
        0.5, 0.5, 0.5;  % Gray
        0, 0.5, 0;  % Dark Green
    ];

    % Define custom names for each plot
    customNames = {
         '\pi_A'
         '\pi_B'
         '\pi_C'
         '\pi_D'
         '\pi_E'
         '\pi_F'
         '\pi_H'
         '\pi_I'
    };

    markers = {'o', 's', '^', 'v', '>', '<', 'p', 'h', '*', '+'};

    % Ensure there are enough colors and names defined
    if size(customColors, 1) < numel(deviations) || length(customNames) < numel(deviations)
        error('Define more colors or names to match the number of deviations');
    end

    % Plot each deviation vector with custom colors and names
    for i = 1:numel(deviations)
        if ~isempty(deviations{i})
            plot(deviations{i}, 'Color', customColors(i, :), 'DisplayName', customNames{i});
        else
            warning('Empty data at index %d, skipping plot for this dataset.', i);
        end
    end

    % Enhance the plot
    xlabel('Iteration');
    ylabel('Deviation');
    title([titleSuffix, ' ']);

    % Customize and display the legend
    legend('show', 'Location', 'southeast');  % Set legend to the bottom right corner
    grid on;  % Optional: Add a grid for easier comparison
    hold off;
end

function plotAccuracyComparison(accuracies, titleSuffix)
>>>>>>> 9c6ccc124 (Reinicializando el repositorio)
    % Create a new figure for the plot
    figure;
    hold on;

<<<<<<< HEAD
    % Define a colormap for visual distinction of the lines
    colors = jet(numAccuracies);  % Adjusted to handle dynamic number of datasets
    markers = {'o', 's', '^', 'v', '>', '<', 'p', 'h', '*', '+'};  % List of markers for different lines

    % Plot each accuracy vector
    for i = 1:numAccuracies
        markerIndex = mod(i-1, length(markers)) + 1;  % Cycle through markers if there are more lines than markers
        plot(accuracies{i}, 'Color', colors(i, :), 'Marker', markers{markerIndex}, 'DisplayName', sprintf('Accuracy PI %d', i));
=======
    % Define custom colors as RGB triplets
    customColors = [
        1, 0, 0;  % Red
        0, 1, 0;  % Green
        0, 0, 1;  % Blue
        1, 1, 0;  % Yellow
        0, 1, 1;  % Cyan
        1, 0, 1;  % Magenta
        0.5, 0.5, 0.5;  % Gray
        0, 0.5, 0;  % Dark Green
        0.5, 0, 0.5  % Purple
    ];

    % Define custom names for each plot
    customNames = {
         '\pi_A'
         '\pi_B'
         '\pi_C'
         '\pi_D'
         '\pi_E'
         '\pi_F'
         '\pi_H'
         '\pi_I'
    };

    markers = {'o', 's', '^', 'v', '>', '<', 'p', 'h', '*', '+'};

    % Ensure there are enough colors and names defined
    if size(customColors, 1) < numel(accuracies) || length(customNames) < numel(accuracies)
        error('Define more colors or names to match the number of accuracies');
    end

    % Plot each accuracy vector with custom colors and names
    for i = 1:numel(accuracies)
        if ~isempty(accuracies{i})
            plot(accuracies{i}, 'Color', customColors(i, :), 'DisplayName', customNames{i});
        else
            warning('Empty data at index %d, skipping plot for this dataset.', i);
        end
>>>>>>> 9c6ccc124 (Reinicializando el repositorio)
    end

    % Enhance the plot
    xlabel('Iteration');
    ylabel('Accuracy');
<<<<<<< HEAD
    title('Accuracy Vectors Comparison');
    legend show;  % Display a legend to identify the vectors
    grid on;  % Optional: Add a grid for easier comparison
end

=======
    title([titleSuffix, ' ']);

    % Customize and display the legend
    legend('show', 'Location', 'southeast');  % Set legend to the bottom right corner
    grid on;  % Optional: Add a grid for easier comparison
    hold off;
end
>>>>>>> 9c6ccc124 (Reinicializando el repositorio)

%%%%%%%%%%%%%%%% TABLA %%%%%%%%%%%%%%%%%%
% 
% function comparePIConfigurations(ruche, iteration, average, numDatasets)
%     % Determine base directory
%     if ruche
%         baseDir = '/gpfs/workdir/costafrelu/temporaryMat/';
%     else
%         baseDir = fullfile('..', 'workdir', 'temporaryMat');
%     end
% 
%     % Construct the directory path
%     directory_tempDir = sprintf('i_%d_avg_%d', iteration, average);
%     fullPath = fullfile(baseDir, directory_tempDir);
% 
%     % Load and rank deviations
%     deviations = loadDeviations(fullPath, numDatasets, 'Deviation');
%     rankDeviations(deviations, 'Deviation');
% 
%     % Load and rank accuracies
%     accuracies = loadDeviations(fullPath, numDatasets, 'Accuracy');
%     rankDeviations(accuracies, 'Accuracy');
% end
% 
% function deviations = loadDeviations(fullPath, numFiles, mode)
%     % Initialize matrix to store deviations or accuracies
%     if strcmp(mode,'Deviation')
%         numIterations = numel(load(fullfile(fullPath, sprintf('Temp_%s_PI_%d.mat', mode, 1))).average_deviation); % Assume all files have the same number of iterations
%     else
%         numIterations = numel(load(fullfile(fullPath, sprintf('Temp_%s_PI_%d.mat', mode, 1))).average_accuracy); % Assume all files have the same number of iterations
%     end
%     deviations = zeros(numFiles, numIterations);
%     for i = 1:numFiles
%         data = load(fullfile(fullPath, sprintf('Temp_%s_PI_%d.mat', mode, i)));
%         if strcmp(mode, 'Deviation')
%             deviations(i, :) = data.average_deviation;
%         else
%             deviations(i, :) = data.average_accuracy;
%         end
%     end
% end
% 
% function rankDeviations(deviations, titleSuffix)
%     % Transpose for easier manipulation
%     deviations = deviations';
%     numIterations = size(deviations, 1);
% 
%     % Prepare a figure
%     figure;
%     hold on;
% 
%     for i = 1:numIterations
%         % Sort and determine ranking
%         %% 
%         if strcmp(titleSuffix, 'Deviation')
%             [~, idx] = sort(deviations(i, :), 'ascend');  % Lowest to highest for deviations
%         else
%             [~, idx] = sort(deviations(i, :), 'descend'); % Highest to lowest for accuracies
%         end
% 
%         % Display ranking information
%         disp(['Iteration ', num2str(i), ' ', titleSuffix, ' Rankings:']);
%         disp(idx); % This displays the indices of the datasets from best to worst or vice versa
%     end
% end
% 
