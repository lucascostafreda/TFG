function plotWeightedSumScore(ruche, iteration, average, numDatasets, weights)
    % Determine base directory
    if ruche
        baseDir = '/gpfs/workdir/costafrelu/temporaryMat/';
    else
        baseDir = fullfile('..', 'workdir', 'temporaryMat');
    end

    % Construct the directory path
    directory_tempDir = sprintf('i_%d_avg_%d', iteration, average);
    fullPath = fullfile(baseDir, directory_tempDir);

    % Load deviations and accuracies
    deviations = loadDeviations(fullPath, numDatasets, 'Deviation');
    accuracies = loadDeviations(fullPath, numDatasets, 'Accuracy');

    % Calculate and plot weighted scores
    plotWeightedScores(deviations, accuracies, weights, 'Weighted Scores');
end

function plotWeightedScores(deviations, accuracies, weights, titleSuffix)
    numDatasets = size(deviations, 1);
    numIterations = size(deviations, 2);
    scores = zeros(numDatasets, numIterations);
    markers = {'o', 's', '^', 'v', '>', '<', 'p', 'h', '+', 'x'}; % Different markers for lines

    for i = 1:numIterations
        % Normalize the scores for the current iteration
        [normDev, normAcc] = normalizeScores(deviations(:, i), accuracies(:, i));
        % Calculate weighted scores for each dataset
        scores(:, i) = weights(1) * normDev + weights(2) * normAcc;
    end

    % Plotting the scores
    figure;
    hold on;
    colors = lines(numDatasets); % Generates a color map with different colors for each dataset
    for j = 1:numDatasets
        plot(scores(j, :), 'Color', colors(j, :), 'Marker', markers{mod(j-1, length(markers)) + 1}, 'DisplayName', sprintf('Dataset %d', j));
    end
    xlabel('Iteration');
    ylabel('Weighted Score');
    title(['Weighted Scores Comparison: ', titleSuffix]); % Ensuring titleSuffix is used
    legend show;
    grid on;
end

function [normalizedDeviation, normalizedAccuracy] = normalizeScores(deviation, accuracy)
    minDev = min(deviation);
    maxDev = max(deviation);
    normalizedDeviation = (deviation - minDev) / (maxDev - minDev);

    minAcc = min(accuracy);
    maxAcc = max(accuracy);
    normalizedAccuracy = (accuracy - minAcc) / (maxAcc - minAcc);
end

function deviations = loadDeviations(fullPath, numFiles, mode)
    % Determine the data field based on the mode
    if strcmp(mode, 'Deviation')
        dataField = 'average_deviation';
    else
        dataField = 'average_accuracy';
    end
    
    % Pre-load the first file to determine the number of iterations
    firstFileData = load(fullfile(fullPath, sprintf('Temp_%s_PI_1.mat', mode)));
    if isfield(firstFileData, dataField)
        numIterations = numel(firstFileData.(dataField));  % Determine the length of the data vector
    else
        error(['Field "', dataField, '" does not exist in the loaded file.']);
    end
    
    % Initialize matrix to store deviations or accuracies
    deviations = zeros(numFiles, numIterations);  % Adjusted to handle dynamic number of iterations
    
    % Load each file and extract the relevant data
    for i = 1:numFiles
        fileData = load(fullfile(fullPath, sprintf('Temp_%s_PI_%d.mat', mode, i)));
        if isfield(fileData, dataField)
            deviations(i, :) = fileData.(dataField);
        else
            error(['Field "', dataField, '" does not exist in file ', num2str(i), '.']);
        end
    end
end

