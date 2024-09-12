function plotWeightedSumScore_15_i(ruche, iteration, average, numDatasets, weights, plotType)
    % Define the maximum number of iterations to process
    maxIterations = 15;

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
    deviations = loadDeviations(fullPath, numDatasets, 'Deviation', maxIterations);
    accuracies = loadDeviations(fullPath, numDatasets, 'Accuracy', maxIterations);

    % Calculate and plot weighted scores
    titleText = sprintf('Weighted Scores [%.2f, %.2f]', weights(1), weights(2));
    plotWeightedScores(deviations, accuracies, weights, titleText, plotType);
end

function plotWeightedScores(deviations, accuracies, weights, titleSuffix, plotType)
    numDatasets = size(deviations, 1);
    numIterations = size(deviations, 2);
    scores = zeros(numDatasets, numIterations);

    % Define a set of colors for each dataset
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
        'Weighted Score A',
        'Weighted Score B',
        'Weighted Score C',
        'Weighted Score D',
        'Weighted Score E',
        'Weighted Score F',
        'High Weighted Score G',
        'Varied Score H',
        'Final Score I'
    };

    markers = {'o', 's', '^', 'v', '>', '<', 'p', 'h', '+', 'x'};

    for i = 1:numIterations
        % Normalize the scores for the current iteration
        [normDev, normAcc] = normalizeScores(deviations(:, i), accuracies(:, i));
        scores(:, i) = weights(1) * normDev + weights(2) * normAcc;
    end

    % Smooth the scores
    smoothedScores = zeros(size(scores));
    for j = 1:numDatasets
        smoothedScores(j, :) = smooth(scores(j, :), 0.1, 'rloess'); % Using 'rloess' for robust local regression smoothing
    end

    % Calculate the rankings for each dataset at each iteration
    rankings = zeros(size(smoothedScores));
    for i = 1:numIterations
        [~, inds] = sort(smoothedScores(:, i), 'descend');
        rankings(inds, i) = linspace(1, numDatasets, numDatasets);
    end

    figure;
    hold on;

    if strcmp(plotType, 'rankings')
        % Plotting the rankings for each dataset
        for j = 1:numDatasets
            datasetRankings = rankings(j, :);
            plot(1:numIterations, datasetRankings, 'Color', customColors(j, :), ...
                'Marker', markers{mod(j-1, length(markers)) + 1}, 'DisplayName', ...
                customNames{j});  % Using custom names for display in legend
        end
        ylabel('Ranking Position');
        set(gca, 'YDir', 'reverse');  % Invert the Y-axis
    elseif strcmp(plotType, 'normalized')
        % Plotting the raw smoothed normalized values for each dataset
        for j = 1:numDatasets
            datasetScores = smoothedScores(j, :);
            plot(1:numIterations, datasetScores, 'Color', customColors(j, :), ...
                'DisplayName', customNames{j});  
        end
        ylabel('Smoothed Normalized Score');
    else
        error('Invalid plotType. Use "rankings" or "normalized".');
    end

    xlabel('Iteration');
    title(['Weighted Scores Comparison: ', titleSuffix]);
    legend('show', 'Location', 'southeast');  % Positioning legend at the bottom right
    grid on;
    hold off;
end

function [normalizedDeviation, normalizedAccuracy] = normalizeScores(deviation, accuracy)
    % Normalizing deviation so that lower values are better
    minDev = min(deviation);
    maxDev = max(deviation);
    normalizedDeviation = 1 - (deviation - minDev) / (maxDev - minDev);  % Inverting the scale

    % Normalizing accuracy so that higher values are better
    minAcc = min(accuracy);
    maxAcc = max(accuracy);
    normalizedAccuracy = (accuracy - minAcc) / (maxAcc - minAcc);
end

function deviations = loadDeviations(fullPath, numFiles, mode, targetLength)
    % Determine the data field based on the mode
    if strcmp(mode, 'Deviation')
        dataField = 'average_deviation';
    else
        dataField = 'average_accuracy';
    end
    
    % Initialize matrix to store deviations or accuracies
    deviations = zeros(numFiles, targetLength);
    
    % Load each file and extract the relevant data
    for i = 1:numFiles
        fileData = load(fullfile(fullPath, sprintf('Temp_%s_PI_%d.mat', mode, i)));
        if isfield(fileData, dataField)
            dataVector = fileData.(dataField);
            if length(dataVector) >= targetLength
                deviations(i, :) = dataVector(1:targetLength);
            else
                error('Data vector in file %d is shorter than the target length.', i);
            end
        else
            error(['Field "', dataField, '" does not exist in file ', num2str(i), '.']);
        end
    end
end
