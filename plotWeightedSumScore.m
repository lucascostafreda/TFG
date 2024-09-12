<<<<<<< HEAD
function plotWeightedSumScore(ruche, iteration, average, numDatasets, weights)
=======
    function plotWeightedSumScore(ruche, iteration, average, numDatasets, weights, plotType)
>>>>>>> 9c6ccc124 (Reinicializando el repositorio)
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
<<<<<<< HEAD
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
=======
    deviations = loadDeviations(fullPath, numDatasets, 'Deviation', iteration);
    accuracies = loadDeviations(fullPath, numDatasets, 'Accuracy', iteration);
   title = sprintf('Weighted Scores [%.2f, %.2f]', weights(1), weights(2));
    % Calculate and plot weighted scores
    plotWeightedScores(deviations, accuracies, weights, title, plotType);
    % plotWeightedRankingEvolution(deviations, accuracies, weights)
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
>>>>>>> 9c6ccc124 (Reinicializando el repositorio)

    for i = 1:numIterations
        % Normalize the scores for the current iteration
        [normDev, normAcc] = normalizeScores(deviations(:, i), accuracies(:, i));
<<<<<<< HEAD
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

=======
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


% function plotWeightedScores(deviations, accuracies, weights, titleSuffix)
%     numDatasets = size(deviations, 1);
%     numIterations = size(deviations, 2);
%     scores = zeros(numDatasets, numIterations);
% 
%     % Define a set of colors for each dataset
%     customColors = [
%         1, 0, 0;  % Red
%         0, 1, 0;  % Green
%         0, 0, 1;  % Blue
%         1, 1, 0;  % Yellow
%         0, 1, 1;  % Cyan
%         1, 0, 1;  % Magenta
%         0.5, 0.5, 0.5;  % Gray
%         0, 0.5, 0;  % Dark Green
%         0.5, 0, 0.5  % Purple
%     ];
% 
%     % Define custom names for each plot
%     customNames = {
%         'Weighted Score A',
%         'Weighted Score B',
%         'Weighted Score C',
%         'Weighted Score D',
%         'Weighted Score E',
%         'Weighted Score F',
%         'High Weighted Score G',
%         'Varied Score H',
%         'Final Score I'
%     };
% 
%     markers = {'o', 's', '^', 'v', '>', '<', 'p', 'h', '+', 'x'};
% 
%     for i = 1:numIterations
%         % Normalize the scores for the current iteration
%         [normDev, normAcc] = normalizeScores(deviations(:, i), accuracies(:, i));
%         scores(:, i) = weights(1) * normDev + weights(2) * normAcc;
%     end
% 
%     % Calculate the rankings for each dataset at each iteration
%     rankings = zeros(size(scores));
%     for i = 1:numIterations
%         [~, inds] = sort(scores(:, i), 'descend');
%         rankings(inds, i) = linspace(1, numDatasets, numDatasets);
%     end
% 
%     figure;
%     hold on;
%     % Plotting the rankings for each dataset
%     for j = 1:numDatasets
%         datasetRankings = rankings(j, :);
%         plot(1:numIterations, datasetRankings, 'Color', customColors(j, :), ...
%             'Marker', markers{mod(j-1, length(markers)) + 1}, 'DisplayName', ...
%             customNames{j});  % Using custom names for display in legend
%     end
% 
%     xlabel('Iteration');
%     ylabel('Ranking Position');
%     title(['Weighted Scores Ranking Comparison: ', titleSuffix]);
%     legend('show', 'Location', 'southeast');  % Positioning legend at the bottom right
%     grid on;
%     set(gca, 'YDir', 'reverse');  % Invert the Y-axis
%     hold off;
% end




% function plotWeightedRankingEvolution(deviations, accuracies, weights)
%     numDatasets = size(deviations, 1);
%     numIterations = size(deviations, 2);
%     weightedRanks = zeros(numDatasets, numIterations);
% 
%     for i = 1:numIterations
%         % Rank the deviations and accuracies separately
%         [~, rankFidelityIndices] = sort(deviations(:, i), 'ascend');  % Assuming lower deviations are better
%         [~, rankAccuracyIndices] = sort(accuracies(:, i), 'descend');  % Assuming higher accuracies are better
% 
%         % Convert indices to rank positions
%         rankFidelities = zeros(numDatasets, 1);
%         rankAccuracies = zeros(numDatasets, 1);
%         rankFidelities(rankFidelityIndices) = 1:numDatasets;
%         rankAccuracies(rankAccuracyIndices) = 1:numDatasets;
% 
%         % Apply weights to ranks
%         weightedRanks(:, i) = weights(1) * rankFidelities + weights(2) * rankAccuracies;
%     end
% 
%     % Invert the weighted ranks so higher is better
%     invertedRanks = max(weightedRanks(:)) + 1 - weightedRanks;
% 
%     % Plotting the weighted rank evolution
%     figure;
%     hold on;
%     colors = lines(numDatasets); % Different colors for each dataset
%     markers = {'o', 's', '^', 'v', '>', '<', 'p', 'h', '+', 'x'}; % Different markers for lines
%     for j = 1:numDatasets
%         plot(1:numIterations, invertedRanks(j, :), 'Color', colors(j, :), 'Marker', markers{mod(j-1, length(markers)) + 1}, 'DisplayName', sprintf('Dataset %d', j));
%     end
%     xlabel('Iteration');
%     ylabel('Inverted Weighted Rank');
%     title('Evolution of Inverted Weighted Rankings Across Iterations');
%     legend show;
%     grid on;
%     hold off;
% end
% 

% function [normalizedDeviation, normalizedAccuracy] = normalizeScores(deviation, accuracy)
%     % Definir el punto medio y la pendiente para la función sigmoide
%     midpointDev = mean(deviation);
%     rangeDev = max(deviation) - min(deviation);
%     steepnessDev = 1 / max(0.001, rangeDev) * 4;  % Menos pronunciada
% 
%     midpointAcc = mean(accuracy);
%     rangeAcc = max(accuracy) - min(accuracy);
%     steepnessAcc = 1 / max(0.001, rangeAcc) * 5;  % Menos pronunciada
% 
%     % Limitar valores extremos, recorte suave
%     clippedDeviation = max(min(deviation, percentile(deviation, 80)), percentile(deviation, 15));
%     clippedAccuracy = max(min(accuracy, percentile(accuracy, 80)), percentile(accuracy, 5));
% 
%     % Ajustar la desviación sigmoide para que disminuya a medida que aumenta la desviación
%     sigmoidDeviation = 1 ./ (1 + exp(steepnessDev * (clippedDeviation - midpointDev)));
%     % La precisión de la sigmoide aumenta a medida que aumenta la precisión
%     sigmoidAccuracy = 1 ./ (1 + exp(-steepnessAcc * (clippedAccuracy - midpointAcc)));
% 
%     % Normalizar las desviaciones y precisiones después de la transformación sigmoide
%     minDev = min(sigmoidDeviation);
%     maxDev = max(sigmoidDeviation);
%     normalizedDeviation = (sigmoidDeviation - minDev) / (maxDev - minDev);
% 
%     minAcc = min(sigmoidAccuracy);
%     maxAcc = max(sigmoidAccuracy);
%     normalizedAccuracy = (sigmoidAccuracy - minAcc) / (maxAcc - minAcc);
% end
% 
% function p = percentile(data, k)
%     % Esta función calcula el k-ésimo percentil de los datos proporcionados
%     p = prctile(data, k);
% end


% function [normalizedDeviation, normalizedAccuracy] = normalizeScores(deviation, accuracy)
%     % Define the midpoint and steepness for the sigmoid function
%     midpointDev = mean(deviation);
%     rangeDev = max(deviation) - min(deviation);
%     steepnessDev = 1 / rangeDev * 4;  % Adjust steepness appropriately
% 
%     midpointAcc = mean(accuracy);
%     rangeAcc = max(accuracy) - min(accuracy);
%     steepnessAcc = 1 / rangeAcc * 10;
% 
%     % Adjust the deviation sigmoid to decrease as deviation increases
%     sigmoidDeviation = 1 ./ (1 + exp(steepnessDev * (deviation - midpointDev)));
%     % Accuracy sigmoid increases as accuracy increases
%     sigmoidAccuracy = 1 ./ (1 + exp(-steepnessAcc * (accuracy - midpointAcc)));
% 
%     % Normalize deviations and accuracies after sigmoid transformation
%     minDev = min(sigmoidDeviation);
%     maxDev = max(sigmoidDeviation);
%     normalizedDeviation = (sigmoidDeviation - minDev) / (maxDev - minDev);
% 
%     minAcc = min(sigmoidAccuracy);
%     maxAcc = max(sigmoidAccuracy);
%     normalizedAccuracy = (sigmoidAccuracy - minAcc) / (maxAcc - minAcc);
% end


function [normalizedDeviation, normalizedAccuracy] = normalizeScores(deviation, accuracy)
    % Normalizing deviation so that lower values are better
    minDev = min(deviation);
    maxDev = max(deviation);
    normalizedDeviation = 1 - (deviation - minDev) / (maxDev - minDev);  % Inverting the scale

    % Normalizing accuracy so that higher values are better
>>>>>>> 9c6ccc124 (Reinicializando el repositorio)
    minAcc = min(accuracy);
    maxAcc = max(accuracy);
    normalizedAccuracy = (accuracy - minAcc) / (maxAcc - minAcc);
end

<<<<<<< HEAD
function deviations = loadDeviations(fullPath, numFiles, mode)
=======
% function [rankFidelity, rankAccuracy] = rankScores(deviation, accuracy)
%     % Ranking para fidelidad: los menos negativos obtienen mejor rango
%     [~, rankFidelity] = sort(deviation, 'ascend');
% 
%     % Ranking para precisión: asumiendo que valores más altos son mejores
%     [~, rankAccuracy] = sort(accuracy, 'ascend');
% end



function deviations = loadDeviations(fullPath, numFiles, mode, targetLength)
>>>>>>> 9c6ccc124 (Reinicializando el repositorio)
    % Determine the data field based on the mode
    if strcmp(mode, 'Deviation')
        dataField = 'average_deviation';
    else
        dataField = 'average_accuracy';
    end
    
<<<<<<< HEAD
    % Pre-load the first file to determine the number of iterations
    firstFileData = load(fullfile(fullPath, sprintf('Temp_%s_PI_1.mat', mode)));
    if isfield(firstFileData, dataField)
        numIterations = numel(firstFileData.(dataField));  % Determine the length of the data vector
    else
        error(['Field "', dataField, '" does not exist in the loaded file.']);
    end
    
    % Initialize matrix to store deviations or accuracies
    deviations = zeros(numFiles, numIterations);  % Adjusted to handle dynamic number of iterations
=======
    % Initialize matrix to store deviations or accuracies
    deviations = zeros(numFiles, targetLength);
>>>>>>> 9c6ccc124 (Reinicializando el repositorio)
    
    % Load each file and extract the relevant data
    for i = 1:numFiles
        fileData = load(fullfile(fullPath, sprintf('Temp_%s_PI_%d.mat', mode, i)));
        if isfield(fileData, dataField)
<<<<<<< HEAD
            deviations(i, :) = fileData.(dataField);
=======
            dataVector = fileData.(dataField);
            if length(dataVector) >= targetLength
                deviations(i, :) = dataVector(1:targetLength);
            else
                error('Data vector in file %d is shorter than the target length.', i);
            end
>>>>>>> 9c6ccc124 (Reinicializando el repositorio)
        else
            error(['Field "', dataField, '" does not exist in file ', num2str(i), '.']);
        end
    end
end
<<<<<<< HEAD

=======
>>>>>>> 9c6ccc124 (Reinicializando el repositorio)
