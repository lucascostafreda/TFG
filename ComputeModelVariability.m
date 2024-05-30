function [avgParams,iterationVariability] = ComputeModelVariability(numRepetitions, filenamePattern, iterations, fullPath_baseDir, fullPath_tempDir)
    % Step 1: Calculate average parameters (this part is similar to the original function)
    avgParams = AveragedRefModel(numRepetitions, filenamePattern, iterations, fullPath_baseDir, fullPath_tempDir); % Assume this function now returns avgParams directly

    % Initialize an array to store the variability measure for each iteration
    iterationVariability = zeros(1, iterations);

    % Step 2 & 3: Compute and aggregate Frobenius norms
    for iter = 1:iterations
        frobNorms = zeros(1, numRepetitions); % To store Frobenius norms for this iteration

        for rep = 1:numRepetitions
            filename = fullfile(fullPath_tempDir, sprintf(filenamePattern, 1));
            loadedData = load(filename);
            frobNorm = 0; % Initialize sum of Frobenius norms for this repetition

            % Calculate the Frobenius norm of the difference between the loaded parameters and the average parameters
            for layer = 1:5 % Assuming 5 layers as before
                wLayerName = ['globalw', num2str(layer)];
                bLayerName = ['globalb', num2str(layer)];

                frobNorm = frobNorm + ...
                    norm(loadedData.allParams(iter).(wLayerName) - avgParams(iter).(wLayerName), 'fro')^2 + ...
                    norm(loadedData.allParams(iter).(bLayerName) - avgParams(iter).(bLayerName), 'fro')^2;
            end

            frobNorms(rep) = sqrt(frobNorm); % Take square root to get the Frobenius norm
        end

        % Step 3: Aggregate the Frobenius norms for this iteration
        % You can choose to average them or compute their standard deviation
        iterationVariability(iter) = std(frobNorms); % Using standard deviation as the measure of variability
    end

    % Optionally, you can plot or otherwise output iterationVariability here
    % fprintf('Iteration Variability (Standard Deviation of Frobenius Norms):\n');
    % disp(iterationVariability);

    % Specify the filename for saving the results
    variabilityFilename = sprintf('ModelVariability_%dRepetitions_%dIterations.mat', numRepetitions, iterations);
    variabilityFilename = fullfile(fullPath_baseDir, variabilityFilename);
    save(variabilityFilename, 'iterationVariability');
end
