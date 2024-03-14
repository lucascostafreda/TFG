function AveragedRefModel(numRepetitions, filenamePattern, iterations)
    % Initialize accumulators for all iterations
    accIterGlobalW = cell(iterations, 5); % For W1 through W5 for each iteration
    accIterGlobalB = cell(iterations, 5); % For B1 through B5 for each iteration

    % Initialize accumulators with zeros based on the first file to match dimensions
    firstFileData = load(sprintf(filenamePattern, 1));
    for iter = 1:iterations
        for layer = 1:5
            accIterGlobalW{iter, layer} = zeros(size(firstFileData.allParams(iter).(['globalw', num2str(layer)])));
            accIterGlobalB{iter, layer} = zeros(size(firstFileData.allParams(iter).(['globalb', num2str(layer)])));
        end
    end
    %%%%%%%%%%%%%%%%%%% Printing weights %%%%%%%%%%%%%%%%%%%%%%%%%
    % Print W1(1,1) for the first iteration of each repetition
    fprintf('\nValues of W1(1,1) for all repetitions in Iteration 1:\n');
    for rep = 1:numRepetitions
        filename = sprintf(filenamePattern, rep);
        loadedData = load(filename);
        fprintf('Repetition %d: %f\n', rep, loadedData.allParams(1).globalw1(1,1));
    end

    % Print W1(1,1) for the last iteration of each repetition
    fprintf('\nValues of W1(1,1) for all repetitions in Last Iteration (%d):\n', iterations);
    for rep = 1:numRepetitions
        filename = sprintf(filenamePattern, rep);
        loadedData = load(filename);
        fprintf('Repetition %d: %f\n', rep, loadedData.allParams(iterations).globalw1(1,1));
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Accumulate parameters from each file for corresponding iterations
    for rep = 1:numRepetitions
        filename = sprintf(filenamePattern, rep);
        loadedData = load(filename);

        for iter = 1:iterations
            for layer = 1:5
                accIterGlobalW{iter, layer} = accIterGlobalW{iter, layer} + loadedData.allParams(iter).(['globalw', num2str(layer)]);
                accIterGlobalB{iter, layer} = accIterGlobalB{iter, layer} + loadedData.allParams(iter).(['globalb', num2str(layer)]);
            end
        end
    end

    % Average the accumulated parameters for each iteration
    avgParams = repmat(struct, iterations, 1); % Initialize structure array for averaged parameters
    for iter = 1:iterations
        for layer = 1:5
            avgParams(iter).(['globalw', num2str(layer)]) = accIterGlobalW{iter, layer} / numRepetitions;
            avgParams(iter).(['globalb', num2str(layer)]) = accIterGlobalB{iter, layer} / numRepetitions;
        end
    end

    % Print averaged value for W1(1,1) for the first iteration
    fprintf('\nAveraged W1(1,1) for Iteration 1: %f\n', avgParams(1).globalw1(1,1));
    
    % Print averaged value for W1(1,1) for the last iteration
    fprintf('Averaged W1(1,1) for Last Iteration (%d): %f\n\n', iterations, avgParams(iterations).globalw1(1,1));

    % Save the averaged parameters for all iterations
    filename = sprintf('Ref_Model_%d_i_%d_r.mat', iterations,numRepetitions);
    save(filename, 'avgParams');
end
