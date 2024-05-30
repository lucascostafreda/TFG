function [unifiedMatrix, chromosomeData, genTracking] = processGenerationData(allDecodedPopulations, allFitnessValues, allDeviationValues, allAccuracyValues, fullPath, iGeneration)    
% unifiedMatrix, chromosomeData, genTracking,

unifiedMatrix = []; % This will hold all chromosomes and their fitness and accuracy values
genTracking = {}; % Cell array to track generation numbers for each chromosome
chromosomeData = struct('key', {}, 'data', {}, 'fitness', {}, 'deviation', {}, 'accuracy', {}, 'generationInfo', {}, 'averageFitness', {}, 'averageDeviation', {}, 'averageAccuracy', {});

numberOfGenerations = numel(allDecodedPopulations);
    for iGen = 1:numberOfGenerations
        currentDecodedPopulation = allDecodedPopulations{iGen};
        currentFitnessValues = allFitnessValues{iGen};
        % Añadir DEVIATIONS (ESTE)
        currentDeviationValues = allDeviationValues{iGen};
        currentAccuracyValues = allAccuracyValues{iGen};

        % Assuming the chromosome population only has 4, duplicate to make it 8
        % DUPLICATE EACH CHROMOSOME
        currentDecodedPopulation = repmat(currentDecodedPopulation, 1, 2); 

        tempMatrix = [currentDecodedPopulation, currentFitnessValues, currentDeviationValues, currentAccuracyValues]; % Properly include accuracy values in the matrix
        
        for iRow = 1:size(tempMatrix, 1)
            tempRowChromosome = tempMatrix(iRow, 1:end-3); % Exclude fitness and accuracy from the chromosome data
            tempFitness = tempMatrix(iRow, end-2);  % The second last entry is fitness
            tempDeviation =  tempMatrix(iRow, end-1);
            tempAccuracy = tempMatrix(iRow, end);  % The last entry is accuracy

            chromosomeKey = mat2str(tempRowChromosome);
            
            % Append this chromosome to the unifiedMatrix
            unifiedMatrix = [unifiedMatrix; tempMatrix(iRow, :)];
            genTracking{end+1} = [iGen]; % Track generation
            
            % Check if this chromosome key already exists
            idx = find(arrayfun(@(x) strcmp(x.key, chromosomeKey), chromosomeData));
            generationPosition = sprintf('Gen %d, Pos %d', iGen, iRow);
            if isempty(idx)
                % New chromosome, create new entry
                chromosomeData(end+1).key = chromosomeKey;
                chromosomeData(end).data = tempRowChromosome;
                chromosomeData(end).fitness = tempFitness;
                chromosomeData(end).deviation = tempDeviation;
                chromosomeData(end).accuracy = tempAccuracy;
                chromosomeData(end).generationInfo = {generationPosition};
                chromosomeData(end).averageFitness = tempFitness;  % Initial average fitness
                chromosomeData(end).averageDeviation = tempDeviation;  % Initial average accuracy
                chromosomeData(end).averageAccuracy = tempAccuracy;  % Initial average accuracy
            else
                % Existing chromosome, update data and fitness
                chromosomeData(idx).fitness = [chromosomeData(idx).fitness, tempFitness];
                chromosomeData(idx).deviation = [chromosomeData(idx).deviation, tempDeviation];
                chromosomeData(idx).accuracy = [chromosomeData(idx).accuracy, tempAccuracy];
                chromosomeData(idx).generationInfo{end+1} = generationPosition;
                chromosomeData(idx).averageFitness = mean(chromosomeData(idx).fitness);  % Update average fitness
                chromosomeData(idx).averageDeviation = mean(chromosomeData(idx).deviation);  % Update average fitness
                chromosomeData(idx).averageAccuracy = mean(chromosomeData(idx).accuracy);  % Update average accuracy
            end
        end
    end
    
    % Update fitness, deviation, and accuracy values in unifiedMatrix to their averages
    for iRow = 1:size(unifiedMatrix, 1)
        tempRowChromosome = unifiedMatrix(iRow, 1:end-3);  % Exclude fitness, accuracy, and deviation from the data
        chromosomeKey = mat2str(tempRowChromosome);
        idx = find(arrayfun(@(x) strcmp(x.key, chromosomeKey), chromosomeData));
        if ~isempty(idx)
            % Use average values from chromosomeData
            unifiedMatrix(iRow, end-2) = chromosomeData(idx).averageFitness;   % Set the fitness value to the average
            unifiedMatrix(iRow, end-1) = chromosomeData(idx).averageDeviation; % Set the deviation value to the average
            unifiedMatrix(iRow, end) = chromosomeData(idx).averageAccuracy;    % Set the accuracy value to the average
        end
    end

% Convert genTracking to a text representation
genTrackingStr = cellfun(@(x) mat2str(x), genTracking, 'UniformOutput', false);
genTrackingStr = genTrackingStr.';
% Create a table that includes unifiedMatrix and generation information
finalTable = array2table(unifiedMatrix, 'VariableNames', {'Chromosome1', 'Chromosome2', 'Chromosome3', 'Chromosome4', 'Chromosome5', 'Chromosome6', 'Chromosome7', 'Chromosome8', 'Fitness', 'Deviation', 'Accuracy'});
finalTable.Generation = genTrackingStr;

subfolderName = 'GenerationData';

savePath = fullfile(fullPath, subfolderName);
if ~exist(savePath, 'dir')
    mkdir(savePath); % Create the directory if it doesn't exist
end

% Save the lists of repeated chromosomes and other data
saveFileName = fullfile(savePath, sprintf('UnifiedSortedChromosomes_Gen%d.mat', iGeneration));
save(saveFileName, 'finalTable', 'chromosomeData');
end
