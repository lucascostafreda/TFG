%% CLEAN-UP
clear; close all; clc;
%tic

%% RUCHE OR NOT
ruche = true;

%% PARAMETERS
available_RBs = 450; %CHANGE
% r = [196.3993,100.4880,61.7281,43.2719, 196.3993,100.4880,61.7281,43.2719];
% optimalEstimatedPI=[1,1,0.793,0.205,1,1,0.793,0.205]; % aproximadamente
global KK;
global KKK;

KKK = 0;
KK=0;

  CQI_indices = [12, 10, 8, 6, 12, 10, 8, 6];
  [r, eta, sigma] = parameters(CQI_indices);
  RB_usedMatrix = [];

  weights=[0.5,0.5]; % [fidelity, accuracy]

%% FL environment
iteration = 1;
averagenumber = 1;
verboseFL = true;
miniBatchSize = 100;
executionEnviroment ='parallel';
AccDevMat = false;
Shuffle='never';
% partNumber
% 1 (1), 3/4 (2), 1/2 (3),  1/4 (4), 1/8 (5), 1/16 (6)s
FragSDS = 1;
percentages = [1/8,  1/4,  1/2,  1, 1/8,  1/4, 1/2, 1];

%% evaluation param
% Mapa para rastrear las repeticiones de cada cromosoma
chromosomeRepetitions = containers.Map  ('KeyType', 'char', 'ValueType', 'int32');
historicalChromosomeFitness = containers.Map('KeyType', 'char', 'ValueType', 'any');
historicalChromosomeAccuracy = containers.Map('KeyType', 'char', 'ValueType', 'any');

%% GA
populationSize = 4;
numberOfGenerations = 2;
adjustPIEpoch = 0;

%
numberOfGenes = 40; 
crossoverProbability = 0.8;
mutationProbability = 0.0625;
tournamentSelectionParameter = 0.5;
numberOfVariables = 8;
tournamentSize = 2;
numberOfReplications = 2;
verbose = true;
BitsXVariable = numberOfGenes / numberOfVariables; 
PIprocessed=numberOfGenerations*populationSize;


%% Paths and Adresses 

  if ruche
      refModelName = 'Ref_Model_15_i_15_r_noQ';
      directory_RefModel = 'refModel_15_i_15_withFragSDS';  
      baseDir_functionOpt = '/gpfs/workdir/costafrelu/FunctionOptimization_noQ_sameDS/';
  else
      refModelName = 'Ref_Model_1_i_1_r_noQ';
      directory_RefModel = 'refModel_1_i_1';
      baseDir_functionOpt = '..\workdir\FunctionOptimization\noQ_sameDS';
  end
directory_tempDir = sprintf('i_%d_avg_%d', iteration, averagenumber);
directory_FO_Trial_Config = sprintf('generation%d_popu%d_ite%d_avg%d_JOINT', numberOfGenerations, populationSize, iteration, averagenumber);

fullPath = createVersionedDirectory(baseDir_functionOpt, directory_FO_Trial_Config);

% Confirm the directory creation
fprintf('Directory created at: %s\n', fullPath);

%% INITIALIZATION
[population, initDecodedPopul, individual_PI_RBs, averageRBsInit] = InitializePopulation(populationSize, numberOfGenes, numberOfVariables, r, available_RBs);
    
bestChromosomesList = []; % This will store each unique best chromosome found
lastBestChromosome = []; % Track the last best chromosome added

% To store data at various stages
allPopulations = cell(numberOfGenerations, 1);
allDecodedPopulations = cell(numberOfGenerations, 1);
allFitnessValues = cell(numberOfGenerations, 1);
allAccuracyValues = cell(numberOfGenerations, 1);
allDeviationValues = cell(numberOfGenerations, 1);

timePerGen=zeros(numberOfGenerations,1);

%% RUN GENERATIONS
for iGeneration = 1:numberOfGenerations
    % Decode Population
    decodedPopulation = DecodePopulation(population, numberOfVariables, BitsXVariable);

    % Registrar chromosomas
    allPopulations{iGeneration} = population;
    allDecodedPopulations{iGeneration} = decodedPopulation;
    

    %% Evaluate Population
    % No evaluar al best chromosome +1 vez
   %% Evaluate Population
    [fidelityFitness, accuracyFitness, RB_used, tiempo] = EvaluatePopulation_joint(decodedPopulation, r, available_RBs,...
        iteration, averagenumber, ruche, verboseFL, miniBatchSize, executionEnviroment, AccDevMat,...
        allDecodedPopulations, allDeviationValues, allAccuracyValues, Shuffle, refModelName, directory_RefModel, directory_tempDir, ...
        historicalChromosomeFitness, historicalChromosomeAccuracy, chromosomeRepetitions, FragSDS, percentages, iGeneration);
        
    allDeviationValues{iGeneration} = fidelityFitness;
    allAccuracyValues{iGeneration} = accuracyFitness;

    RB_usedMatrix = vertcat(RB_usedMatrix, RB_used);
    timePerGen(iGeneration)=tiempo;

    %fitness = calculateTotalScore(fidelityFitness, accuracyFitness, weights); 
    allFitnessValues = calculateGlobalScores(allDeviationValues, allAccuracyValues, weights);
    fitness = allFitnessValues{iGeneration};

    % Save the fitness scores for the current generation.
    allFitnessArray = vertcat(allFitnessValues{:});
    [globalBestFitness, globalBestIndex] = max(allFitnessArray);
    
    % Determine which generation and local index the best fitness occurs at
    cumulativeSizes = [0; cumsum(cellfun(@numel, allFitnessValues))];  % Prepend 0 to handle indexing for the first generation
    generationForBest = find(cumulativeSizes >= globalBestIndex, 1, 'first') - 1;  % Adjust to get the correct generation index
    indexInGeneration = globalBestIndex - cumulativeSizes(generationForBest);  % Adjust to get the local index within the generation
    
    % Now, generationForBest and indexInGeneration are correctly computed
    bestDecodedChromosome = allDecodedPopulations{generationForBest}(indexInGeneration, :);
    bestChromosome=EncodeChromosome(bestDecodedChromosome, numberOfVariables, numberOfGenes);

    % Update and track best chromosomes
    if isempty(lastBestChromosome) || ~isequal(bestDecodedChromosome, lastBestChromosome)
        bestChromosomesList = [bestChromosomesList; [bestDecodedChromosome, globalBestFitness]];
        lastBestChromosome = bestDecodedChromosome;
    end
    % Optionally log or display the best fitness found
    % fprintf('Best fitness: %f found in generation %d\n', currentBestFitness if iGeneration == 1 else globalBestFitness, generationForBest if iGeneration > 1 else 1);


%% Mejorar Chromosomas

    % if mod(iGeneration,adjustPIEpoch)==0
    % 
    %     sortedCombinedAllGenerations = CombineAndSortGenerations(allDecodedPopulations, allFitnessValues);
    % 
    %     numToSelect = floor(populationSize / 2); %ojo, que si es impar, seleccionará menos de la mitad, has de seleccionar uno pas para improve o GA
    %     decodedSelectedChromosomes_H = sortedCombinedAllGenerations(1:numToSelect, 1:end-1);
    % 
    %     if mod((populationSize/2),2)~=0
    %         decodedSelectedChromosomes_GA = sortedCombinedAllGenerations(1:numToSelect+1, 1:end-1);
    %         selectedFitness = sortedCombinedAllGenerations(1:numToSelect+1, end);
    %     else
    %         decodedSelectedChromosomes_GA = sortedCombinedAllGenerations(1:numToSelect, 1:end-1);
    %         selectedFitness = sortedCombinedAllGenerations(1:numToSelect, end);
    %     end        
    % 
    %     % Increment Chromosomes and encode them
    %     decodedImprovedChromosomes = IncrementalAdjustment(decodedSelectedChromosomes_H, r, available_RBs);
    %     improvedChromosomes = EncodeChromosomes(decodedImprovedChromosomes, numberOfVariables, numberOfGenes);
    % 
    %     % encode non-imporved chromosomes and apply GA operations
    %     selectedChromosomes = EncodeChromosomes(decodedSelectedChromosomes_GA, numberOfVariables, numberOfGenes);
    %     selectedChromosomes = ApplyGeneticOperators(selectedChromosomes, selectedFitness, tournamentSelectionParameter,...
    %         tournamentSize, crossoverProbability, mutationProbability, r, available_RBs, numberOfVariables,...
    %         bestChromosome, numberOfReplications); %añade un chromosoma de más!
    % 
    %     newPopulation = vertcat(improvedChromosomes, selectedChromosomes);
    % 
    % else

        newPopulation = ApplyGeneticOperators(population, fitness, tournamentSelectionParameter,...
            tournamentSize, crossoverProbability, mutationProbability, r, available_RBs, numberOfVariables,...
            bestChromosome, numberOfReplications);

    % end
    % COPY THE NEW POPULATION ONTO CURRENT POPULATION
    population = newPopulation;
    
    fprintf('\n -------- End Generation: %d --------- \n', iGeneration);
end

%% SAVE TABLES AND RESULTS INSTEAD OF DISPLAYING
% Assuming verbose is set to true if you want to save the results
if verbose
    %% BEST CHROMOSOME
    save(fullfile(fullPath, 'BestChromosomes.mat'), 'bestChromosomesList');
    save(fullfile(fullPath, 'timePerGen.mat'), 'timePerGen');
    fprintf('File BestChromosomes.mat has been successfully saved');

    %% SAVING THE SORTED AND UNIFIED MATRIX and RB_USED
    % INITIALIZATION FOR UNIFIED MATRIX AND GENERATION TRACKING
    unifiedMatrix = []; % This will hold all chromosomes and their fitness and accuracy values
    genTracking = {}; % Cell array to track generation numbers for each chromosome
    chromosomeData = struct('key', {}, 'data', {}, 'fitness', {}, 'deviation', {}, 'accuracy', {}, 'generationInfo', {}, 'averageFitness', {}, 'averageDeviation', {}, 'averageAccuracy', {});
    
    % PROCESSING AND UNIFYING CHROMOSOMES FROM ALL GENERATIONS
    numberOfGenerations = numel(allDecodedPopulations);
    for iGen = 1:numberOfGenerations
        currentDecodedPopulation = allDecodedPopulations{iGen};
        currentFitnessValues = allFitnessValues{iGen};
        % Añadir DEVIATIONS (ESTE)
        currentDeviationValues = allDeviationValues{iGen};
        currentAccuracyValues = allAccuracyValues{iGen};
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
    
    % Update fitness and accuracy values in unifiedMatrix to the last known values
    for iRow = 1:size(unifiedMatrix, 1)
        tempRowChromosome = unifiedMatrix(iRow, 1:end-3);  % Exclude fitness, accuracy and deviation from the data
        chromosomeKey = mat2str(tempRowChromosome);
        idx = find(arrayfun(@(x) strcmp(x.key, chromosomeKey), chromosomeData));
        lastFitness = chromosomeData(idx).fitness(end);
        lastDeviation = chromosomeData(idx).deviation(end);
        lastAccuracy = chromosomeData(idx).accuracy(end);
        unifiedMatrix(iRow, end-2) = lastFitness;  % Set the fitness value to the last recorded
        unifiedMatrix(iRow, end-1) = lastDeviation;  % Set the deviation value to the last recorded
        unifiedMatrix(iRow, end) = lastAccuracy;  % Set the accuracy value to the last recorded
    end
    
    % Convert genTracking to a text representation
    genTrackingStr = cellfun(@(x) mat2str(x), genTracking, 'UniformOutput', false);
    genTrackingStr = genTrackingStr.';
    % Create a table that includes unifiedMatrix and generation information
    finalTable = array2table(unifiedMatrix, 'VariableNames', {'Chromosome1', 'Chromosome2', 'Chromosome3', 'Chromosome4', 'Chromosome5', 'Chromosome6', 'Chromosome7', 'Chromosome8', 'Fitness', 'Deviation', 'Accuracy'});
    finalTable.Generation = genTrackingStr;
    
    % Save the lists of repeated chromosomes and other data
    save(fullfile(fullPath, 'UnifiedSortedChromosomes.mat'), 'finalTable', 'chromosomeData');

    save(fullfile(fullPath, 'RB_usedMatrix.mat'), 'RB_usedMatrix');

    fprintf('Files UnifiedSortedChromosomes.mat and RB_usedMatrix.mat have been successfully saved.\n');
    
end


function fullPath = createVersionedDirectory(baseDir, directory2)
    % Create the full path
    fullPath = fullfile(baseDir, directory2);

    % Check if the directory exists
    if exist(fullPath, 'dir')
        % If the directory exists, find the next version that does not
        version = 1;
        while true
            newPath = sprintf('%s_v%d', fullPath, version);
            if ~exist(newPath, 'dir')
                fullPath = newPath;
                break;
            end
            version = version + 1;
        end
    end
    % Create the directory
    mkdir(fullPath);
end

%toc

function allFitnessValues = calculateGlobalScores(allFidelity, allAccuracy, weights)
    % Consolidate all fidelity and accuracy data into single arrays
    consolidatedFidelity = vertcat(allFidelity{:});
    consolidatedAccuracy = vertcat(allAccuracy{:});
    
    % Normalize these consolidated data
    [normFid, normAcc] = normalizeGlobalScores(consolidatedFidelity, consolidatedAccuracy);
    
    % Calculate scores using normalized data and weights
    allScores = weights(1) * normFid + weights(2) * normAcc;
    
    % Split the scores back into their respective generations
    allFitnessValues = cell(size(allFidelity));
    index = 1;
    for iGen = 1:length(allFidelity)
        endIndex = index + size(allFidelity{iGen}, 1) - 1;
        allFitnessValues{iGen} = allScores(index:endIndex);
        index = endIndex + 1;
    end
end

function [normalizedFidelity, normalizedAccuracy] = normalizeGlobalScores(fidelity, accuracy)
    % Identify valid indices where fidelity is not -15 and accuracy is not 0
    validIndices = fidelity ~= -15 & accuracy ~= 0;
    
    % Initialize normalized scores
    normalizedFidelity = zeros(size(fidelity));
    normalizedAccuracy = zeros(size(accuracy));
    
    % Only proceed with normalization if there are valid entries
    if any(validIndices)
        validFidelity = fidelity(validIndices);
        validAccuracy = accuracy(validIndices);
        
        minFidelity = min(validFidelity);
        maxFidelity = max(validFidelity);
        normalizedFidelity(validIndices) = (validFidelity - minFidelity) / (maxFidelity - minFidelity);
        
        minAccuracy = min(validAccuracy);
        maxAccuracy = max(validAccuracy);
        normalizedAccuracy(validIndices) = (validAccuracy - minAccuracy) / (maxAccuracy - minAccuracy);
    end
end


