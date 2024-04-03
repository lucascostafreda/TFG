%% CLEAN-UP
clear; close all; clc;
tic

%% PARAMETERS
available_RBs = 450; %CHANGE
% r = [196.3993,100.4880,61.7281,43.2719, 196.3993,100.4880,61.7281,43.2719];
global KK;
KK=0;

  %CHANGE
  CQI_indices = [12, 10, 8, 6, 12, 10, 8, 6];
  [r, eta, sigma] = parameters(CQI_indices);
  RB_usedMatrix = [];

iteration = 1;
averagenumber = 1;
populationSize = 2;
numberOfGenes = 32; %CHANGE
crossoverProbability = 0.8;
mutationProbability = 0.0625;
tournamentSelectionParameter = 0.5;
numberOfGenerations = 1;
numberOfVariables = 8;%CHANGE
tournamentSize = 2;
numberOfReplications = 2;
verbose = true;
BitsXVariable = numberOfGenes / numberOfVariables; 

%% RUCHE OR NOT
ruche = false;

%% INITIALIZATION
[population, initDecodedPopul, individual_PI_RBs, averageRBsInit] = InitializePopulation(populationSize, numberOfGenes, numberOfVariables, r, available_RBs);
    
bestChromosomesList = []; % This will store each unique best chromosome found
lastBestChromosome = []; % Track the last best chromosome added

% To store data at various stages
allPopulations = cell(numberOfGenerations, 1);
allDecodedPopulations = cell(numberOfGenerations, 1);

allFitnessValues = cell(numberOfGenerations, 1);

%% RUN GENERATIONS
for iGeneration = 1:numberOfGenerations
    % Decode Population
    decodedPopulation = DecodePopulation(population, numberOfVariables, BitsXVariable);

    % Registrar chromosomas
    allPopulations{iGeneration} = population;
    allDecodedPopulations{iGeneration} = decodedPopulation;

    % Evaluate Population
    [fitness, RB_used] = EvaluatePopulation(decodedPopulation, r, available_RBs, iteration, averagenumber, ruche);
    
    RB_usedMatrix=[RB_usedMatrix,RB_used];

    %%%%%%%%%%%%%%%%%%%% TO TRY WITHOUT FL %%%%%%%%%%%%%%%%%%%%%%
    % for iGeneration=1:numberOfGenerations
    % load('UnifiedSortedChromosomes.mat');
    % randIndices = randperm(size(sortedUnifiedMatrix, 1), populationSize);
    % decodedPopulation = sortedUnifiedMatrix(randIndices, 1:8);
    % for i = 1:populationSize
    %     population(i, :) = EncodeChromosome(decodedPopulation(i, :), numberOfVariables, numberOfGenes);
    % end
    % 
    % fitness = sortedUnifiedMatrix(randIndices, 9);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 %% AFTER EVALUATION... 
    [maximumFitness, bestIndividualIndex] = max(fitness);

    % Registrar bestChromosoma
    bestChromosome = population(bestIndividualIndex,:);
    bestDecodedChromosome = DecodeChromosome(bestChromosome, numberOfVariables);
    allFitnessValues{iGeneration} = fitness;

    % Keep track of best chromosomes
    % Check if this bestChromosome is different from the last added
    if isempty(lastBestChromosome) || ~isequal(bestDecodedChromosome, lastBestChromosome)
        bestChromosomesList = [bestChromosomesList; bestDecodedChromosome]; % Add it to the list % Maybe could add the index of the generation it was created
        lastBestChromosome = bestDecodedChromosome; % Update the last best chromosome
    end

    newPopulation = population;

    %% TOURNAMENT, CROSSOVER, MUTATION
    for i = 1:2:populationSize
        % Tournament Selection
        i1 = TournamentSelect(fitness, tournamentSelectionParameter, tournamentSize);
        i2 = TournamentSelect(fitness, tournamentSelectionParameter, tournamentSize); % Ensure different selection
        chromosome1 = population(i1,:);
        chromosome2 = population(i2,:);
        
        % Crossover
        if rand < crossoverProbability
            newChromosomePair = Cross(chromosome1, chromosome2, r, available_RBs, numberOfVariables);
            newPopulation(i,:) = newChromosomePair(1,:);
            newPopulation(i+1,:) = newChromosomePair(2,:);
        else
            newPopulation(i,:) = chromosome1;
            newPopulation(i+1,:) = chromosome2;
        end
    end
    % Mutation
    newPopulation = Mutate(newPopulation, mutationProbability);
    
    
    % Insert Best Individual
    % bestChromosome = population(bestIndividualIndex,:);
    newPopulation = InsertBestIndividual(newPopulation, bestChromosome, numberOfReplications);
    
    % COPY THE NEW POPULATION ONTO CURRENT POPULATION
    population = newPopulation;
    % end
end

%% SAVE TABLES AND RESULTS INSTEAD OF DISPLAYING
% Assuming verbose is set to true if you want to save the results
if verbose
    %% BEST CHROMOSOME
    if ruche 
        save(fullfile('/gpfs/workdir/costafrelu/', 'BestChromosomes.mat'), 'bestChromosomesList');    
    else
        save(fullfile('..\workdir\', 'BestChromosomes.mat'), 'bestChromosomesList');
    end
    fprintf('File BestChromosomes.mat has been successfully saved');

    %% SAVING THE SORTED AND UNIFIED MATRIX and RB_USED
    % INITIALIZATION FOR UNIFIED MATRIX AND GENERATION TRACKING
    unifiedMatrix = []; % This will hold unique chromosomes and their highest fitness values
    genTracking = {}; % Cell array to track generation numbers for each chromosome
    
    % PROCESSING AND UNIFYING CHROMOSOMES FROM ALL GENERATIONS
    for iGen = 1:numberOfGenerations
        currentDecodedPopulation = allDecodedPopulations{iGen};
        currentFitnessValues = allFitnessValues{iGen};
        currentFitnessValues = currentFitnessValues'; % Transpose
        % Combine current generation's decoded population with fitness values
        tempMatrix = [currentDecodedPopulation, currentFitnessValues];
    
        % Iterate through each row in tempMatrix to ensure chromosome uniqueness
        for iRow = 1:size(tempMatrix, 1)
            tempRowChromosome = tempMatrix(iRow, 1:end-1); % Extract the chromosome part, excluding the fitness value
            tempFitness = tempMatrix(iRow, end); % Extract the fitness value
            
            % Check if the chromosome part of this row already exists in the chromosome part of unifiedMatrix
            idx = find(arrayfun(@(idx) isequal(tempRowChromosome, unifiedMatrix(idx, 1:end-1)), 1:size(unifiedMatrix, 1)));
            
            if isempty(idx)
                % Chromosome is unique, add it to the unifiedMatrix and track its generation
                unifiedMatrix = [unifiedMatrix; tempMatrix(iRow, :)];
                genTracking{end+1} = [iGen]; % Start tracking generations for this chromosome
            else
                % Chromosome exists, check if the new fitness is higher
                if tempFitness > unifiedMatrix(idx, end)
                    unifiedMatrix(idx, end) = tempFitness; % Update with the higher fitness value
                end
                % Append the current generation to the tracking list for this chromosome
                genTracking{idx} = [genTracking{idx}, iGen];
            end
        end
    end
    
    % SORTING THE UNIFIED MATRIX BASED ON FITNESS VALUES
    [~, sortIndex] = sort(unifiedMatrix(:, end), 'descend');
    sortedUnifiedMatrix = unifiedMatrix(sortIndex, :);
    
    % Save the sortedUnifiedMatrix with a new structure that includes generation tracking
    finalMatrix = {sortedUnifiedMatrix, genTracking};
    
    if ruche
        % Save paths for UNIX-like file system
        save(fullfile('/gpfs/workdir/costafrelu/', 'UnifiedSortedChromosomes.mat'), 'finalMatrix');
        save('/gpfs/workdir/costafrelu/RB_usedMatrix.mat', 'RB_usedMatrix');
    else
        % Save paths for Windows file system
        save(fullfile('..\workdir', 'UnifiedSortedChromosomes.mat'), 'finalMatrix');
        save('..\workdir\RB_usedMatrix.mat', 'RB_usedMatrix');
    end
    
    fprintf('Files UnifiedSortedChromosomes.mat and RB_usedMatrix.mat have been successfully saved.\n');
    
    end


toc
