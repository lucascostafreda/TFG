%% CLEAN-UP
clear; close all; clc;
%tic

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

iteration = 1;
averagenumber = 1;
verboseFL = true;
miniBatchSize = 50;
executionEnviroment ='parallel';
AccDevMat = false;

%%
populationSize = 2;
numberOfGenes = 32; 
crossoverProbability = 0.8;
mutationProbability = 0.0625;
tournamentSelectionParameter = 0.5;
numberOfGenerations = 1;
numberOfVariables = 8;
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

tiempoWithoutVerbose=[];
tiempoWithVerbose=[];

%% RUN GENERATIONS
for iGeneration = 1:numberOfGenerations
    % Decode Population
    decodedPopulation = DecodePopulation(population, numberOfVariables, BitsXVariable);

    % Registrar chromosomas
    allPopulations{iGeneration} = population;
    allDecodedPopulations{iGeneration} = decodedPopulation;

    % Evaluate Population
    % No evaluar al best chromosome +1 vez
    [fitness, RB_used, tiempo] = EvaluatePopulation(decodedPopulation, r, available_RBs, iteration, averagenumber, ruche, verboseFL, miniBatchSize, executionEnviroment, AccDevMat, allDecodedPopulations, allFitnessValues);
    
    if verboseFL
        tiempoWithVerbose=[tiempoWithVerbose,tiempo];
    else
        tiempoWithoutVerbose=[tiempoWithoutVerbose,tiempo];
    end

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
        bestChromosomesList = [bestChromosomesList; [bestDecodedChromosome,maximumFitness]]; % Add it to the list % Maybe could add the index of the generation it was created
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
    % HERE WE HAVE ALL CHROMOSOMES AJUSTED TO THE RESOURCE LIMITATION
    
    % Mutation
    newPopulation = Mutate(newPopulation, mutationProbability, r, available_RBs, numberOfVariables);
    
    
    % Insert Best Individual
    % bestChromosome = population(bestIndividualIndex,:);
    newPopulation = InsertBestIndividual(newPopulation, bestChromosome, numberOfReplications);
    
    % COPY THE NEW POPULATION ONTO CURRENT POPULATION
    population = newPopulation;
    
    fprintf('End Generation %d', iGeneration);
end

%% SAVE TABLES AND RESULTS INSTEAD OF DISPLAYING
% Assuming verbose is set to true if you want to save the results
if verbose
    %% BEST CHROMOSOME

    directory = sprintf('generation%d_popu%d_ite%d', numberOfGenerations, populationSize, iteration);
    
    if ruche 
        baseDir = '/gpfs/workdir/costafrelu/';
    else
        baseDir = '..\workdir';
    end
    
    fullPath = fullfile(baseDir, directory);
    
    if ~exist(fullPath, 'dir')
        mkdir(fullPath);
    end
    
    save(fullfile(fullPath, 'BestChromosomes.mat'), 'bestChromosomesList');
    fprintf('File BestChromosomes.mat has been successfully saved');

    %% SAVING THE SORTED AND UNIFIED MATRIX and RB_USED
    % INITIALIZATION FOR UNIFIED MATRIX AND GENERATION TRACKING
    unifiedMatrix = []; % This will hold unique chromosomes and their highest fitness values
    genTracking = {}; % Cell array to track generation numbers for each chromosome
    
    % PROCESSING AND UNIFYING CHROMOSOMES FROM ALL GENERATIONS
    numberOfGenerations = numel(allDecodedPopulations); % Asumiendo que esta variable está definida
    for iGen = 1:numberOfGenerations
        currentDecodedPopulation = allDecodedPopulations{iGen};
        currentFitnessValues = allFitnessValues{iGen};
        %currentFitnessValues = currentFitnessValues'; % Transpose
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

    % Convertir genTracking a una representación de texto
    genTrackingStr = cellfun(@(x) mat2str(x), genTracking, 'UniformOutput', false);
    genTrackingStr = genTrackingStr.';
    % Crear una tabla que incluya unifiedMatrix y la información de generación
    finalTable = array2table(unifiedMatrix, 'VariableNames', {'Chromosome1', 'Chromosome2', 'Chromosome3', 'Chromosome4', 'Chromosome5', 'Chromosome6', 'Chromosome7', 'Chromosome8', 'Fitness'});
    finalTable.Generation = genTrackingStr; % Agregar columna de generaciones

    
    % SORTING THE UNIFIED MATRIX BASED ON FITNESS VALUES
    % [~, sortIndex] = sort(unifiedMatrix(:, end), 'descend');
    % sortedUnifiedMatrix = unifiedMatrix(sortIndex, :);
    
    % Save the sortedUnifiedMatrix with a new structure that includes generation tracking
    % finalMatrix = {sortedUnifiedMatrix, genTracking};

    %finalMatrix = {unifiedMatrix, genTracking};
    
    save(fullfile(fullPath, 'UnifiedSortedChromosomes.mat'), 'finalTable');
    save(fullfile(fullPath, 'RB_usedMatrix.mat'), 'RB_usedMatrix');

    fprintf('Files UnifiedSortedChromosomes.mat and RB_usedMatrix.mat have been successfully saved.\n');
    
end


%toc
