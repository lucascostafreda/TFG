%% CLEAN-UP
clear; close all; clc;
tic

%% PARAMETERS
available_RBs = 450;
% r = [196.3993,100.4880,61.7281,43.2719, 196.3993,100.4880,61.7281,43.2719];

  CQI_indices = [12, 10, 8, 6, 12, 10, 8, 6];
  [r, eta, sigma] = parameters(CQI_indices);
 
iteration = 1;
averagenumber = 1;
populationSize = 4;
numberOfGenes = 32;
crossoverProbability = 0.8;
mutationProbability = 0.0625;
tournamentSelectionParameter = 0.5;
numberOfGenerations = 2;
numberOfVariables = 8;
tournamentSize = 2;
numberOfReplications = 2;
verbose = true;
draw_plots = true;
runparallel = false;
numberOfBits = numberOfGenes / numberOfVariables; % igual que numberOfGenes

%% INITIALIZATION
% historicalMatrix = zeros(populationSize, 2);
%historicalTable = table([], [], [], 'VariableNames', {'Genes', 'DecodedValues', 'Fitness'});
[population, initDecodedPopul, individualRBs, averageRBsInit] = InitializePopulation(populationSize, numberOfGenes, numberOfVariables, r, available_RBs);
    
bestChromosomes = []; % This will store each unique best chromosome found
lastBestChromosome = []; % Track the last best chromosome added

% To store data at various stages
allPopulations = cell(numberOfGenerations, 1);
allDecodedPopulations = cell(numberOfGenerations, 1);

allFitnessValues = cell(numberOfGenerations, 1);
%intermediatePopulations = struct();

%% RUN GENERATIONS
for iGeneration = 1:numberOfGenerations
    % Decode Population
    decodedPopulation = DecodePopulation(population, numberOfVariables, numberOfBits);
    allPopulations{iGeneration} = population;
    allDecodedPopulations{iGeneration} = decodedPopulation;

    % Evaluate Population
    fitness = EvaluatePopulation(decodedPopulation, runparallel, r, available_RBs, iteration, averagenumber);
    [maximumFitness, bestIndividualIndex] = max(fitness);
    bestChromosome = population(bestIndividualIndex,:);

    % Check if this bestChromosome is different from the last added
    if isempty(lastBestChromosome) || ~isequal(bestChromosome, lastBestChromosome)
        bestChromosomes = [bestChromosomes; bestChromosome]; % Add it to the list
        lastBestChromosome = bestChromosome; % Update the last best chromosome
    end

    allFitnessValues{iGeneration} = fitness;
    % Saving initial state
    % intermediatePopulations(iGeneration).initialPopulation = population;
    % intermediatePopulations(iGeneration).initialFitness = fitness;
    
    newPopulation=population;

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
    
    % Evaluate Post-Mutation Population
    %decodedPopulation = DecodePopulation(population, numberOfVariables, numberOfBits);
    %fitness = EvaluatePopulation(decodedPopulation, runparallel, r, available_RBs, iteration, averagenumber);
    
    % Saving post-mutation state
    %intermediatePopulations(iGeneration).postMutationPopulation = population;
    %intermediatePopulations(iGeneration).postMutationDecodedPopulation = decodedPopulation;
    % intermediatePopulations(iGeneration).postMutationFitness = fitness;
    
    % Insert Best Individual
    bestChromosome = population(bestIndividualIndex,:);
    newPopulation = InsertBestIndividual(newPopulation, bestChromosome, numberOfReplications);
    
    % COPY THE NEW POPULATION ONTO CURRENT POPULATION
    population = newPopulation;

    % Store population and fitness
   
end

% %% PLOT AND DISPLAY RESULTS
% if verbose
%     for iGen = 1:numberOfGenerations
%         disp(['Generation ', num2str(iGen)]);
%         disp(allPopulations{iGen});
%         disp(allFitnessValues{iGen});
%     end 
% end

%% SAVE TABLES AND RESULTS INSTEAD OF DISPLAYING
% Assuming verbose is set to true if you want to save the results
if verbose
    filenameBestChromosomes = fullfile('C:\Users\Lucas\OneDrive\Documentos\MATLAB\TFG\workdir\', 'BestChromosomes.mat');
    save(filenameBestChromosomes, 'bestChromosomes');

    %% INITIALIZATION FOR UNIFIED MATRIX
    unifiedMatrix = []; % This will hold unique chromosomes from all generations and their fitness values

    %% PROCESSING AND UNIFYING CHROMOSOMES FROM ALL GENERATIONS
    for iGen = 1:numberOfGenerations
        currentDecodedPopulation = allDecodedPopulations{iGen};
        currentFitnessValues = allFitnessValues{iGen};

        % Combine current generation's decoded population with fitness values
        tempMatrix = [currentDecodedPopulation, currentFitnessValues];

        % Iterate through each row in tempMatrix to ensure chromosome uniqueness
        for iRow = 1:size(tempMatrix, 1)
            tempRowChromosome = tempMatrix(iRow, 1:end-1); % Extract the chromosome part, excluding the fitness value
            
            % Check if the chromosome part of this row already exists in the chromosome part of unifiedMatrix
            chromosomeExists = any(arrayfun(@(idx) isequal(tempRowChromosome, unifiedMatrix(idx, 1:end-1)), 1:size(unifiedMatrix, 1)));

            if ~chromosomeExists
                unifiedMatrix = [unifiedMatrix; tempMatrix(iRow, :)]; % Append the entire row if chromosome is unique
            end
        end
    end

    %% SORTING THE UNIFIED MATRIX BASED ON FITNESS VALUES
    % Assuming the fitness value is the last column
    [~, sortIndex] = sort(unifiedMatrix(:, end), 'descend');
    sortedUnifiedMatrix = unifiedMatrix(sortIndex, :);

    %% SAVING THE SORTED AND UNIFIED MATRIX
    filenameUnifiedSorted = fullfile('C:\Users\Lucas\OneDrive\Documentos\MATLAB\TFG\workdir\', 'UnifiedSortedChromosomes.mat');
    save(filenameUnifiedSorted, 'sortedUnifiedMatrix');
end


toc
