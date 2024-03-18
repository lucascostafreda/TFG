%% CLEAN-UP
clear; close all; clc;
tic

%% PARAMETERS
available_RBs = 450;
r = [196.3993,100.4880,61.7281,43.2719, 196.3993,100.4880,61.7281,43.2719];
iteration = 1;
averagenumber = 1;
populationSize = 5;
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
numberOfBits = numberOfGenes / numberOfVariables;

%% INITIALIZATION
historicalMatrix = zeros(populationSize, 2);
historicalTable = table([], [], [], 'VariableNames', {'Genes', 'DecodedValues', 'Fitness'});
[population, ~, ~, ~] = InitializePopulation(populationSize, numberOfGenes, numberOfVariables, r, available_RBs);

% To store data at various stages
allPopulations = cell(numberOfGenerations, 1);
allFitnessValues = cell(numberOfGenerations, 1);
intermediatePopulations = struct();

%% RUN GENERATIONS
for iGeneration = 1:numberOfGenerations
    % Decode Population
    decodedPopulation = DecodePopulation(population, numberOfVariables, numberOfBits);
    
    % Evaluate Population
    fitness = EvaluatePopulation(decodedPopulation, runparallel, r, available_RBs, iteration, averagenumber);
    [maximumFitness, bestIndividualIndex] = max(fitness);
    
    % Saving initial state
    intermediatePopulations(iGeneration).initialPopulation = population;
    intermediatePopulations(iGeneration).initialFitness = fitness;
    
    %% TOURNAMENT, CROSSOVER, MUTATION
    for i = 1:2:populationSize
        % Tournament Selection
        i1 = TournamentSelect(fitness, tournamentSelectionParameter, tournamentSize);
        i2 = TournamentSelect(fitness, tournamentSelectionParameter, tournamentSize); % Ensure different selection
        chromosome1 = population(i1,:);
        chromosome2 = population(i2,:);
        
        % Crossover
        if rand < crossoverProbability
            newChromosomePair = Cross(chromosome1, chromosome2, r, available_RBs, numberOfVariables, numberOfGenes);
            population(i,:) = newChromosomePair(1,:);
            population(i+1,:) = newChromosomePair(2,:);
        else
            population(i,:) = chromosome1;
            population(i+1,:) = chromosome2;
        end
    end
    % Mutation
    population = Mutate(population, mutationProbability);
    
    % Evaluate Post-Mutation Population
    decodedPopulation = DecodePopulation(population, numberOfVariables, numberOfBits);
    fitness = EvaluatePopulation(decodedPopulation, runparallel, r, available_RBs, iteration, averagenumber);
    
    % Saving post-mutation state
    intermediatePopulations(iGeneration).postMutationPopulation = population;
    intermediatePopulations(iGeneration).postMutationFitness = fitness;
    
    % Insert Best Individual
    bestChromosome = population(bestIndividualIndex,:);
    population = InsertBestIndividual(population, bestChromosome, numberOfReplications);
    
    % Store population and fitness
    allPopulations{iGeneration} = population;
    allFitnessValues{iGeneration} = fitness;
end

%% PLOT AND DISPLAY RESULTS
if verbose
    for iGen = 1:numberOfGenerations
        disp(['Generation ', num2str(iGen)]);
        disp(allPopulations{iGen});
        disp(allFitnessValues{iGen});
    end
end

toc
