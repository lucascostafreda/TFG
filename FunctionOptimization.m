%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%   Alp Sayin - alpsayin[at]alpsayin[dot]com - https://alpsayin.com
%   Matlab Genetic Algorithm
%   Spring 2012
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% CLEAN-UP
clear;close all;clc;
tic
%% PARAMETERS

available_RBs = 450;
% r = [43.2719, 61.7281, 100.4880, 196.3993, 43.2719, 61.7281, 100.4880, 196.3993];
r = [196.3993,100.4880,61.7281,43.2719, 196.3993,100.4880,61.7281,43.2719];
iteration = 1 ;
averagenumber = 1 ;

populationSize = 2; %cantidad de PI config
%numberOfGenes = 8; % 8 (2x4 bits)
numberOfGenes = 32;
crossoverProbability = 0.8 ;
mutationProbability = 0.0625;
tournamentSelectionParameter = 0.5;
numberOfGenerations = 2;
%numberOfVariables = 2;
numberOfVariables = 8;
tournamentSize = 2; %cambiar tb
numberOfReplications = 2;
verbose = true;
draw_plots = true;
% UNLESS THE FITNESS FUNCTION IS REALLY DIFFICULT TO COMPUTE, IT'S FASTER
% NOT TO USE PARALLEL COMPUTATION
runparallel = false; 

historicalMatrix=zeros(populationSize,2);
numberOfBits = numberOfGenes/numberOfVariables;

%% VARIABLES
fitness = zeros(populationSize, 1);

allPopulations = cell(numberOfGenerations, 1); % Store populations for each generation
allFitnessValues = cell(numberOfGenerations, 1); % Store fitness values for each generation

historicalTable = table([], [], [], 'VariableNames', {'Genes', 'DecodedValues', 'Fitness'});

%% PLOTTING SETUP


%% INITIATE POPULATION
[population, initDecodedPopulation, individualRBs, averageRBsInit] = InitializePopulation(populationSize, numberOfGenes, numberOfVariables, r, available_RBs); % possible PI config,

%% RUN GENERATIONS
for iGeneration = 1: numberOfGenerations % 2 (numbOfGenerations)* 15 (PI config) * numIter (dentro FL) * averaging
    
    %% FIND MAXIMUM FITNESS OF POPULATION

    decodedPopulation = DecodePopulation(population, numberOfVariables, numberOfBits);
    fitness = EvaluatePopulation(decodedPopulation, runparallel, r, available_RBs, iteration, averagenumber); %FL Model
    [maximumFitness, bestIndividualIndex] = max(fitness);
    xBest = decodedPopulation(bestIndividualIndex,:);

    % Print out
    if verbose
        fprintf('Maximum Fitness: %d\n',maximumFitness);
        fprintf('Best Solution: %d\n',xBest); 

    end

    %% register data
    allPopulations{iGeneration} = decodedPopulation;
    allFitnessValues{iGeneration} = fitness;
    
    for iInd = 1:populationSize
        individualGenes = population(iInd, :);
        individualDecoded = decodedPopulation(iInd, :);
        individualFitness = fitness(iInd);
        
        % Convert genes to a string format to use as a unique identifier
        geneStr = mat2str(individualGenes);
    
        % Find the index of the geneStr in the his  toricalTable, if it exists
        rowIndex = find(strcmp(historicalTable.Genes, geneStr));
    
        if isempty(rowIndex)
            % If this set of genes does not exist in the table, add a new row
            newRow = {geneStr, individualDecoded, individualFitness};
            historicalTable = [historicalTable; newRow];
        else
            % If this set of genes exists and the new fitness is greater than the stored one, update it
            if individualFitness > historicalTable.Fitness(rowIndex)
                historicalTable.Fitness(rowIndex) = individualFitness;
            end
        end
    end



    %% COPY POPULATION
    newPopulation = population;

% PONER RESTRICCIONES PARA MEJORES ACTUALIZACIONES

    %% NEW GENERATION
    for i = 1:tournamentSize:populationSize
        %% TOURNAMENT SELECTION
        i1 = TournamentSelect(fitness,tournamentSelectionParameter,tournamentSize);
        i2 = TournamentSelect(fitness,tournamentSelectionParameter,tournamentSize); %que no escoja el mismo
        chromosome1 = population(i1,:);
        chromosome2 = population(i2,:);

        %% CROSS-OVER
        r = rand;
        if ( r < crossoverProbability) 
                newChromosomePair = Cross(chromosome1, chromosome2, r, available_RBs, numberOfVariables); % con restricciones
                newPopulation(i,:) = newChromosomePair(1,:);
                newPopulation(i+1,:) = newChromosomePair(2,:);
        else
            newPopulation(i,:) = chromosome1;
            newPopulation(i+1,:) = chromosome2;
        end
    end

    %% MUTATE
    newPopulation = Mutate(newPopulation, mutationProbability);
    
    %% PRESERVATION OF PREVIOUS BEST SOLUTION
    bestChromosome = population(bestIndividualIndex,:);
    newPopulation = InsertBestIndividual(newPopulation, bestChromosome, numberOfReplications);
        
    %% COPY THE NEW POPULATION ONTO CURRENT POPULATION
    population = newPopulation;

end
    
 %% PLOT CURRENT SITUATION
if draw_plots
    
    for iGen = 1:numberOfGenerations
        fprintf('Generation %d:\n', iGen);
        currentPopulation = allPopulations{iGen}; % Access the population for the i-th generation
        currentFitnessValues = allFitnessValues{iGen}; % Access fitness values for the i-th generation
        for iInd = 1:populationSize
            fprintf('Individual %d: Genes: %s, Fitness: %.2f\n', iInd, mat2str(currentPopulation(iInd,:)), currentFitnessValues(iInd));
        end
    end

    % Display the entire table
    disp(historicalTable);
end
% Print out
% fprintf('Maximum Fitness: %d\n',maximumFitness);
% fprintf('Best Solution: %d\n',xBest); 
toc