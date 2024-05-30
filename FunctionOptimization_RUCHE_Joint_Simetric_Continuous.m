%% CLEAN-UP
clear; close all; clc;
%tic

%% RUCHE OR NOT
ruche = true;

%% PARAMETERS
%available_RBs = 450;
available_RBs = 225;
% r = [196.3993,100.4880,61.7281,43.2719, 196.3993,100.4880,61.7281,43.2719];
global KK;
global KKK;

KKK = 0;
KK=0;

  CQI_indices = [12, 10, 8, 6];
  [r, eta, sigma] = parameters(CQI_indices);
  RB_usedMatrix = [];

  weights=[0.5,0.5]; % [fidelity, accuracy]

%% FL environment
iteration = 1; % initialization is what imapcts the learning the most short term
averagenumber = 1;
verboseFL = false;
miniBatchSize = 100;
executionEnviroment ='parallel';
AccDevMat = false;
Shuffle='never';
long = false;

%% Computational capabilities
FragSDS = 1;
percentages = [1/8, 1/4, 1/2, 1];

%% evaluation param
% Mapa para rastrear las repeticiones de cada cromosoma
chromosomeRepetitions = containers.Map  ('KeyType', 'char', 'ValueType', 'int32');
historicalChromosomeFitness = containers.Map('KeyType', 'char', 'ValueType', 'any');
historicalChromosomeAccuracy = containers.Map('KeyType', 'char', 'ValueType', 'any');
chromosomeFitnessMap = containers.Map('KeyType', 'char', 'ValueType', 'any');

%% GA
populationSize = 20;
numberOfGenerations = 10;
adjustPIEpoch = 4;
increment=0.01;

%%
numberOfGenes = 20; %%%%%% AQUI %%%%%%%
crossoverProbability = 0.8;
mutationProbability = 0.0625;
tournamentSelectionParameter = 0.5;
numberOfVariables = 4; %%%%%% AQUI %%%%%%%
tournamentSize = 2;
numberOfReplications = 2;
verbose = true;
BitsXVariable = numberOfGenes / numberOfVariables; 
PIprocessed=numberOfGenerations*populationSize;


%% Paths and Adresses 

  if ruche
      if ~long
          refModelName = 'Ref_Model_15_i_15_r_noQ';
          directory_RefModel = 'refModel_15_i_15_withFragSDS';  
      else
          refModelName = 'Ref_Model_70_i_3_r_noQ';     
          directory_RefModel = 'refModel_70_i_3_avg'; 
      end
      baseDir_functionOpt = '/gpfs/workdir/costafrelu/FunctionOptimization_noQ_sameDS/';
  else
      refModelName = 'Ref_Model_1_i_1_r_noQ';
      directory_RefModel = 'refModel_1_i_1';
      baseDir_functionOpt = '..\workdir\FunctionOptimization\noQ_sameDS';
  end

directory_tempDir = sprintf('i_%d_avg_%d', iteration, averagenumber);

directory_FO_Trial_Config = sprintf('gen%d_popu%d_ite%d_avg%d_fid%.2f_acc%.2f_JOINT_simetric', numberOfGenerations, populationSize, iteration, averagenumber, weights(1), weights(2));

fullPath = createVersionedDirectory(baseDir_functionOpt, directory_FO_Trial_Config);

% Confirm the directory creation
fprintf('Directory created at: %s\n', fullPath);

%% INITIALIZATION
%%%%%% AQUI %%%%%%%%
[population, initDecodedPopul, individual_PI_RBs, averageRBsInit] = InitializePopulation(populationSize, numberOfGenes, numberOfVariables, r, available_RBs);
%%%%%%%%%%%%%%%%%%%
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
            %SE PODRÍAN HABER GENERADO CROMOSOMAS ILÍCITOS
    % Registrar chromosomas
    allPopulations{iGeneration} = population;
    allDecodedPopulations{iGeneration} = decodedPopulation;
    
    %% Evaluate Population
    % No evaluar al best chromosome +1 vez
   %% Evaluate Population
    [fidelityFitness, accuracyFitness, RB_used, tiempo] = EvaluatePopulation_joint_simetric(decodedPopulation, r, available_RBs,...
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

    %Compute BestChromosome
    [bestChromosomesList, bestChromosome] = CalculateBestChromosomes(allDecodedPopulations, allFitnessValues, bestChromosomesList, chromosomeFitnessMap, numberOfVariables, numberOfGenes);

    %% Mejorar Chromosomas
    if mod(iGeneration,adjustPIEpoch)==0

        sortedCombinedAllGenerations = CombineAndSortGenerations(allDecodedPopulations, allFitnessValues);
   
        numToSelect = floor(populationSize / 2); %ojo, que si es impar, seleccionará menos de la mitad, has de seleccionar uno pas para improve o GA
        decodedSelectedChromosomes_H = sortedCombinedAllGenerations(1:numToSelect, 1:end-1);

        if mod((populationSize/2),2)~=0
            decodedSelectedChromosomes_GA = sortedCombinedAllGenerations(1:numToSelect+1, 1:end-1);
            selectedFitness = sortedCombinedAllGenerations(1:numToSelect+1, end);
        else
            decodedSelectedChromosomes_GA = sortedCombinedAllGenerations(1:numToSelect, 1:end-1);
            selectedFitness = sortedCombinedAllGenerations(1:numToSelect, end);
        end        

        % Increment Chromosomes and encode them
        decodedImprovedChromosomes = SmartIncrementalAdjustment(decodedSelectedChromosomes_H, r, available_RBs, increment, percentages);
        [improvedChromosomes, ~] = ProcessAndAdjustChromosomes(decodedImprovedChromosomes, numberOfVariables, numberOfGenes, r, available_RBs, percentages);
        %improvedChromosomes = EncodeChromosomes(decodedImprovedChromosomes, numberOfVariables, numberOfGenes);

        % encode non-imporved chromosomes and apply GA operations
        selectedChromosomes = EncodeChromosomes(decodedSelectedChromosomes_GA, numberOfVariables, numberOfGenes);
        selectedChromosomes = ApplyGeneticOperators(selectedChromosomes, selectedFitness, tournamentSelectionParameter,...
            tournamentSize, crossoverProbability, mutationProbability, r, available_RBs, numberOfVariables,...
            bestChromosome, numberOfReplications); %añade un chromosoma de más!
        
        selectedChromosomes = DecodePopulation(selectedChromosomes, numberOfVariables, BitsXVariable);
        [selectedChromosomes, ~] = ProcessAndAdjustChromosomes(selectedChromosomes, numberOfVariables, numberOfGenes, r, available_RBs, percentages);

        newPopulation = vertcat(improvedChromosomes, selectedChromosomes);

    else

%% GENETIC ALGORITHM

        newPopulation = ApplyGeneticOperators(population, fitness, tournamentSelectionParameter,...
            tournamentSize, crossoverProbability, mutationProbability, r, available_RBs, numberOfVariables,...
            bestChromosome, numberOfReplications);
        
        newPopulation = DecodePopulation(newPopulation, numberOfVariables, BitsXVariable);
        [newPopulation, ~] = ProcessAndAdjustChromosomes(newPopulation, numberOfVariables, numberOfGenes, r, available_RBs, percentages);

    end

    population = newPopulation;
    
    fprintf('\n -------- End Generation: %d --------- \n', iGeneration);


    [unifiedMatrix, chromosomeData, genTracking] = processGenerationData(allDecodedPopulations, allFitnessValues, allDeviationValues, allAccuracyValues, fullPath, iGeneration);
    
    save(fullfile(fullPath, 'BestChromosomes.mat'), 'bestChromosomesList');
    save(fullfile(fullPath, 'timePerGen.mat'), 'timePerGen');
    save(fullfile(fullPath, 'RB_usedMatrix.mat'), 'RB_usedMatrix');

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

function saveChromosomeData(fullPath, unifiedMatrix, chromosomeData, genTracking, iGen)
    fileName = fullfile(fullPath, sprintf('ChromosomeData_Gen%d.mat', iGen));
    save(fileName, 'unifiedMatrix', 'chromosomeData', 'genTracking');
    fprintf('Data for Generation %d saved successfully.\n', iGen);
end