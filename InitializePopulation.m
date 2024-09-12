<<<<<<< HEAD
=======
<<<<<<< HEAD
function [population,decodedPopul, individualRBs, averageRBsInit] = InitializePopulation(populationSize, numberOfGenes, numberOfVariables, r, available_RBs)
    numberOfBits = numberOfGenes / numberOfVariables; % Assuming equal distribution of bits per user
=======
>>>>>>> 9c6ccc124 (Reinicializando el repositorio)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%   Alp Sayin - alpsayin[at]alpsayin[dot]com - https://alpsayin.com
%   Matlab Genetic Algorithm
%   Spring 2012
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [population,decodedPopul, individualRBs, averageRBsInit] = InitializePopulation(populationSize, numberOfGenes, numberOfVariables, r, available_RBs)


    numberOfBits = numberOfGenes / numberOfVariables; % Assuming equal distribution of bits per user.

<<<<<<< HEAD
=======
>>>>>>> 021be8e40437ec950ed2e73942ea9f70655ac38e
>>>>>>> 9c6ccc124 (Reinicializando el repositorio)
    % Normalize tempPI based on inverse of r for the whole set of variables
    invR = 1 ./ r;
    baseTempPI = invR / sum(invR);
    baseTempPI = baseTempPI / max(baseTempPI); % Ensure max baseTempPI is scaled to 1
<<<<<<< HEAD
    
=======
<<<<<<< HEAD
=======
    
>>>>>>> 021be8e40437ec950ed2e73942ea9f70655ac38e
>>>>>>> 9c6ccc124 (Reinicializando el repositorio)
    population = zeros(populationSize, numberOfGenes);
    for i = 1:populationSize
        % Initialize tempPI for each individual within constraints
        tempPI = zeros(1, numberOfVariables);
        while true
            for j = 1:numberOfVariables
                % Apply the randomization logic based on baseTempPI
                tempPI(j) = randomizePI(baseTempPI(j),baseTempPI(j)>0.5);
            end
<<<<<<< HEAD
            
=======
<<<<<<< HEAD
            tempPI = IncrementalAdjustment(tempPI, r, available_RBs); %IMPROVE CHROMOSOMES
=======
            
>>>>>>> 021be8e40437ec950ed2e73942ea9f70655ac38e
>>>>>>> 9c6ccc124 (Reinicializando el repositorio)
            % Check if this tempPI meets the resource constraint
            if sum(tempPI .* r) <= available_RBs
                % Encode tempPI into binary representation
                for j = 1:numberOfVariables
                    binaryValue = round(tempPI(j) * (2^numberOfBits - 1));
                    binaryRepresentation = int2bit(binaryValue, numberOfBits, true);
                    binaryRepresentation=binaryRepresentation';
                    
                    startPos = (j-1) * numberOfBits + 1;
                    population(i, startPos:startPos+numberOfBits-1) = binaryRepresentation;
                end
                break; % Valid individual found, exit loop
            end
            % If constraint is not met, loop retries with a new set of random PI values
        end
    end
    % Decode and calculate RB usage
    [decodedPopul, individualRBs] = decodeAndCalculateRB(population, numberOfVariables, r, numberOfBits);
    %AQUI, DECODED AND ENCODED... 
    averageRBsInit = mean(individualRBs); % Calculate average RBs used
end

function piValue = randomizePI(basePI, isAboveHalf)
    if basePI == 1
<<<<<<< HEAD
=======
<<<<<<< HEAD
        % piValue = 1; % Keep max probabilities at 1
        piValue = rand(1) * 0.3 + 0.6;
    elseif isAboveHalf
        piValue = rand(1) * 0.4 + 0.5; % Random value in (0.5, 1]
    else
        piValue = rand(1) * 0.5 + 0.1; % Random value in [0, 0.5]
=======
>>>>>>> 9c6ccc124 (Reinicializando el repositorio)
        piValue = 1; % Keep max probabilities at 1
    elseif isAboveHalf
        piValue = rand(1) * 0.5 + 0.5; % Random value in (0.5, 1]
    else
        piValue = rand(1) * 0.5; % Random value in [0, 0.5]
<<<<<<< HEAD
=======
>>>>>>> 021be8e40437ec950ed2e73942ea9f70655ac38e
>>>>>>> 9c6ccc124 (Reinicializando el repositorio)
    end
end

function [decodedPopul, individualRBs] = decodeAndCalculateRB(population, numberOfVariables, r, numberOfBits)   
    populationSize = size(population, 1);
    decodedPopul = zeros(populationSize, numberOfVariables);
    individualRBs = zeros(populationSize, 1);
    
    for i = 1:populationSize
        decodedPopul(i, :) = DecodeChromosome(population(i, :), numberOfVariables);
        individualRBs(i) = sum(decodedPopul(i, :) .* r); % Ensure compliance with constraint
    end
end 
    