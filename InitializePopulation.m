%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%   Alp Sayin - alpsayin[at]alpsayin[dot]com - https://alpsayin.com
%   Matlab Genetic Algorithm
%   Spring 2012
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [population,decodedPopul, individualRBs, averageRBsInit] = InitializePopulation(populationSize, numberOfGenes, numberOfVariables, r, available_RBs)

% b1=[1,1];
% b2=[1,1];
% 
% population = (rand(populationSize, numberOfGenes)<0.5).*1;
% 
% return

%  population = zeros(populationSize, numberOfGenes); % Pre-allocate for speed
%  decodedPopul = zeros(populationSize, numberOfVariables);
%  individualRBs = zeros(populationSize, 1);
%  individualCount = 0;
% 
%     while individualCount < populationSize
%         tempIndividual = (rand(1, numberOfGenes) < 0.5) .* 1;
%         updatedPI = DecodeChromosome(tempIndividual, numberOfVariables);
%         % Asumiendo que r es un vector con el mismo número de elementos que numberOfVariables
%         % updated_PI = decodedIndividual(index,:);
%         %PI=[1,1,updatedPI,1,1,updatedPI];
%         PI = [updatedPI, updatedPI];
%         temp = sum(r .* PI);
%         if temp <= available_RBs && temp>available_RBs-20
%             individualCount = individualCount + 1;
%             population(individualCount, :) = tempIndividual;
%             decodedPopul(individualCount, :) = updatedPI;
%             individualRBs(individualCount)=sum(r .* PI);
%             averageRBsInit = mean(individualRBs);
%         end
%     end
% end

%%
    numberOfBits = numberOfGenes / numberOfVariables; % Assuming equal distribution of bits per user.

    % Normalize tempPI based on inverse of r for the whole set of variables
    invR = 1 ./ r;
    baseTempPI = invR / sum(invR);
    baseTempPI = baseTempPI / max(baseTempPI); % Ensure max baseTempPI is scaled to 1

    %Iteratively increase sorted PI to maximize resource use without exceeding constraints
    % for i = 1:length(sortedTempPI)
    %     while true
    %         incrementedPI = sortedTempPI;
    %         incrementedPI(i) = incrementedPI(i) + 0.05; % Try incrementing the current user's probability
    % 
    %         if sum(incrementedPI .* r(originalIndices)) <= available_RBs
    %             sortedTempPI(i) = incrementedPI(i); % Accept the increment
    %         else
    %             break; % Stop if the next increment would exceed available RBs
    %         end
    % 
    %         if sortedTempPI(i) >= 1
    %             break; % Also break if probability reaches 1
    %         end
    %     end
    % end
    % sortedTempPI = min(sortedTempPI, 1); % Ensure no probability exceeds 1
    % 
    % % Reposition sorted PI values back to their original positions
    % tempPI(originalIndices) = sortedTempPI;
    % 

    population = zeros(populationSize, numberOfGenes);
    for i = 1:populationSize
        % Initialize tempPI for each individual within constraints
        tempPI = zeros(1, numberOfVariables);
        while true
            for j = 1:numberOfVariables
                % Apply the randomization logic based on baseTempPI
                tempPI(j) = randomizePI(baseTempPI(j),baseTempPI(j)>0.5);
            end
            
            % Check if this tempPI meets the resource constraint
            if sum(tempPI .* r) <= available_RBs
                % Encode tempPI into binary representation
                for j = 1:numberOfVariables
                    binaryValue = round(tempPI(j) * (2^numberOfBits - 1));
                    binaryRepresentation = de2bi(binaryValue, numberOfBits, 'left-msb');
                    
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
    averageRBsInit = mean(individualRBs); % Calculate average RBs used
end

function piValue = randomizePI(basePI, isAboveHalf)
    if basePI == 1
        piValue = 1; % Keep max probabilities at 1
    elseif isAboveHalf
        piValue = rand(1) * 0.5 + 0.5; % Random value in (0.5, 1]
    else
        piValue = rand(1) * 0.5; % Random value in [0, 0.5]
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
