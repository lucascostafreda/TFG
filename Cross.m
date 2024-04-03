%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%   Alp Sayin - alpsayin[at]alpsayin[dot]com - https://alpsayin.com
%   Matlab Genetic Algorithm
%   Spring 2012
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% function newChromosomePair = Cross( chromosome1, chromosome2)
% 
% nGenes = size(chromosome1,2) ; 
% crossoverPoint = 1 + fix(rand*(nGenes-1));
% assert(crossoverPoint>0 && crossoverPoint<=nGenes);
% newChromosomePair(1, :) = [chromosome1(1:crossoverPoint) chromosome2(crossoverPoint+1:end)];
% newChromosomePair(2, :) = [chromosome2(1:crossoverPoint) chromosome1(crossoverPoint+1:end)];

function newChromosomePair = Cross(chromosome1, chromosome2, r, available_RBs, numberOfVariables)
    nGenes = size(chromosome1, 2); 
    crossoverPoint = 1 + fix(rand*(nGenes-1));
    assert(crossoverPoint > 0 && crossoverPoint <= nGenes);
    
    % Perform crossover
    tempOffspring1 = [chromosome1(1:crossoverPoint) chromosome2(crossoverPoint+1:end)];
    tempOffspring2 = [chromosome2(1:crossoverPoint) chromosome1(crossoverPoint+1:end)];
    
    % Decode, verify and potentially repair the offspring
    newChromosomePair = zeros(2, nGenes); % Preallocate for efficiency
    for i = 1:2
        if i == 1
            tempOffspring = tempOffspring1;
        else
            tempOffspring = tempOffspring2;
        end
        decodedPI = DecodeChromosome(tempOffspring, numberOfVariables);
        
        % Check if the offspring meets the resource constraint
        if sum(decodedPI .* r) > available_RBs
            % Repair the offspring
            repairedPI = RepairOffspring(decodedPI, r, available_RBs);
            % Re-encode the repaired offspring back into binary form
            tempOffspring = EncodeChromosome(repairedPI, numberOfVariables, nGenes);
        end
        newChromosomePair(i, :) = tempOffspring;
    end
end


