function [encodedChromosomes, decodedChromosomes] = ProcessAndAdjustChromosomes(chromosomesMatrix, numberOfVariables, numberOfGenes, r, available_RBs, percentages)
    numChromosomes = size(chromosomesMatrix, 1);
    encodedChromosomes = zeros(numChromosomes, numberOfGenes); % Preallocate matrix for encoded chromosomes
    decodedChromosomes = zeros(numChromosomes, length(r)); % Preallocate matrix for decoded chromosomes
    
    for idx = 1:numChromosomes
        
        decodedChromosome = chromosomesMatrix(idx, :);

        if sum(decodedChromosome .* r) > available_RBs
            % Repair the decoded chromosome
            decodedChromosome = RepairDecodedChromosome(decodedChromosome, r, available_RBs);
        end

        % Encode the repaired chromosome
        encodedChromosome = EncodeChromosome(decodedChromosome, numberOfVariables, numberOfGenes);
        
        % Verify if the repaired and encoded chromosome meets the constraints
        decodedChromosome = DecodeChromosome(encodedChromosome, numberOfVariables);
        
        if sum(decodedChromosome .* r) > available_RBs %OJO, está mal
            % Adjust the chromosome to meet constraints
            [encodedChromosome, decodedChromosome] = AdjustEncodedChromosome(decodedChromosome, numberOfVariables, numberOfGenes, r, available_RBs,  percentages);
        end
        
        % Store results
        encodedChromosomes(idx, :) = encodedChromosome;
        decodedChromosomes(idx, :) = decodedChromosome;
    end
end


function decodedChromosome = RepairDecodedChromosome(decodedChromosome, r, available_RBs)
      % Calculate initial resource usage
<<<<<<< HEAD
=======
<<<<<<< HEAD
    totalUsage = sum(decodedChromosome .* r); 
    % Calculate the excess usage beyond the available resources
    excessUsage = totalUsage - available_RBs;
    if excessUsage > 0
        % Calculate the total weight of resources
        totalWeight = sum(r);
        % Calculate the uniform reduction factor
        uniformReduction = excessUsage / totalWeight;
        % Apply uniform reduction to each component, ensuring no values fall below zero
        decodedChromosome = max(decodedChromosome - uniformReduction, 0);
=======
>>>>>>> 9c6ccc124 (Reinicializando el repositorio)
    totalUsage = sum(decodedChromosome .* r);
    
    % Calculate the excess usage beyond the available resources
    excessUsage = totalUsage - available_RBs;
    
    if excessUsage > 0
        % Calculate the total weight of resources
        totalWeight = sum(r);
        
        % Calculate the uniform reduction factor
        uniformReduction = excessUsage / totalWeight;
        
        % Apply uniform reduction to each component, ensuring no values fall below zero
        decodedChromosome = max(decodedChromosome - uniformReduction, 0);
        
<<<<<<< HEAD
=======
>>>>>>> 021be8e40437ec950ed2e73942ea9f70655ac38e
>>>>>>> 9c6ccc124 (Reinicializando el repositorio)
        % Recalculate total usage to ensure it is within the limit due to rounding or precision issues
        totalUsage = sum(decodedChromosome .* r);
        while totalUsage > available_RBs
            % Reduce slightly to correct any overages
            additionalReduction = (totalUsage - available_RBs) / totalWeight;
            decodedChromosome = max(decodedChromosome - additionalReduction, 0);
            totalUsage = sum(decodedChromosome .* r);
        end
    end
end

function [encodedChromosome, decodedChromosome] = AdjustEncodedChromosome(decodedChromosome, numberOfVariables, numberOfGenes, r, available_RBs, percentages)
    % Normalización de los valores de 'r' y 'percentages'
    normalizedR = r / sum(r);
    normalizedPercentages = percentages / sum(percentages);

    % Double check the constraint after encoding
    while sum(decodedChromosome .* r) > available_RBs
        % Calculate relevance scores
        effectivenessScores = normalizedPercentages ./ normalizedR;
        [~, order] = sort(effectivenessScores, 'ascend');  % Order to adjust by least cost effectiveness

        % Reduce the quantification level one step at a time, following the order of importance
        for idx = order
            currentStepSize = 1 / (2^5 - 1);  % smallest quantification unit
            if decodedChromosome(idx) > currentStepSize
                decodedChromosome(idx) = decodedChromosome(idx) - currentStepSize;
                encodedChromosome = EncodeChromosome(decodedChromosome, numberOfVariables, numberOfGenes);
                if sum(decodedChromosome .* r) <= available_RBs
                    break;  % If within limits, stop adjusting
                end
            end
        end
    end
end


