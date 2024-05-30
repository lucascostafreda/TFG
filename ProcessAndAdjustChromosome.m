function [encodedChromosome, decodedChromosome] = ProcessAndAdjustChromosome(decodedChromosome, numberOfVariables, numberOfGenes, r, available_RBs)

    if sum(decodedChromosome .* r) > available_RBs
        % Reparar el cromosoma decodificado
        decodedChromosome = RepairDecodedChromosome(decodedChromosome, r, available_RBs);
    end
    
    % Codificar el cromosoma reparado
    encodedChromosome = EncodeChromosome(decodedChromosome, numberOfVariables, numberOfGenes);

    % Verificar si el cromosoma codificado cumple con las restricciones
    if sum(decodedChromosome .* r) > available_RBs
        % Ajustar el cromosoma codificado para cumplir las restricciones
        encodedChromosome = AdjustEncodedChromosome(encodedChromosome, numberOfVariables, numberOfGenes, r, available_RBs);
    end
end

function decodedChromosome = RepairDecodedChromosome(decodedChromosome, r, available_RBs)
      % Calculate initial resource usage
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

function encodedChromosome = AdjustEncodedChromosome(encodedChromosome, numberOfVariables, numberOfGenes, r, available_RBs, percentages, increment)
    decodedChromosome = DecodeChromosome(encodedChromosome, numberOfVariables);
    initialUsage = sum(decodedChromosome .* r);
    maxIterations = 100;  % Limit to prevent infinite loops
    iteration = 0;

    while initialUsage > available_RBs && iteration < maxIterations
        % Calculate the decrement factor based on percentages and inverse of current resource usage
        scaledDecrements = increment * percentages ./ r;

        % Ensure decrements do not take any chromosome value below zero
        possibleDecrements = min(scaledDecrements, decodedChromosome);

        % Calculate the new potential values with adjusted decrements
        newValues = decodedChromosome - possibleDecrements;

        % Calculate the new usage if these decrements were applied
        newUsage = sum(newValues .* r);

        if newUsage <= available_RBs
            decodedChromosome = newValues;
            break;  % Stop if within resource limits
        else
            % Adjust the decrement and try again
            increment = increment * 0.9;  % Reduce increment to finer tune the adjustment
        end

        iteration = iteration+1;
    end

    % Re-encode the chromosome after adjustment
    encodedChromosome = EncodeChromosome(decodedChromosome, numberOfVariables, numberOfGenes);
end

