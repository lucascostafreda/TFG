function improvedChromosomes = SmartIncrementalAdjustment(selectedChromosomes, r, available_RBs, increment, percentages)
    % Initialize improved chromosomes array
    improvedChromosomes = selectedChromosomes;

    % Calculate the current usage of resources
    currentUsage = sum(improvedChromosomes .* r, 2); % row-wise sum for each chromosome

    % Iterate over each chromosome
    for chromoIdx = 1:size(selectedChromosomes, 1)
        while true
            % Calculate weight for incrementing based on percentages and inverse of current resource usage
            scaledIncrements = increment * percentages ./ r;

            % Adjust increments to ensure they do not exceed 1 when added to current values
            possibleIncrements = min(scaledIncrements, 1 - improvedChromosomes(chromoIdx, :));

            % Calculate the potential new values with adjusted increments
            newValues = improvedChromosomes(chromoIdx, :) + possibleIncrements;
            
            % Calculate the new usage if these values were applied
            newUsage = sum(newValues .* r);
            
            % Check if the new usage exceeds the available resources
            if newUsage > available_RBs
                break; % Stop if it would exceed the resource limit
            else
                % Update chromosome values and current usage
                improvedChromosomes(chromoIdx, :) = newValues;
                currentUsage(chromoIdx) = newUsage;
            end
        end
    end
end
