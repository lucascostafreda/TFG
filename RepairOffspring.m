function repairedPI = RepairOffspring(decodedPI, r, available_RBs)
    % Sort 'r' and 'decodedPI' in descending order of 'r'
    [sortedR, sortOrder] = sort(r, 'descend');
    sortedPI = decodedPI(sortOrder);
    
    % Calculate initial resource usage
    totalUsage = sum(sortedPI .* sortedR);
    
    % Adjust PI values starting from the worst link conditions
    for i = 1:length(sortedPI)
        if totalUsage <= available_RBs
            break; % Stop if within the resource limit
        end
        
        % Calculate how much the current PI can be reduced
        maxReduction = (totalUsage - available_RBs) / sortedR(i);
        
        % Reduce the PI value as needed, without going below 0
        reduction = min(maxReduction, sortedPI(i));
        sortedPI(i) = sortedPI(i) - reduction;
        
        % Update the total resource usage
        totalUsage = sum(sortedPI .* sortedR);
    end
    
    % Ensure no PI value is below 0 after adjustments
    sortedPI = max(sortedPI, 0);
    
    % Revert the order back to the original
    repairedPI = zeros(1, length(decodedPI));
    repairedPI(sortOrder) = sortedPI;
end
