function repairedPI = RepairOffSpring_Joint(decodedPI, r, available_RBs)
    % Calculate initial resource usage
    totalUsage = sum(decodedPI .* r);
    
    % Calculate the excess usage beyond the available resources
    excessUsage = totalUsage - available_RBs;
    
    if excessUsage > 0
        % Calculate the total weight of resources
        totalWeight = sum(r);
        
        % Calculate the uniform reduction factor
        uniformReduction = excessUsage / totalWeight;
        
        % Apply uniform reduction to each component, ensuring no values fall below zero
        decodedPI = max(decodedPI - uniformReduction, 0);
        
        % Recalculate total usage to ensure it is within the limit due to rounding or precision issues
        totalUsage = sum(decodedPI .* r);
        while totalUsage > available_RBs
            % Reduce slightly to correct any overages
            additionalReduction = (totalUsage - available_RBs) / totalWeight;
            decodedPI = max(decodedPI - additionalReduction, 0);
            totalUsage = sum(decodedPI .* r);
        end
    end
    
    % The repaired PI is the adjusted decodedPI array
    repairedPI = decodedPI;
end
