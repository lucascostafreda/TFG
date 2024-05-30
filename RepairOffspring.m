% function repairedPI = RepairOffspring(decodedPI, r, available_RBs)
%     % Sort 'r' and 'decodedPI' in descending order of 'r'
%     [sortedR, sortOrder] = sort(r, 'descend');
%     sortedPI = decodedPI(sortOrder);
% 
%     % Calculate initial resource usage
%     totalUsage = sum(sortedPI .* sortedR);
% 
%     % Adjust PI values starting from the worst link conditions
%     for i = 1:length(sortedPI)
%         if totalUsage <= available_RBs
%             break; % Stop if within the resource limit
%         end
% 
%         % Calculate how much the current PI can be reduced
%         maxReduction = (totalUsage - available_RBs) / sortedR(i);
% 
% 
%         reduction = min(maxReduction, sortedPI(i)); % so it's not negative
%         sortedPI(i) = sortedPI(i) - reduction;
% 
%         % Update the total resource usage
%         totalUsage = sum(sortedPI .* sortedR);
%     end
% 
%     % Ensure no PI value is below 0 after adjustments
%     sortedPI = max(sortedPI, 0);
% 
%     % Revert the order back to the original
%     repairedPI = zeros(1, length(decodedPI));
%     repairedPI(sortOrder) = sortedPI;
% end

function repairedPI = RepairOffspring(decodedPI, r, available_RBs)
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
