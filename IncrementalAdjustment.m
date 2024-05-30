function improvedChromosomes = IncrementalAdjustment(selectedChromosomes, r, available_RBs)
    % Initialize improved chromosomes array
    improvedChromosomes = selectedChromosomes;

    % Define the thresholds for r value tiers
    lowThreshold = 50; % Below this is considered low
    mediumThreshold = 150; % Below this is considered medium, above is high
    
    % Define improvement steps for each tier
    lowTierStep = 0.15; % For r around 43.271
    mediumTierStep = 0.1; % For r around 61.728 and 100.488
    highTierStep = 0.035; % For r around 196.399
    lowTierMaxSteps = 2; 
    mediumTierMaxSteps = 3; 
    highTierMaxSteps = 5; 

    % Iterate over selected chromosomes
    for chromoIdx = 1:size(selectedChromosomes, 1)
        % Sort chromosome components based on radio link conditions (r)
        [sortedR, sortOrder] = sort(r, 'ascend');
        sortedChromo = selectedChromosomes(chromoIdx, sortOrder);
        
        currentUsage = sum(sortedChromo .* sortedR);
        
        % Adjust chromosome values incrementally
        for compIdx = 1:size(selectedChromosomes, 2)
            % Check if total usage is within the resource limit
            if currentUsage >= available_RBs 
                break; % Stop if within the resource limit
            end
            
            % Determine the dynamic improvement step based on the r value
            [dynamicStep,maxStepsPerComponent] = determineImprovementStep(sortedR(compIdx));
            
            % Incrementally adjust chromosome component value
            for step = 1:maxStepsPerComponent
                % Check if adding the dynamic improvement step exceeds resource limit
                if currentUsage + sortedR(compIdx) * dynamicStep <= available_RBs
                    % Apply dynamic improvement step
                    improvedChromosomes(chromoIdx, sortOrder(compIdx)) = ...
                        improvedChromosomes(chromoIdx, sortOrder(compIdx)) + dynamicStep;
                    currentUsage = sum(improvedChromosomes(chromoIdx, :) .* r); % Update total usage
                   
                    % Ensure component value does not exceed one
                    if improvedChromosomes(chromoIdx, sortOrder(compIdx))   > 1
                        improvedChromosomes(chromoIdx, sortOrder(compIdx)) = 1;
                        currentUsage = sum(improvedChromosomes(chromoIdx, :) .* r);
                        break;
                    end
                else
                    % Exit loop if resource limit exceeded
                    break;
                end
            end
        end
    end

% Function to determine the improvement step based on r value
    function [stepSize, numSteps] = determineImprovementStep(rValue)
    if rValue < lowThreshold
        stepSize = lowTierStep;
        numSteps = lowTierMaxSteps;
    elseif rValue < mediumThreshold
        stepSize = mediumTierStep;
        numSteps = mediumTierMaxSteps;
    else
        stepSize = highTierStep;
        numSteps = highTierMaxSteps;
    end
end

end