function sortedCombinedAllGenerations = CombineAndSortGenerations(allDecodedPopulations, allFitnessValues)
    % Initialize an empty matrix to combine all chromosomes and their fitness
    combinedAllGenerations = [];

    % Iterate through all accumulated generations to combine them
    for iGen = 1:numel(allDecodedPopulations)
        currentPopulation = allDecodedPopulations{iGen};
        currentFitness = allFitnessValues{iGen};

        % Ensure currentFitness is a column if it's not already
        if isrow(currentFitness)
            currentFitness = currentFitness';
        end

        % Combine chromosomes and their fitness into a single matrix for this generation
        combined = [currentPopulation, currentFitness];

        % Add to the large matrix combining all generations
        combinedAllGenerations = [combinedAllGenerations; combined];
    end

    % Sort all combined chromosomes by fitness in descending order
    [~, sortedIndices] = sort(combinedAllGenerations(:, end), 'descend');
    sortedCombinedAllGenerations = combinedAllGenerations(sortedIndices, :);
end
