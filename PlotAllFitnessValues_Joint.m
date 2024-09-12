function PlotAllFitnessValues_Joint(finalTable)
<<<<<<< HEAD
    % Ensure all fitness values are positive (already handled, assuming no negative values should be present)
=======
<<<<<<< HEAD
% Ensure all fitness values are positive (already handled, assuming no negative values should be present)
=======
    % Ensure all fitness values are positive (already handled, assuming no negative values should be present)
>>>>>>> 021be8e40437ec950ed2e73942ea9f70655ac38e
>>>>>>> 9c6ccc124 (Reinicializando el repositorio)
    finalTable.Fitness = abs(finalTable.Fitness);

    % Exclude fitness values of 0
    finalTable = finalTable(finalTable.Fitness ~= 0, :);
<<<<<<< HEAD

    % Convert the generation column to a cell array of numbers
    generationsArray = cellfun(@(x) str2num(char(x)), finalTable.Generation, 'UniformOutput', false);
=======
<<<<<<< HEAD
    finalTable = finalTable(finalTable.Fitness >= 0.3, :);

    % Extract the first four columns to identify repeated chromosomes
    firstFourCols = finalTable{:, 1:4};
    
    % Identify unique rows and their indices
    [~, uniqueIdx, idx] = unique(firstFourCols, 'rows', 'first');

    % Initialize an array to keep track of repeated rows
    isRepeated = true(height(finalTable), 1);
    isRepeated(uniqueIdx) = false;

    % Convert the generation column to a cell array of numbers
    if isnumeric(finalTable.Generation)
        generationsArray = num2cell(finalTable.Generation);
    else
        % Convert the generation column to a cell array of numbers
        generationsArray = cellfun(@(x) str2num(char(x)), finalTable.Generation, 'UniformOutput', false);
    end
=======

    % Convert the generation column to a cell array of numbers
    generationsArray = cellfun(@(x) str2num(char(x)), finalTable.Generation, 'UniformOutput', false);
>>>>>>> 021be8e40437ec950ed2e73942ea9f70655ac38e
>>>>>>> 9c6ccc124 (Reinicializando el repositorio)

    % Prepare the figure for plotting
    figure;
    hold on;

<<<<<<< HEAD
=======
<<<<<<< HEAD
    % Plot unique values
    uniqueFitness = finalTable.Fitness(~isRepeated);
    uniqueGenerations = cellfun(@(x) x(1), generationsArray(~isRepeated));
    h1 = scatter(uniqueGenerations, uniqueFitness, 'bo', 'DisplayName', 'Unique Fitness');
=======
>>>>>>> 9c6ccc124 (Reinicializando el repositorio)
    % Container to mark repeated values
    isRepeated = false(height(finalTable), 1);

    % Identify repeated fitness values across multiple generations
    for iRow = 1:height(finalTable)
        if length(generationsArray{iRow}) > 1
            isRepeated(iRow) = true;
        end
    end

    % Plot unique values
    uniqueFitness = finalTable.Fitness(~isRepeated);
    uniqueGenerations = cellfun(@(x) x(1), generationsArray(~isRepeated));
    scatter(uniqueGenerations, uniqueFitness, 'bo');
<<<<<<< HEAD
=======
>>>>>>> 021be8e40437ec950ed2e73942ea9f70655ac38e
>>>>>>> 9c6ccc124 (Reinicializando el repositorio)

    % Plot repeated values
    repeatedFitness = finalTable.Fitness(isRepeated);
    repeatedGenerationsList = generationsArray(isRepeated);
<<<<<<< HEAD
    for i = 1:length(repeatedFitness)
        scatter(repeatedGenerationsList{i}, repmat(repeatedFitness(i), size(repeatedGenerationsList{i})), 'ro');
=======
<<<<<<< HEAD
    h2 = gobjects(sum(isRepeated), 1); % Preallocate graphic objects for legend handling
    for i = 1:sum(isRepeated)
        h2(i) = scatter(repeatedGenerationsList{i}, repmat(repeatedFitness(i), size(repeatedGenerationsList{i})), 'ro', 'DisplayName', 'Repeated Fitness');
=======
    for i = 1:length(repeatedFitness)
        scatter(repeatedGenerationsList{i}, repmat(repeatedFitness(i), size(repeatedGenerationsList{i})), 'ro');
>>>>>>> 021be8e40437ec950ed2e73942ea9f70655ac38e
>>>>>>> 9c6ccc124 (Reinicializando el repositorio)
    end

    hold off;
    xlabel('Generation');
    ylabel('Fitness Value');
<<<<<<< HEAD
=======
<<<<<<< HEAD
    title('Fitness Values by Generation (Excluding values under 0.3)');
    grid on;

    % Update legend to show appropriate entries
    if sum(isRepeated) == 0
        legend(h1, 'Unique Fitness', 'Location', 'best');
    else
        legend([h1, h2(1)], {'Unique Fitness', 'Repeated Fitness'}, 'Location', 'best');
    end
 end
=======
>>>>>>> 9c6ccc124 (Reinicializando el repositorio)
    title('Fitness Values by Generation (Excluding 0), Red for Repeats');
    grid on;
    legend({'Unique Fitness', 'Repeated Fitness'}, 'Location', 'best');
end

% function PlotAllFitnessValues_Joint(Table)
%     % Filter out fitness values of 0
%     FilteredTable = Table(Table.Fitness ~= 0, :);
% 
%     % Order the filtered table by fitness in descending order (best to worst)
%     SortedFilteredTable = sortrows(FilteredTable, 'Fitness', 'descend');
% 
%     % Plot all ordered and filtered fitness values
%     PlotOrderedFilteredFitness(SortedFilteredTable);
% 
%     % Function to plot all ordered and filtered fitness values
%     function PlotOrderedFilteredFitness(PlotData)
%         % Use a sequential index for the x-axis
%         sequentialIndex = 1:height(PlotData);
%         fitnessValues = PlotData.Fitness;
% 
%         figure;
%         scatter(sequentialIndex, fitnessValues, 'filled');
%         xlabel('Sequential Chromosome Number (Ordered by Fitness)');
%         ylabel('Fitness Value');
%         title('All Fitness Values Ordered Descending (Excluding 0)');
%         grid on;
%     end
% end
<<<<<<< HEAD
=======
>>>>>>> 021be8e40437ec950ed2e73942ea9f70655ac38e
>>>>>>> 9c6ccc124 (Reinicializando el repositorio)
