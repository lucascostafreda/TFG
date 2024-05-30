function PlotAllFitnessValues_Joint(finalTable)
    % Ensure all fitness values are positive (already handled, assuming no negative values should be present)
    finalTable.Fitness = abs(finalTable.Fitness);

    % Exclude fitness values of 0
    finalTable = finalTable(finalTable.Fitness ~= 0, :);

    % Convert the generation column to a cell array of numbers
    generationsArray = cellfun(@(x) str2num(char(x)), finalTable.Generation, 'UniformOutput', false);

    % Prepare the figure for plotting
    figure;
    hold on;

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

    % Plot repeated values
    repeatedFitness = finalTable.Fitness(isRepeated);
    repeatedGenerationsList = generationsArray(isRepeated);
    for i = 1:length(repeatedFitness)
        scatter(repeatedGenerationsList{i}, repmat(repeatedFitness(i), size(repeatedGenerationsList{i})), 'ro');
    end

    hold off;
    xlabel('Generation');
    ylabel('Fitness Value');
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
