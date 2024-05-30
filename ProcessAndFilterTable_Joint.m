function [FilteredTable,Top15Chromosomes, Top15Chromosomes_noSinglePI, MostRepeatedChromosomes] = ProcessAndFilterTable_Joint(Table)
    % Convertir la columna de generaciones a numérica, extrayendo solo el primer valor
    firstGenValues = cellfun(@(x) str2double(regexp(x, '\d+', 'match', 'once')), Table.Generation);
    Table.FirstGen = firstGenValues;
    
    % Asegurarse de que los valores de fitness ya son positivos y en el rango [0, 1]
    
    % Inicializar la tabla filtrada y el mejor valor de fitness registrado (ahora buscamos el máximo porque 1 es el mejor)
    FilteredData = [];
    bestFitnessSoFar = 0; % Inicialización para buscar el valor máximo
    
    % Obtener las generaciones únicas y ordenarlas
    uniqueGenerations = unique(Table.FirstGen);
    
    for iGen = uniqueGenerations'
        % Seleccionar filas de la generación actual y ordenarlas por fitness de menor a mayor
        currentGenRows = Table(Table.FirstGen == iGen, :);
        sortedCurrentGen = sortrows(currentGenRows, 'Fitness', 'ascend');
        
        % Filtrar los cromosomas que tienen un fitness mejor que el mejor fitness registrado hasta el momento
        for iRow = 1:height(sortedCurrentGen)
            if sortedCurrentGen.Fitness(iRow) > bestFitnessSoFar
                bestFitnessSoFar = sortedCurrentGen.Fitness(iRow);
                FilteredData = [FilteredData; sortedCurrentGen(iRow, :)];
            end
        end
    end
    
    % La tabla filtrada ya tiene los fitness en valores positivos y ordenados como se desea
    FilteredTable = FilteredData;
    
    % Graficar la evolución de los fitnesses seleccionados
    PlotFitnessEvolution(FilteredTable);
    
    Top15Chromosomes = ExtractTop15Chromosomes(Table); % Now passing original Table
    Top15Chromosomes_noSinglePI = ExtractTop15Chromosomes_noSinglePI(Table);
    MostRepeatedChromosomes = ExtractMostRepeatedChromosomes(Table);

end


% Función para graficar la evolución de los fitnesses
function PlotFitnessEvolution(PlotData)
    % Usar un índice secuencial para el eje x
    sequentialIndex = 1:height(PlotData);
    fitnessValues = PlotData.Fitness;
    
    figure;
    scatter(sequentialIndex, fitnessValues, 'filled');
    xlabel('Sequential Chromosome Number');
    ylabel('Fitness Value');
    title('Evolution of Selected Fitness Values Over Time (Ascending)');
    grid on;
    % MATLAB ajustará automáticamente los límites del eje y para acomodar los datos
end

function Top15Chromosomes = ExtractTop15Chromosomes(Table)
    % Remove rows where Fitness is NaN
    Table = Table(~isnan(Table.Fitness), :);
    
    % Convert chromosome data to strings for easy comparison
    % Assuming chromosomes are stored in the first N columns
    N = 8;  % Adjust N according to how many genes each chromosome has
    
    % Convert each column to strings and concatenate for uniqueness
    chromosomeStrings = strings(size(Table, 1), 1);
    for i = 1:N
        chromosomeStrings = chromosomeStrings + "|" + string(Table{:, i});
    end
    
    % Find indices of unique chromosomes
    [~, ia, ~] = unique(chromosomeStrings, 'stable');
    uniqueTable = Table(ia, :);  % Use indices of unique chromosomes to create a table of unique chromosomes
    
    % Sort the table of unique chromosomes by fitness in descending order
    sortedUniqueTable = sortrows(uniqueTable, 'Fitness', 'descend');
    
    % Select the top 15 unique chromosomes
    if height(sortedUniqueTable) > 15
        Top15Chromosomes = sortedUniqueTable(1:15, :);
    else
        Top15Chromosomes = sortedUniqueTable;  % If there are fewer than 15, take all
    end
end

function Top15Chromosomes_noSinglePI = ExtractTop15Chromosomes_noSinglePI(Table)
    % Remove rows where Fitness is NaN
    Table = Table(~isnan(Table.Fitness), :);
    
    % Convert chromosome data to strings for easy comparison
    % Assuming chromosomes are stored in the first N columns
    N = 8;  % Adjust N according to how many genes each chromosome has
    
    % Convert each column to strings and concatenate for uniqueness
    chromosomeStrings = strings(size(Table, 1), 1);
    for i = 1:N
        chromosomeStrings = chromosomeStrings + "|" + string(Table{:, i});
    end
    
    % Find indices and counts of unique chromosomes
    [uniqueChromosomes, ia, ic] = unique(chromosomeStrings, 'stable');
    counts = accumarray(ic, 1);  % Count occurrences of each chromosome
    
    % Filter to keep only chromosomes that appear more than once
    repeatedChromosomesIdx = ia(counts > 1);
    if isempty(repeatedChromosomesIdx)
        Top15Chromosomes_noSinglePI = [];  % Return empty if no chromosomes are repeated
        return;
    end
    
    % Create a table of repeated unique chromosomes
    repeatedTable = Table(repeatedChromosomesIdx, :);  
    
    % Sort the table of repeated unique chromosomes by fitness in descending order
    sortedRepeatedTable = sortrows(repeatedTable, 'Fitness', 'descend');
    
    % Select the top 15 repeated unique chromosomes
    if height(sortedRepeatedTable) > 15
        Top15Chromosomes_noSinglePI = sortedRepeatedTable(1:15, :);
    else
        Top15Chromosomes_noSinglePI = sortedRepeatedTable;  % If there are fewer than 15, take all
    end
end
function MostRepeatedChromosomes = ExtractMostRepeatedChromosomes(Table)
    
    % Remove rows where Fitness is NaN or 0
    fitnessColumn = 'Fitness'; % Name of the fitness column
    Table = Table(~isnan(Table{:, fitnessColumn}) & Table{:, fitnessColumn} ~= 0, :);

    % Convert chromosome data to strings for easy comparison
    % Assuming chromosomes are stored in the first N columns
    N = 8;  % Adjust N according to how many genes each chromosome has
    
    % Convert each column to strings and concatenate for uniqueness
    chromosomeStrings = strings(size(Table, 1), 1);
    for i = 1:N
        chromosomeStrings = chromosomeStrings + "|" + string(Table{:, i});
    end
    
    % Find indices and unique chromosomes
    [uniqueChromosomes, ia, ic] = unique(chromosomeStrings, 'stable');
    counts = accumarray(ic, 1);  % Count occurrences of each chromosome
    
    % Ensure that the 'Generation' column is numeric
    generationData = Table{:, 'Generation'};
    if ~isnumeric(generationData)
        generationData = str2double(generationData);
    end
    
    % Calculate the last generation for each unique chromosome
    lastGeneration = accumarray(ic, generationData, [], @max); % Use @max to find the last generation

    % Create a table with unique chromosomes, their counts, and last generation found
    uniqueTable = Table(ia, :);
    uniqueTable.RepeatCount = counts;  % Add a new column with repeat counts
    uniqueTable.LastGeneration = lastGeneration;  % Add a new column with the last generation found
    
    % Sort the table by repeat count in descending order
    sortedByRepeatCount = sortrows(uniqueTable, 'RepeatCount', 'descend');
    
    % Select the top 15 most repeated chromosomes
    if height(sortedByRepeatCount) > 15
        MostRepeatedChromosomes = sortedByRepeatCount(1:15, :);
    else
        MostRepeatedChromosomes = sortedByRepeatCount;  % If there are fewer than 15, take all
    end
end




