% function PlotAllFitnessValues(Table)
%     % Convertir todos los valores de fitness a positivos y excluir los que tengan valor 15
%     Table.Fitness = abs(Table.Fitness);
%     FilteredTable = Table(Table.Fitness ~= 15, :);
% 
%     % Ordenar la tabla filtrada por fitness en orden descendente
%     SortedFilteredTable = sortrows(FilteredTable, 'Fitness', 'descend');
% 
%     % Graficar todos los valores de fitness ordenados y filtrados
%     PlotOrderedFilteredFitness(SortedFilteredTable);
% 
%     % Función para graficar todos los valores de fitness ordenados y filtrados
%     function PlotOrderedFilteredFitness(PlotData)
%         % Usar un índice secuencial para el eje x
%         sequentialIndex = 1:height(PlotData);
%         fitnessValues = PlotData.Fitness;
% 
%         figure;
%         scatter(sequentialIndex, fitnessValues, 'filled');
%         xlabel('Sequential Chromosome Number (Ordered by Fitness)');
%         ylabel('Fitness Value');
%         title('All Fitness Values Ordered Descending (Excluding -15)');
%         grid on;
%     end
% end

function PlotAllFitnessValues(finalTable)
    % Convertir la columna de generaciones a un array de celdas de números
    generationsArray = cellfun(@(x) str2num(char(x)), finalTable.Generation, 'UniformOutput', false);
    
    % Asegurarse de que todos los valores de fitness sean positivos
    finalTable.Fitness = abs(double(finalTable.Fitness));
    
    % Excluir los valores de fitness de -15
    finalTable = finalTable(finalTable.Fitness ~= 15, :);
    
    % Preparar la figura para graficar
    figure;
    hold on;
    
    % Crear un contenedor para marcar valores repetidos
    isRepeated = false(height(finalTable), 1);
    
    % Identificar valores de fitness repetidos en múltiples generaciones
    for iRow = 1:height(finalTable)
        if length(generationsArray{iRow}) > 1
            isRepeated(iRow) = true;
        end
    end
    
    % Graficar los valores únicos
    uniqueFitness = finalTable.Fitness(~isRepeated);
    uniqueGenerations = cellfun(@(x) x(1), generationsArray(~isRepeated));
    scatter(uniqueGenerations, uniqueFitness, 'bo');
    
    % Graficar los valores repetidos
    repeatedFitness = finalTable.Fitness(isRepeated);
    repeatedGenerationsList = generationsArray(isRepeated);
    for i = 1:length(repeatedFitness)
        scatter(repeatedGenerationsList{i}, repmat(repeatedFitness(i), size(repeatedGenerationsList{i})), 'ro');
    end
    
    hold off;
    xlabel('Generation');
    ylabel('Fitness Value');
    title('Fitness Values by Generation (Excluding -15), Red for Repeats');
    grid on;
    legend({'Unique Fitness', 'Repeated Fitness'}, 'Location', 'best');
end



