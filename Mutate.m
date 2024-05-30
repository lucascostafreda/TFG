
function tempPopulation = Mutate(tempPopulation, mutationProbability, r, available_RBs, numberOfVariables)
    % Obtiene el tamaño de la población y el número de genes
    [populationSize, nGenes] = size(tempPopulation);
    
    % Itera sobre cada cromosoma en la población
    for i = 1:populationSize
        % Crea una copia del cromosoma actual antes de mutar
        originalChromosome = tempPopulation(i,:);
        
        % Determina índices de mutación basados en la probabilidad
        mutationIndices = rand(1, nGenes) < mutationProbability;
        
        % Realiza la mutación (inversión de bit) para los índices seleccionados
        tempPopulation(i, mutationIndices) = 1 - tempPopulation(i, mutationIndices);
        
        % % Decodifica el cromosoma después de mutar
        % decodedPI = DecodeChromosome(tempPopulation(i,:), numberOfVariables);
        % 
        % % Verifica si el cromosoma cumple con la restricción de recursos
        % if sum(decodedPI .* r) > available_RBs
        %     % Si no cumple, repite el proceso de reparación y re-codificación
        %     repairedPI = RepairOffspring(decodedPI, r, available_RBs);
        %     tempPopulation(i,:) = EncodeChromosome(repairedPI, numberOfVariables, nGenes);
        % end
    end
end
