function newPopulation = ApplyGeneticOperators(population, fitness, tournamentSelectionParameter, tournamentSize, crossoverProbability, mutationProbability, r, available_RBs, numberOfVariables, bestChromosome, numberOfReplications)
    populationSize = size(population, 1);
    newPopulation = population;
    
    for i = 1:2:populationSize
        % Tournament Selection
        i1 = TournamentSelect(fitness, tournamentSelectionParameter, tournamentSize);
        i2 = TournamentSelect(fitness, tournamentSelectionParameter, tournamentSize); % Ensure different selection
        chromosome1 = population(i1,:);
        chromosome2 = population(i2,:);
        
        % Crossover
        if rand < crossoverProbability
            newChromosomePair = Cross(chromosome1, chromosome2, r, available_RBs, numberOfVariables);
            newPopulation(i,:) = newChromosomePair(1,:);
            newPopulation(i+1,:) = newChromosomePair(2,:);
        else
            newPopulation(i,:) = chromosome1;
            newPopulation(i+1,:) = chromosome2;
        end
    end
    
    % Mutation
    newPopulation = Mutate(newPopulation, mutationProbability, r, available_RBs, numberOfVariables);
    %AQUÍ ES MUY PROBABLE QUE SE GENEREN Chromosomas Invalidos por la QUANTIFICACION
    
    % Insert Best Individual
    newPopulation = InsertBestIndividual(newPopulation, bestChromosome, numberOfReplications);

    return
end
