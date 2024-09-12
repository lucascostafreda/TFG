function [bestChromosomesList, bestChromosome] = CalculateBestChromosomes(allDecodedPopulations, allFitnessValues, bestChromosomesList, chromosomeFitnessMap, numberOfVariables, numberOfGenes)
    % Flatten the list of all decoded populations and their fitness values
    allDecodedPopulationsCombined = vertcat(allDecodedPopulations{:});
    allFitnessCombined = vertcat(allFitnessValues{:});

    % Collect fitness for each chromosome
    for idx = 1:length(allFitnessCombined)
        chromosomeVector = allDecodedPopulationsCombined(idx, :);
        chromosomeString = num2str(chromosomeVector, '%0.8f,');  % Use a high precision floating-point format
<<<<<<< HEAD

=======
<<<<<<< HEAD
        chromosomeString = strip(chromosomeString, 'right', ',');
        
=======

>>>>>>> 021be8e40437ec950ed2e73942ea9f70655ac38e
>>>>>>> 9c6ccc124 (Reinicializando el repositorio)
        if isKey(chromosomeFitnessMap, chromosomeString)
            % Add new fitness to the existing vector
            chromosomeFitnessMap(chromosomeString) = [chromosomeFitnessMap(chromosomeString), allFitnessCombined(idx)];
        else
            % Start a new fitness vector for new chromosomes
            chromosomeFitnessMap(chromosomeString) = [allFitnessCombined(idx)];
        end
    end

    % Calculate the best chromosome with the maximum representative fitness
    maxRepresentativeFitness = -inf;
    bestChromosomeString = '';
    keys = chromosomeFitnessMap.keys;
    for k = 1:length(keys)
        fitnessList = chromosomeFitnessMap(keys{k});
        representativeFitness = mean(fitnessList);

        if representativeFitness > maxRepresentativeFitness
            maxRepresentativeFitness = representativeFitness;
            bestChromosomeString = keys{k};
        end
    end

<<<<<<< HEAD
=======
<<<<<<< HEAD
    bestChromosome = str2num(bestChromosomeString);
    % Check the length of bestChromosome
    if length(bestChromosome) ~= numberOfVariables
        if ~isempty(bestChromosomesList) && size(bestChromosomesList, 1) > 0
            % Revert to the last valid best chromosome
            bestChromosome = bestChromosomesList{end, 1};
            disp('Mismatch in the number of variables. Reverting to the previous valid best chromosome.');
        else
            % Handle first iteration or no valid best chromosomes found
            bestChromosome = zeros(1, numberOfVariables);  % Consider other default logic or initialization
            disp('No valid previous chromosome to revert to. Using default initialization.');
        end
    end

=======
>>>>>>> 9c6ccc124 (Reinicializando el repositorio)
    % Convert the best chromosome string back to numeric vector
    bestChromosome = str2num(bestChromosomeString);  % Convert string back to numeric array

    % Check the length of bestChromosome
    if length(bestChromosome) ~= numberOfVariables
        error('The length of bestChromosome does not match the expected number of variables.');
    end

    
<<<<<<< HEAD
=======
>>>>>>> 021be8e40437ec950ed2e73942ea9f70655ac38e
>>>>>>> 9c6ccc124 (Reinicializando el repositorio)
    % Save the best chromosome and its representative fitness
    if exist('bestChromosomesList', 'var')
        bestChromosomesList = [bestChromosomesList; {bestChromosome, maxRepresentativeFitness}];
    else
        bestChromosomesList = {bestChromosome, maxRepresentativeFitness};
    end

    % Encode the best chromosome
    bestChromosome = EncodeChromosome(bestChromosome, numberOfVariables, numberOfGenes);
end
