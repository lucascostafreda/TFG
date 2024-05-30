function [deviationFitness, accuracyFitness, RB_usedAverage, elapsedTime] = EvaluatePopulation_joint(decodedPIs, r, available_RBs, iteration,...
    averagenumber, ruche, verboseFL, miniBatchSize, executionEnviroment, AccDevMat, allDecodedPopulations, allDeviationValues, allAccuracyValues,...
    Shuffle, refModelName, directory_RefModel, directory_tempDir, historicalChromosomeFitness, historicalChromosomeAccuracy, chromosomeRepetitions, FragSDS, percentages, iGeneration)

global KK;
global KKK;

dupPI = 0;
excesPI = 0;
tic;  % Start timing the evaluation process

% Initialize output variables
deviationFitness = zeros(size(decodedPIs, 1), 1);
accuracyFitness = zeros(size(decodedPIs, 1), 1);
RB_usedAverage = zeros(size(decodedPIs, 1), 1);

for index = 1:size(decodedPIs, 1)
    previousFitness = 0;
    previousAccuracy = 0;
    updated_PI = decodedPIs(index,:); 
    piKey = sprintf('%.2f_', updated_PI); % Create a unique key for each chromosome
    piKey = piKey(1:end-1);  % Remove the trailing underscore

    % Check and update chromosome repetition count
    if isKey(chromosomeRepetitions, piKey)
        chromosomeRepetitions(piKey) = chromosomeRepetitions(piKey) + 1;
    else
        chromosomeRepetitions(piKey) = 1;       
    end

    % Check if there are existing fitness values to compare with
    fitnessValuesAvailable = ~isempty(allDeviationValues) && any(cellfun(@(c) ~isempty(c), allDeviationValues));

    found = false;
    if fitnessValuesAvailable
        % Iterate over all generations
        for iGen = iGeneration:-1:1

            if iGen == iGeneration && index == 1
                continue;
            end

                if iGen == iGeneration && index > 1
                  for iChrom = index-1:-1:1
                     if isequal(decodedPIs(iChrom, :), updated_PI) %this iteration
                        if chromosomeRepetitions(piKey) >= 3 
                            found = false;
                            chromosomeRepetitions(piKey) = 1; % Reset repetition counter
                            previousFitness = deviationFitness(iChrom);
                            previousAccuracy = accuracyFitness(iChrom); 
                            if ~isKey(historicalChromosomeFitness, piKey)
                                historicalChromosomeFitness(piKey)=previousFitness;
                            end
                            if ~isKey(historicalChromosomeAccuracy, piKey)
                                historicalChromosomeAccuracy(piKey) = previousAccuracy;
                            end
                        else  
                            found = true;
                            aux = deviationFitness(iChrom); %Coger los valores pasados
                            deviationFitness(index) = aux;
                            aux = accuracyFitness(iChrom);
                            accuracyFitness(index)= aux;
    
                            RB_usedAverage(index) = sum(r .* updated_PI);
                        end
                        dupPI = dupPI + 1;
                        fprintf('\n ------------- Chromosome is duplicated ------------- \n\n');
                        fprintf(' Duplication with Chromosome %d from Generation %d.\n', iChrom, iGen);
                        break; % Found a match, no need to search further % CHECKEAR!!
                     end
                  end
                  if found || previousFitness~=0
                        break; 
                  else
                        continue; %siguiente GEN
                  end
            end

            % posterior generations from here
            % Check previous generations
            prevPopulation = allDecodedPopulations{iGen}; 
            for iChrom = size(prevPopulation, 1):-1:1
                if isequal(prevPopulation(iChrom, :), updated_PI)
                    if chromosomeRepetitions(piKey) >= 3
                        found = false;
                        chromosomeRepetitions(piKey) = 1; % Reset repetition counter
                        previousFitness = allDeviationValues{iGen}(iChrom);
                        previousAccuracy = allAccuracyValues{iGen}(iChrom); 
                        if ~isKey(historicalChromosomeFitness, piKey)
                            historicalChromosomeFitness(piKey)=previousFitness;
                        end
                        if ~isKey(historicalChromosomeAccuracy, piKey)
                            historicalChromosomeAccuracy(piKey)=previousAccuracy;
                        end
                    else
                        found = true;
                        aux = allDeviationValues{iGen}(iChrom); % There is no generation 2 in the first one
                        deviationFitness(index) = aux;
                        aux = allAccuracyValues{iGen}(iChrom);
                        accuracyFitness(index) = aux;
                        RB_usedAverage(index) = sum(r .* updated_PI);
                    end
                    dupPI = dupPI + 1;
                    fprintf('\n ------------- Chromosome is duplicated ------------- \n\n');
                    fprintf(' Duplication with Chromosome %d from Generation %d.\n', iChrom, iGen);
                    break;
                end
            end
           
            if found || previousFitness~=0 || previousAccuracy~=0
                break;  % Exit the loop early if a match was found
            end % Salir de la busqeuda por las generation porque ya se ha encontrado
        end  % FOR Generation (iGeneration,1)
    end %allFintessValue no está vacío (no deja pasar 1ra ite)
    
    if not(found) 
        % Evaluar el cromosoma si no se encontró una coincidencia
        KKK = KKK + 1;
        resourceUsage = sum(r .* updated_PI);
    
        if resourceUsage > available_RBs
            fprintf('\n ------------- Chromosome surpasses the RB_available threshold ------------- \n\n');
            excesPI=excesPI+1;
            deviationFitness(index) = -15;
            accuracyFitness(index) = 0;
            continue;
        else
    
            KK = KK + 1;  
            % Evaluate the chromosome to obtain fitness and resource usage
            [a, accuracy, RB_used] = runFLEnviroment_RUCHE_noQ_sameDS(updated_PI, iteration, averagenumber, r, ruche, verboseFL, miniBatchSize, executionEnviroment, AccDevMat, Shuffle, refModelName, directory_RefModel,directory_tempDir, FragSDS, percentages);
            % a= round(6 + (8-6).*rand(1,1), 2);
            % accuracy = 0.40 + (0.70 - 0.40) * rand(1, 1);
            % 
            % RB_used=400;
            % 
            aux = -a;  % Assuming 'a' represents some form of fitness negativity
            auxAcc = accuracy;  % Direct use of accuracy as obtained from the function
        
            if previousFitness ~= 0|| previousAccuracy~=0
                % Update historical data if it exists
                historicalChromosomeFitness(piKey) = [historicalChromosomeFitness(piKey), aux];
                historicalChromosomeAccuracy(piKey) = [historicalChromosomeAccuracy(piKey), auxAcc];
        
                % Calculate and log historical values
                aux_historicalFitness = historicalChromosomeFitness(piKey);
                aux_historicalAccuracy = historicalChromosomeAccuracy(piKey);
                if numel(aux_historicalFitness) > 1
                    fprintf('Valores históricos (Fitness): ');  
                    fprintf('%.2f ', aux_historicalFitness); 
                    fprintf('\n');  
                    fprintf('Valores históricos (Accuracy): ');  
                    fprintf('%.2f ', aux_historicalAccuracy); 
                    fprintf('\n');  
                end
                aux = mean(aux_historicalFitness);
                auxAcc = mean(aux_historicalAccuracy);
                numRep = length(historicalChromosomeFitness(piKey));
                deviationFitness(index) = aux;
                accuracyFitness(index) = auxAcc;
        
                fprintf('Valor promediado (Fitness): %.2f, Veces promediado: %d \n', aux, numRep);
                fprintf('Valor promediado (Accuracy): %.2f, Veces promediado: %d \n', auxAcc, numRep);
                fprintf('PI: %s, ha sido repetido \n', piKey); 
        
            else
                deviationFitness(index) = aux;
                accuracyFitness(index) = auxAcc;
            end
            
            RB_usedAverage(index) = RB_used;  % Update resource usage
            
        end
    end
    fprintf('\n ------------- Total PI: %d , PI FL processed %d ------------- \n\n', KKK, KK); 
    fprintf('------------- Generation: %d, Duplicated PI  in Gen: %d , Excessed PI in Gen: %d ------------- \n\n', iGeneration, dupPI, excesPI);    

    end
    
    elapsedTime = toc;
end
