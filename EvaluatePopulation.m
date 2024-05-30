
function [fitnessValue, RB_usedAverage, elapsedTime] = EvaluatePopulation(decodedPIs, r, available_RBs, iteration,...
    averagenumber, ruche, verboseFL, miniBatchSize, executionEnviroment, AccDevMat, allDecodedPopulations, allFitnessValues,...
    Shuffle, refModelName, directory, historicalChromosomeFitness, chromosomeRepetitions, FragSDS, percentages  , iGeneration)

global KK;
global KKK;

dupPI=0;
excesPI=0;
tic;

% Inicialización de variables de salida
fitnessValue = zeros(size(decodedPIs, 1), 1);
RB_usedAverage = zeros(size(decodedPIs, 1), 1); % Modificado para mantener un valor por cromosoma

for index = 1:size(decodedPIs, 1)
    previousFitness=0;
    updated_PI = decodedPIs(index,:); 
    % piKey = mat2str(updated_PI);
    piKey = sprintf('%.2f_', updated_PI); % Concatena cada número con dos decimales seguido de un guión bajo
    piKey = piKey(1:end-1); % Elimina el último guión bajo
    found = false;
    
    % Incrementar el conteo de repeticiones para este cromosoma o inicializarlo si es nuevo
    if isKey(chromosomeRepetitions, piKey)
        chromosomeRepetitions(piKey) = chromosomeRepetitions(piKey) + 1;
    else
        chromosomeRepetitions(piKey) = 1;       
    end

    % Verificar si allFitnessValues tiene elementos y si esos elementos no están vacíos
    fitnessValuesAvailable = ~isempty(allFitnessValues) && any(cellfun(@(c) ~isempty(c), allFitnessValues));
    
    if fitnessValuesAvailable
        % Iterate over all generations
        for iGen = iGeneration:-1:1
            %SI primera Gen y no es el primer Chromosoma...

            if iGen==iGeneration && index==1% índice del chromosoma en cuestion dentro de GEN ACTUAL
                continue; %Pasamos a la sigueinte GEN, no tengo con que comparar
            end

            if iGen==iGeneration && index>1
                for iChrom = index-1:-1:1
                     if isequal(decodedPIs(iChrom, :), updated_PI)
                        if chromosomeRepetitions(piKey) >= 3
                            found = false; % No marcar como encontrado para proceder con la evaluación normal
                            chromosomeRepetitions(piKey) = 1; % Reiniciar el contador de repeticiones
                            previousFitness = fitnessValue(iChrom); %error aquí
                            if ~isKey(historicalChromosomeFitness, piKey)  
                                historicalChromosomeFitness(piKey) = previousFitness;
                            end   
                        else
                            found = true;
                            fitnessValue(index) = fitnessValue(iChrom);
                            dupPI = dupPI + 1;
                            RB_usedAverage(index) = sum(r .* updated_PI);  
                        end
                        fprintf('\n ------------- Chromosome is duplicated ------------- \n\n');
                        fprintf(' Duplication with Chromosome %d from Generation %d.\n', iChrom, iGen); 
                        break; % Found a match, no need to search further
                     end
                end
                % Si se encuentra salimos de la búsqueda
                % Si no, pasamos a la siguiente Gen
                if found || previousFitness~=0
                    break; 
                else
                    continue; %siguiente GEN
                end
            end
            
            
            % Aquí si que se puede ustilizar allFitnessValues
            prevPopulation = allDecodedPopulations{iGen}; 
            for iChrom = size(prevPopulation, 1):-1:1
                if isequal(prevPopulation(iChrom, :), updated_PI)
                    % Check repetition and manage fitness accordingly
                    if chromosomeRepetitions(piKey) >= 3
                        found = false; % No marcar como encontrado para proceder con la evaluación normal
                        chromosomeRepetitions(piKey) = 1; % Reiniciar el contador de repeticiones
                        previousFitness = allFitnessValues{iGen}(iChrom); %error aquí
                        if ~isKey(historicalChromosomeFitness, piKey)  
                            historicalChromosomeFitness(piKey) = previousFitness;
                        end   
                    else
                        found = true;
                        aux = allFitnessValues{iGen}(iChrom);
                        fitnessValue(index) = aux;
                        dupPI = dupPI + 1;
                        RB_usedAverage(index) = sum(r .* updated_PI);
                    end
                    fprintf('\n ------------- Chromosome is duplicated ------------- \n\n');
                    fprintf(' Duplication with Chromosome %d from Generation %d.\n', iChrom, iGen);
                    break;
                end
            end
            if found || previousFitness~=0
                break; 
            end % Salir de la busqeuda por las generation porque ya se ha encontrado
          
        end % FOR Generation (iGeneration,1)
    end %allFintessValue no está vacío (no deja pasar 1ra ite)
    
    if found
        continue; % Pasar al siguiente cromosoma sin reevaluar
    end
    
    % Evaluar el cromosoma si no se encontró una coincidencia
    KKK = KKK + 1;
    resourceUsage = sum(r .* updated_PI);

    if resourceUsage > available_RBs
        fprintf('\n ------------- Chromosome surpasses the RB_available threshold ------------- \n\n');
        excesPI=excesPI+1;
        fitnessValue(index) = -15;
    else
        KK = KK + 1;  
        % [a, RB_used] = runFLEnviroment_RUCHE(updated_PI, iteration, averagenumber, r, ruche, verboseFL, miniBatchSize, executionEnviroment, AccDevMat);
        [a, RB_used] = runFLEnviroment_RUCHE_noQ_sameDS(updated_PI, iteration, averagenumber, r, ruche, verboseFL, miniBatchSize, executionEnviroment,...
                AccDevMat, Shuffle, refModelName, directory, FragSDS, percentages);
        % a= round(6 + (8-6).*rand(1,1), 2);
        % RB_used=400;
        aux = -a;
        if previousFitness~=0
            
            historicalChromosomeFitness(piKey) = [historicalChromosomeFitness(piKey), aux]; 

            aux_historicalChromosomeFitness = historicalChromosomeFitness(piKey);
            if numel(aux_historicalChromosomeFitness) > 1
                fprintf('Valores históricos: ');  
                fprintf('%.2f ', aux_historicalChromosomeFitness); 
                fprintf('\n');  
            else
                fprintf('Valor histórico: %.2f\n', aux_historicalChromosomeFitness);  
            end
            aux = mean(aux_historicalChromosomeFitness);
            numRep = length(historicalChromosomeFitness(piKey));
            fitnessValue(index)=aux;

            fprintf('Valor promediado: %.2f, Veces promediado: %d \n', aux, numRep-1);
            fprintf('PI: %s, ha sido repetido \n', piKey); 
        else
            fitnessValue(index)=aux;
        end
        RB_usedAverage(index) = RB_used; % Asignar RB_used a este índice
    end

    fprintf('\n ------------- Total PI: %d , PI FL processed %d ------------- \n\n', KKK, KK); 
    fprintf('------------- Generation: %d, Duplicated PI  in Gen: %d , Excessed PI in Gen: %d ------------- \n\n', iGeneration, dupPI, excesPI);    

end
elapsedTime = toc;
end
