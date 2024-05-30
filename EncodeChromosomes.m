function chromosomes = EncodeChromosomes(decodedChromosomes, numberOfVariables, numberOfGenes)
    % decodedChromosomes: A matrix where each row represents the normalized PI values for each variable, in [0, 1] for a single chromosome
    % numberOfVariables: The total number of variables represented in a chromosome
    % numberOfGenes: The total length of a chromosome in bits
    
    numberOfChromosomes = size(decodedChromosomes, 1); % Number of chromosomes based on the input matrix rows
    numberOfBits = numberOfGenes / numberOfVariables; % Assuming equal distribution of bits per variable
    chromosomes = zeros(numberOfChromosomes, numberOfGenes); % Initialize matrix to hold all chromosomes
    
    for chromosomeIndex = 1:numberOfChromosomes
        decodedValues = decodedChromosomes(chromosomeIndex, :); % Extract decoded values for the current chromosome
        
        for variableIndex = 1:numberOfVariables
            % Scale the decoded value to the corresponding binary range
            scaledValue = round(decodedValues(variableIndex) * (2^numberOfBits - 1));
            
            % Convert the scaled value into a binary vector
            binaryVector = de2bi(scaledValue, numberOfBits, 'left-msb');
            
            % Ensure the binary vector has the correct length, in case de2bi produces a shorter vector
            binaryVector = [binaryVector, zeros(1, numberOfBits - length(binaryVector))];
            
            % Insert the binary vector into the correct position in the chromosome
            startPos = (variableIndex - 1) * numberOfBits + 1;
            endPos = startPos + numberOfBits - 1;
            chromosomes(chromosomeIndex, startPos:endPos) = binaryVector;
        end
    end
end
