function chromosome = EncodeChromosome(decodedValues, numberOfVariables, numberOfGenes)
    % decodedValues: The normalized PI values for each variable, in [0, 1]
    % numberOfVariables: The total number of variables represented in the chromosome
    % numberOfGenes: The total length of the chromosome in bits
    
    numberOfBits = numberOfGenes / numberOfVariables; % Assuming equal distribution of bits per variable
    chromosome = zeros(1, numberOfGenes); % Initialize chromosome as a binary vector
    
    for index = 1:numberOfVariables
        % Scale the decoded value to the corresponding binary range
        scaledValue = round(decodedValues(index) * (2^numberOfBits - 1));
        
        % Convert the scaled value into a binary vector
        binaryVector = de2bi(scaledValue, numberOfBits, 'left-msb');
        
        % Insert the binary vector into the correct position in the chromosome
        startPos = (index - 1) * numberOfBits + 1;
        endPos = startPos + numberOfBits - 1;
        chromosome(startPos:endPos) = binaryVector;
    end
end
