% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % 
% %   Alp Sayin - alpsayin[at]alpsayin[dot]com - https://alpsayin.com
% %   Matlab Genetic Algorithm
% %   Spring 2012
% % 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% function probabilities = DecodePopulation(population, numberOfVariables, numberOfBits)
%     % Assume population is a matrix where each row is an individual
%     % and we're using 4 bits per variable, with numberOfVariables indicating
%     % how many probabilities each individual represents.
% 
%     populationSize = size(population, 1);
%     % numberOfBits = 4; % Fixed number of bits per probability
%     probabilities = zeros(populationSize, numberOfVariables);
% 
%     for index = 1:numberOfVariables
%         geneRangeStart = ((index-1) * numberOfBits) + 1;
%         geneRangeEnd = index * numberOfBits;
%         for i = 1:populationSize
%             individualBits = population(i, geneRangeStart:geneRangeEnd);
%             decimal = bi2de(individualBits, 'left-msb') / (2^numberOfBits - 1);
%             probabilities(i, index) = decimal;
%         end
%     end
% end

function probabilities = DecodePopulation(population, numberOfVariables, numberOfBits)
    populationSize = size(population, 1);
    probabilities = zeros(populationSize, numberOfVariables);

    for index = 1:numberOfVariables
        geneRangeStart = ((index-1) * numberOfBits) + 1;
        geneRangeEnd = index * numberOfBits;
        for i = 1:populationSize
            individualBits = population(i, geneRangeStart:geneRangeEnd);
            % Convert individualBits to an integer using the custom function
            individualInteger = binaryVectorToDecimal(individualBits, 'MSB');
            decimal = individualInteger / (2^numberOfBits - 1);
            probabilities(i, index) = decimal;
        end
    end
end

