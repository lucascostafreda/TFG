%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%   Alp Sayin - alpsayin[at]alpsayin[dot]com - https://alpsayin.com
%   Matlab Genetic Algorithm
%   Spring 2012
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% function x = DecodeChromosome (chromosome, numberOfVariables, variableRange)
% 
% x = zeros(1,numberOfVariables);
% numberOfBits = size(chromosome,2) / numberOfVariables;
% 
% for index = 1:numberOfVariables
%     x(index) = 0.0 ;
%     geneRangeStart = (((index-1)*numberOfBits)+1);
%     geneRangeEnd = index*numberOfBits;
%     x(index) = sum(chromosome( geneRangeStart:geneRangeEnd ).*(2.^-(1:numberOfBits)));
% %     % Deprecated - to be deleted in the next iteration
% %     for jj = 1:numberOfBits
% %         x(index) = x(index) + chromosome(jj + (index-1)*numberOfBits )*(2^(-jj)) ;
% %     end
%     x(index) = -variableRange + 2*variableRange*x(index)/(1-2^(-numberOfBits));
% end

function decodedValues = DecodeChromosome(chromosome, numberOfVariables)
    % Assume that 'chromosome' is a vector representing a single individual.
    % 'numberOfVariables' indicates how many variables the chromosome represents.
    % The chromosome is decoded into values within the [0, 1] range.

    numberOfBits = length(chromosome) / numberOfVariables;
    decodedValues = zeros(1, numberOfVariables);

    for index = 1:numberOfVariables
        geneRangeStart = ((index - 1) * numberOfBits) + 1;
        geneRangeEnd = index * numberOfBits;
        geneBits = chromosome(geneRangeStart:geneRangeEnd);
        % Convert the binary gene segment to a decimal value normalized to [0, 1]
        decimal = bi2de(geneBits, 'left-msb') / (2^numberOfBits - 1);
        decodedValues(index) = decimal;
    end
end