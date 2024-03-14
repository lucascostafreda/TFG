%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%   Alp Sayin - alpsayin[at]alpsayin[dot]com - https://alpsayin.com
%   Matlab Genetic Algorithm
%   Spring 2012
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [population,decodedPopul, individualRBs, averageRBsInit] = InitializePopulation(populationSize, numberOfGenes, numberOfVariables, r, available_RBs)

% b1=[1,1];
% b2=[1,1];
% 
% population = (rand(populationSize, numberOfGenes)<0.5).*1;
% 
% return

 population = zeros(populationSize, numberOfGenes); % Pre-allocate for speed
 decodedPopul = zeros(populationSize, numberOfVariables);
 individualRBs = zeros(populationSize, 1);
 individualCount = 0;

    while individualCount < populationSize
        tempIndividual = (rand(1, numberOfGenes) < 0.5) .* 1;
        updatedPI = DecodeChromosome(tempIndividual, numberOfVariables);
        % Asumiendo que r es un vector con el mismo número de elementos que numberOfVariables
        % updated_PI = decodedIndividual(index,:);
        %PI=[1,1,updatedPI,1,1,updatedPI];
        PI = [updatedPI, updatedPI];
        temp = sum(r .* PI);
        if temp <= available_RBs && temp>available_RBs-20
            individualCount = individualCount + 1;
            population(individualCount, :) = tempIndividual;
            decodedPopul(individualCount, :) = updatedPI;
            individualRBs(individualCount)=sum(r .* PI);
            averageRBsInit = mean(individualRBs);
        end
    end
end

%% DEPRECATED - to be deleted in the next iteration

% population = zeros(populationSize, numberOfGenes);
% for ii = 1: populationSize
%     for jj = 1: numberOfGenes
%         s = rand;
%         if (s < 0.5)
%             population(ii,jj)=0;
%         else 
%             population(ii,jj)=1;
%         end
%     end
% end