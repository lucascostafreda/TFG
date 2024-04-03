%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%   Alp Sayin - alpsayin[at]alpsayin[dot]com - https://alpsayin.com
%   Matlab Genetic Algorithm
%   Spring 2012
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [fitnessValue, RB_usedAverage] = EvaluatePopulation(decodedPIs, r, available_RBs, iteration, averagenumber, ruche)
global KK;

    for index = 1:size(decodedPIs, 1) 
        KK=KK+1;
        fprintf('\n ------------- %d ------------- \n\n', KK);
        
        updated_PI = decodedPIs(index,:);
        resourceUsage = sum(r.*updated_PI);

        if resourceUsage > available_RBs 
            fitnessValue(index)= -15;
        else 
            [a, RB_usedAverage] = runFLEnviroment_RUCHE(updated_PI,iteration, averagenumber, r, ruche);
            %a = runFLEnviroment(updated_PI,iteration, averagenumber, r);
            fitnessValue(index)= -a ;
        end
    end
end