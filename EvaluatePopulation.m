%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%   Alp Sayin - alpsayin[at]alpsayin[dot]com - https://alpsayin.com
%   Matlab Genetic Algorithm
%   Spring 2012
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function fitnessValue = EvaluatePopulation(decodedPIs, runParallel, r, available_RBs, iteration, averagenumber)

if ~exist('runParallel','var')
    runParallel = false;
end

fitnessValue = zeros(size(decodedPIs,1),1);
% if runParallel
%     parfor index = 1:size(decodedPIs,1)
%         fitnessValue(index) = EvaluateIndividual(decodedPIs(index,:));
%     end
% else
%     for index = 1:size(x,1)
%         fitnessValue(index) = EvaluateIndividual(decodedPIs(index,:));
%     end
% end
if runParallel
    parfor index = 1:size(decodedPIs, 1)
        updated_PI = decodedPIs(index,:);
        % PI = [1,1, updated_PI, 1,1, updated_PI];
        PI = [updated_PI, updated_PI];
        resourceUsage = sum(r.*PI);
        
        if resourceUsage > available_RBs 
            fitnessValue(index)= -15;
        else 
            a = runFLEnviroment(PI,iteration, averagenumber);
            fitnessValue(index)= -a ;
        end
    end
else 
    for index = 1:size(decodedPIs, 1)
        updated_PI = decodedPIs(index,:);
        % PI = [1,1, updated_PI, 1,1, updated_PI];
        % PI = [updated_PI, updated_PI];
        resourceUsage = sum(r.*updated_PI);
        if resourceUsage > available_RBs 
            fitnessValue(index)= -15;
        else 
            a = runFLEnviroment(updated_PI,iteration, averagenumber);
            fitnessValue(index)= -a ;
        end
    end
end