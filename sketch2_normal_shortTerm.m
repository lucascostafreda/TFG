%% COMPARISON WITH BEST ESTIMATED PI

  CQI_indices = [12, 10, 8, 6, 12, 10, 8, 6];
  [r, eta, sigma] = parameters(CQI_indices);
  RB_usedMatrix = [];

iteration = 1;
averagenumber = 1;
verboseFL = true;
miniBatchSize = 100;
executionEnviroment ='parallel';
AccDevMat=true;
Shuffle='every-epoch';
long=false;

% 1 (1), 3/4 (2), 1/2 (3),  1/4 (4), 1/8 (5), 1/16 (6)s

%% RUCHE OR NOT
ruche = true;
if ruche
      if ~long
          refModelName = 'Ref_Model_15_i_15_r_noQ';
          directory_RefModel = 'refModel_15_i_15_withFragSDS';  
      else
          refModelName = 'Ref_Model_70_i_3_r_noQ';     
          directory_RefModel = 'refModel_70_i_3_avg'; 
      end  
      % baseDir = '/gpfs/workdir/costafrelu/Sketch.m/';  
else
      if ~long
          refModelName = 'Ref_Model_15_i_15_r_noQ';
          directory_RefModel = 'refModel_15_i_15_withFragSDS';  
      else
          refModelName = 'Ref_Model_70_i_3_r_noQ';     
          directory_RefModel = 'refModel_70_i_3_avg'; 
      end  
      % baseDir = fullfile('..', 'OptimizationOfPI', 'Opt_noOpt','WithAndWithoutDSFrag');
end

directory_tempDir = sprintf('0.9PI_100%_accuracyConvergens_i_%d_avg_%d', iteration, averagenumber); % para ACC y DEV este es el nombre de la ULTIMA CARPETA
FragSDS = 1;

percentages = [1,1,1,1,1,1,1,1];

chromosomes = [
    1, 1, 1, 1, 1, 1, 1, 1;
    0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9;
    0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8;
    0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6; 
];

for k = 1:size(percentages,1) 
    directory_tempDir = sprintf('%s_percentage_%d', directory_tempDir, k);
    for j = 1:size(chromosomes, 1)  % Iterate over each chromosome configuration
        currentChromosome = chromosomes(j, :);
        % filename = sprintf('fitness_%d_i_%d_av_chromosome%d.mat', iteration, averagenumber, j);
        % fullFilename = fullfile(baseDir, filename);
        [accuracyFitness, RB_used] = runFLEnviroment_RUCHE_sameDS_withDev(currentChromosome, iteration, averagenumber, r,...
            ruche, verboseFL, miniBatchSize, executionEnviroment, AccDevMat, Shuffle, refModelName, directory_RefModel, directory_tempDir, FragSDS, percentages, j, k);
    end
end
  